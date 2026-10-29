"""Phase 1: n-gram language models with KenLM (modified Kneser-Ney smoothing).

Build:  data/tokens/train.txt --lmplz--> models/kenlm-N.arpa --build_binary--> models/kenlm-N.bin

Two details that are easy to get wrong:
    <unk>   lmplz refuses it in training text (KenLM reserves the word). Its own
            fix, --skip_symbols, deletes the token and glues its neighbours
            together. We rename it to <oov>, an ordinary word, so KenLM learns
            how likely "some unknown name" is in each context, as the GRU will.
    log10   KenLM scores are log base 10; the harness uses natural log.

Usage:
    uv run python -m autocomplete.models.kenlm_model --order 5      # build one model
"""

import argparse
import math
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from autocomplete.vocab import TOKENS, Vocab, read_ids

KENLM_BIN = Path("third_party/kenlm/build/bin")
MODELS = Path("models")
LN10 = math.log(10)
OOV = "<oov>"
TABLE_SIZE = 5_000  # precomputed contexts (speed fix 4)
TABLE_MIN_SUCCESSORS = 300  # only contexts whose last pair had this many different successors
TABLE_K = 8  # answers stored per context: enough for beam sizes up to 8


def model_path(order: int) -> Path:
    return MODELS / f"kenlm-{order}.bin"


def build(order: int) -> None:
    MODELS.mkdir(exist_ok=True)
    arpa = MODELS / f"kenlm-{order}.arpa"
    tmp = MODELS / "tmp"
    tmp.mkdir(exist_ok=True)
    t = time.time()
    # Stream train tokens into lmplz, renaming <unk> on the way. -S caps lmplz's
    # sorting memory so the laptop stays usable.
    with (TOKENS / "train.txt").open() as src, arpa.open("w") as out:
        lmplz = subprocess.Popen(
            [str(KENLM_BIN / "lmplz"), "-o", str(order), "-S", "40%", "-T", str(tmp), "--discount_fallback"],
            stdin=subprocess.PIPE, stdout=out, stderr=subprocess.PIPE, text=True,
        )
        for line in src:
            lmplz.stdin.write(line.replace("<unk>", OOV))
        lmplz.stdin.close()
        err = lmplz.stderr.read()
        if lmplz.wait() != 0:
            raise RuntimeError(err[-2000:])
    built = time.time() - t
    subprocess.run([str(KENLM_BIN / "build_binary"), "trie", str(arpa), str(model_path(order))], check=True, capture_output=True)
    arpa_mb = arpa.stat().st_size / 1e6
    arpa.unlink()  # the binary is all we need; ARPA text can be several GB
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"kenlm-{order}: lmplz {built:.0f}s, ARPA {arpa_mb:,.0f} MB -> binary {model_path(order).stat().st_size / 1e6:,.0f} MB")


class KenLM:
    """Exact next-token distributions, fast.

    For a token w that never FOLLOWED the last context token in training, no
    n-gram containing w applies, so KenLM backs off to w's unigram:
        log P(w | ctx) = [sum of the context's back-off weights] + log P_unigram(w)
    The bracket is the same for every such w. So we score only the last
    token's successors exactly (hundreds, not 20,000), probe one non-successor
    to learn the bracket, and fill in the rest. Exact, not an approximation;
    `fast=False` keeps the brute-force path to verify that.
    """

    def __init__(self, vocab: Vocab, order: int = 5, fast: bool = True):
        import kenlm

        self.name = f"kenlm-{order}"
        self.order = order
        self.context_size = order - 1  # an n-gram model only sees the last n-1 tokens
        self.model = kenlm.Model(str(model_path(order)))
        self._kenlm = kenlm
        self.words = [OOV if t == "<unk>" else t for t in vocab.tokens]
        self.fast = fast
        self._cache: dict[tuple[int, ...], np.ndarray] = {}
        if fast:
            self._unigram = self._brute([]) / LN10  # log10 P_unigram(w), null context
            self._successors = successors(vocab)
            self._rare_first = np.argsort(self._unigram)  # probe candidates: rare tokens first

    def _state(self, context: list[int]):
        """KenLM state after reading the context (no <s>: a cursor is mid-file)."""
        state, out = self._kenlm.State(), self._kenlm.State()
        self.model.NullContextWrite(state)
        for i in context[-self.context_size :]:
            self.model.BaseScore(state, self.words[i], out)
            state, out = out, state
        return state

    def _brute(self, context: list[int]) -> np.ndarray:
        state, out = self._state(context), self._kenlm.State()
        score = self.model.BaseScore
        return np.array([score(state, w, out) for w in self.words]) * LN10

    def _fast(self, context: list[int]) -> np.ndarray:
        if not context:
            return self._unigram * LN10
        state, out = self._state(context), self._kenlm.State()
        score = self.model.BaseScore
        succ = self._successors.get(context[-1], np.empty(0, dtype=np.int64))
        is_succ = np.zeros(len(self.words), dtype=bool)
        is_succ[succ] = True
        probe = next(int(i) for i in self._rare_first if not is_succ[i])
        backoff = score(state, self.words[probe], out) - self._unigram[probe]
        logp = self._unigram + backoff
        logp[succ] = [score(state, self.words[i], out) for i in succ]
        return logp * LN10

    def next_logprobs(self, context: list[int]) -> np.ndarray:
        key = tuple(context[-self.context_size :])
        if key not in self._cache:
            self._cache[key] = self._fast(list(key)) if self.fast else self._brute(list(key))
        return self._cache[key]

    # ---- speed fix 2: exact top-k without the full 20,000-token distribution ----
    #
    # If w never followed the PAIR (a, b) in training, no 3/4/5-gram with w
    # applies, so  log P(w | ..., a, b) = C + log P(w | b)  with C the same for
    # every such w. So: score the pair's successors exactly (usually a handful),
    # learn C from one probe, and take the best of the rest from a list of
    # bigram scores sorted once at startup. The top-k is exact, not approximate.

    def _prepare_topk(self) -> None:
        if hasattr(self, "_tri_codes"):
            return
        t = time.time()
        v = len(self.words)
        vocab = Vocab([("<unk>" if w == OOV else w) for w in self.words])
        ids, starts = read_ids("train", vocab)
        ok = np.ones(len(ids), dtype=bool)
        ok[starts[:-1]] = False  # first token of a file: no previous token
        ok[np.minimum(starts[:-1] + 1, len(ids) - 1)] = False  # second token: no pair before it
        pos = np.nonzero(ok[2:])[0] + 2
        a, b, c = (ids[pos - 2].astype(np.int64), ids[pos - 1].astype(np.int64), ids[pos].astype(np.int64))
        self._tri_codes = np.unique((a * v + b) * v + c)
        # Bigram level: P(w | b) for every successor w of b, sorted best first,
        # and b's back-off for everything else: log P(w | b) = bo[b] + log P_unigram(w).
        state, out = self._kenlm.State(), self._kenlm.State()
        null = self._kenlm.State()
        self.model.NullContextWrite(null)
        self._bi: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._bo = np.zeros(v)
        for bid, succ in self._successors.items():
            self.model.BaseScore(null, self.words[bid], state)
            lp = np.array([self.model.BaseScore(state, self.words[w], out) for w in succ])
            order = np.argsort(-lp)
            self._bi[bid] = (succ[order], lp[order])
            is_succ = np.zeros(v, dtype=bool)
            is_succ[succ] = True
            probe = next(int(i) for i in self._rare_first if not is_succ[i])
            self._bo[bid] = self.model.BaseScore(state, self.words[probe], out) - self._unigram[probe]
        self._uni_desc = self._rare_first[::-1]
        print(f"[{self.name}] top-k tables ready in {time.time() - t:.0f}s ({len(self._tri_codes):,} trigrams)")

    # ---- speed fix 4: precomputed answers for the frequent "open" contexts ----
    #
    # top_next is slow exactly when many different tokens followed the last pair
    # (`self .`, `<EOL> <INDENT>`): it must score all of them. Those contexts are
    # also among the most FREQUENT in code, so we answer them once, offline, and
    # store the result. Values come from top_next itself, so they are exact.

    def _table_path(self) -> Path:
        return MODELS / f"{self.name}.topk.npz"

    def build_table(self, size: int = TABLE_SIZE, min_successors: int = TABLE_MIN_SUCCESSORS) -> None:
        self._prepare_topk()
        t = time.time()
        v, n = len(self.words), self.context_size
        vocab = Vocab([("<unk>" if w == OOV else w) for w in self.words])
        ids, starts = read_ids("train", vocab)
        file_of = np.searchsorted(starts, np.arange(len(ids)), side="right") - 1
        # Every window of n consecutive tokens inside one file is a context that was followed by something.
        pos = np.arange(n - 1, len(ids) - 1)
        pos = pos[file_of[pos - n + 1] == file_of[pos + 1]]
        # Count identical rows of n token ids. (Packing them into one int64 would
        # overflow for n = 5: 20,000^5 is far above 2^63.)
        windows = np.ascontiguousarray(np.stack([ids[pos - n + 1 + j] for j in range(n)], axis=1))
        rows = windows.view(np.dtype((np.void, windows.dtype.itemsize * n))).ravel()
        uniq, counts = np.unique(rows, return_counts=True)
        contexts = uniq.view(np.int32).reshape(-1, n)
        keys, tok, lps = [], [], []
        for row in np.argsort(-counts):
            ctx = contexts[row].tolist()
            a, b = ctx[-2], ctx[-1]
            lo, hi = np.searchsorted(self._tri_codes, [(a * v + b) * v, (a * v + b + 1) * v])
            if hi - lo < min_successors:
                continue  # already fast: not worth a table entry
            top_ids, top_lps = self.top_next(ctx, TABLE_K, {vocab.unk}, use_table=False)
            keys.append(ctx)
            tok.append(top_ids)
            lps.append(top_lps)
            if len(keys) == size:
                break
        np.savez(self._table_path(), keys=np.array(keys, dtype=np.int32), ids=np.array(tok, dtype=np.int32), lps=np.array(lps))
        print(f"[{self.name}] precomputed {len(keys):,} contexts in {time.time() - t:.0f}s -> {self._table_path()}")

    def _load_table(self) -> dict[tuple[int, ...], tuple[list[int], list[float]]]:
        if not hasattr(self, "_table"):
            self._table = {}
            if self._table_path().exists():
                d = np.load(self._table_path())
                self._table = {tuple(k.tolist()): (i.tolist(), lp.tolist()) for k, i, lp in zip(d["keys"], d["ids"], d["lps"])}
        return self._table

    def top_next(self, context: list[int], k: int, exclude: set[int], use_table: bool = True) -> tuple[list[int], list[float]]:
        """The k most probable next tokens (natural log), never one from `exclude`."""
        ctx = context[-self.context_size :]
        if use_table and k <= TABLE_K and len(exclude) == 1 and self.words[next(iter(exclude))] == OOV:
            hit = self._load_table().get(tuple(ctx))
            if hit:
                return hit[0][:k], hit[1][:k]  # stored best-first, so the first k ARE the top k
        if len(ctx) < 2:
            logp = self.next_logprobs(ctx).copy()
            logp[list(exclude)] = -np.inf
            top = np.argpartition(-logp, k)[:k]
            top = top[np.argsort(-logp[top])]  # best first, like every other path
            return [int(i) for i in top], [float(logp[i]) for i in top]
        self._prepare_topk()
        v = len(self.words)
        a, b = ctx[-2], ctx[-1]
        score = self.model.BaseScore
        state, out = self._state(ctx), self._kenlm.State()

        lo, hi = np.searchsorted(self._tri_codes, [(a * v + b) * v, (a * v + b + 1) * v])
        pair_succ = (self._tri_codes[lo:hi] % v).astype(np.int64)
        in_pair = set(pair_succ.tolist())
        cands = {int(w): score(state, self.words[w], out) for w in pair_succ if w not in exclude}

        # C from one probe that never followed (a, b): full score minus its bigram-level score.
        state_b, null = self._kenlm.State(), self._kenlm.State()
        self.model.NullContextWrite(null)
        score(null, self.words[b], state_b)
        probe = next(int(i) for i in self._rare_first if int(i) not in in_pair)
        c = score(state, self.words[probe], out) - score(state_b, self.words[probe], out)

        # The best k of b's other successors (already sorted by P(w | b)) ...
        succ_ids, succ_lp = self._bi.get(b, (np.empty(0, dtype=np.int64), np.empty(0)))
        taken = 0
        for w, lp in zip(succ_ids.tolist(), succ_lp.tolist()):
            if w not in in_pair and w not in exclude:
                cands[w] = c + lp
                taken += 1
                if taken == k:
                    break
        # ... and the best k tokens that never followed b at all (unigram order).
        is_succ_b = set(succ_ids.tolist())
        taken = 0
        for w in self._uni_desc.tolist():
            if w not in is_succ_b and w not in in_pair and w not in exclude:
                cands[w] = c + self._bo[b] + self._unigram[w]
                taken += 1
                if taken == k:
                    break
        best = sorted(cands.items(), key=lambda x: -x[1])[:k]
        return [w for w, _ in best], [lp * LN10 for _, lp in best]  # best first

    def stream_logprobs(self, ids: np.ndarray) -> np.ndarray:
        sentence = " ".join(self.words[i] for i in ids)
        # bos=True: KenLM was trained with <s> at the start of every file.
        scores = [s for s, _, _ in self.model.full_scores(sentence, bos=True, eos=False)]
        return np.array(scores[: len(ids)]) * LN10


def successors(vocab: Vocab) -> dict[int, np.ndarray]:
    """For each token: every token that ever followed it in train (same files KenLM read)."""
    ids, starts = read_ids("train", vocab)
    first = np.zeros(len(ids), dtype=bool)
    first[starts[:-1]] = True
    keep = ~first[1:]
    v = len(vocab)
    codes = np.unique(ids[:-1][keep].astype(np.int64) * v + ids[1:][keep])
    prev, nxt = codes // v, codes % v
    bounds = np.searchsorted(prev, np.arange(v + 1))
    return {p: nxt[bounds[p] : bounds[p + 1]] for p in range(v) if bounds[p + 1] > bounds[p]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, nargs="+", default=[5])
    args = ap.parse_args()
    for order in args.order:
        build(order)


if __name__ == "__main__":
    main()
