"""Step 5: turn Python source into normalized tokens with ONE shared vocabulary.

Both models (KenLM and the GRU) read exactly these tokens, so their scores,
including perplexity, are directly comparable.

Rules per Python token type:
    NAME, OP               the text itself ("def", "(", "self"), or <unk> if not in the vocabulary
    NEWLINE                <EOL>   end of a logical line: the stop signal for line completion
    INDENT / DEDENT        <INDENT> / <DEDENT>
    COMMENT, NL, ENCODING  dropped (prose and layout, not code)
    STRING                 kept if common, short and without whitespace ("utf-8"), else <STR>
    f-string (3.12 pieces) one <STR> for the whole f-string
    NUMBER                 kept if common, else <NUM>

Two passes, because normalizing needs the vocabulary:
    pass 1  count tokens in TRAIN (vocabulary) and VAL (only to measure OOV)
    pass 2  tokenize every split again, normalize, write data/tokens/<split>.txt
Pass 1 only counts: storing ~30M Python strings would need several GB of RAM.

Output, one source file per line (KenLM's input format):
    data/tokens/<split>.txt          space-separated tokens
    data/tokens/<split>.meta.jsonl   path, package, n_tokens for each line
    data/tokens/vocab.json

Usage:
    uv run python -m autocomplete.data.tokens --vocab-size 20000   # what we use
    uv run python -m autocomplete.data.tokens                      # V from the OOV table

V = 20,000: val OOV never flattens (11.9% at 5k -> 9.1% at 50k) because unseen
packages bring their own names ("localness of software", Tu et al. 2014). No
global vocabulary fixes that, so we took the size the GRU can afford.
"""

import argparse
import io
import json
import keyword
import tokenize
from collections import Counter
from multiprocessing import Pool

from autocomplete import config

SPECIALS = ["<unk>", "<STR>", "<NUM>", "<EOL>", "<INDENT>", "<DEDENT>"]
LITERAL_MIN_COUNT = 20  # a string/number must appear this often in train to be kept
LITERAL_MAX_LEN = 20
VOCAB_SIZES = [5_000, 10_000, 20_000, 50_000]
MIN_OOV_GAIN = 0.005  # grow V only while it cuts val OOV by at least 0.5 points
TOKENS = config.DATA / "tokens"
SPLIT_NAMES = ("train", "val", "test", "savio")

DROP = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER}


def raw_tokens(code: str, partial: bool = False) -> list[tuple[str, str]] | None:
    """(kind, text) pairs, kind in {"tok", "str", "num", "special"}. None if tokenize fails.

    partial=True is for code being typed ("for i in range("): the tokenizer
    raises at the unclosed bracket, so we keep what it read before the error.
    """
    out: list[tuple[str, str]] = []
    fstring_depth = 0  # Python 3.12 splits f-strings into pieces; skip everything inside
    try:
        for t in tokenize.generate_tokens(io.StringIO(code).readline):
            if t.type == tokenize.FSTRING_START:
                if fstring_depth == 0:
                    out.append(("special", "<STR>"))
                fstring_depth += 1
            elif t.type == tokenize.FSTRING_END:
                fstring_depth -= 1
            elif fstring_depth or t.type in DROP:
                continue
            elif t.type == tokenize.NEWLINE:
                out.append(("special", "<EOL>"))
            elif t.type == tokenize.INDENT:
                out.append(("special", "<INDENT>"))
            elif t.type == tokenize.DEDENT:
                out.append(("special", "<DEDENT>"))
            elif t.type == tokenize.STRING:
                out.append(("str", t.string))
            elif t.type == tokenize.NUMBER:
                out.append(("num", t.string))
            else:
                out.append(("tok", t.string))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        if not partial:
            return None
    return out


def literal_shape_ok(text: str) -> bool:
    """Could this string/number ever be kept? (Short, and no whitespace: KenLM splits on it.)"""
    return len(text) <= LITERAL_MAX_LEN and not any(ch.isspace() for ch in text)


def count_file(code: str) -> Counter | None:
    raw = raw_tokens(code)
    if raw is None:
        return None
    # Kind-prefixed keys keep e.g. the NAME `x` and a string `x` apart. Literals
    # that can never be kept are counted under their placeholder.
    c: Counter = Counter()
    for kind, text in raw:
        if kind in ("str", "num") and not literal_shape_ok(text):
            c[("special", "<STR>" if kind == "str" else "<NUM>")] += 1
        else:
            c[(kind, text)] += 1
    return c


def build_vocab(train: Counter, size: int) -> list[str]:
    """Specials first, then the most frequent tokens; literals need LITERAL_MIN_COUNT."""
    candidates = [
        (n, text)
        for (kind, text), n in train.items()
        if kind == "tok" or (kind in ("str", "num") and n >= LITERAL_MIN_COUNT)
    ]
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return SPECIALS + [text for _, text in candidates[: size - len(SPECIALS)]]


def oov_stats(counts: Counter, vocab: set[str]) -> dict[str, float]:
    """Share of tokens that would become <unk>, <STR>, <NUM> under this vocabulary."""
    total = sum(counts.values())
    unk = sum(n for (kind, text), n in counts.items() if kind == "tok" and text not in vocab)
    strs = sum(n for (kind, text), n in counts.items() if (kind == "str" and text not in vocab) or text == "<STR>")
    nums = sum(n for (kind, text), n in counts.items() if (kind == "num" and text not in vocab) or text == "<NUM>")
    return {"unk": unk / total, "str": strs / total, "num": nums / total}


def normalize(raw: list[tuple[str, str]], vocab: set[str]) -> list[str]:
    out = []
    for kind, text in raw:
        if kind == "special" or text in vocab:
            out.append(text)
        else:
            out.append({"tok": "<unk>", "str": "<STR>", "num": "<NUM>"}[kind])
    return out


_VOCAB: set[str] = set()


def _init_worker(vocab: set[str]) -> None:
    global _VOCAB
    _VOCAB = vocab


def tokenize_file(code: str) -> list[str] | None:
    raw = raw_tokens(code)
    return None if raw is None else normalize(raw, _VOCAB)


def load_split(name: str) -> list[dict]:
    return [json.loads(line) for line in (config.SPLITS / f"{name}.jsonl").open()]


def count_split(pool: Pool, records: list[dict]) -> tuple[Counter, int]:
    total: Counter = Counter()
    failed = 0
    for c in pool.imap_unordered(count_file, (r["code"] for r in records), chunksize=50):
        if c is None:
            failed += 1
        else:
            total.update(c)
    return total, failed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab-size", type=int, help="skip the automatic choice")
    args = ap.parse_args()

    splits = {name: load_split(name) for name in SPLIT_NAMES}

    # ---- pass 1: count -------------------------------------------------------
    with Pool(8) as pool:
        train_counts, train_failed = count_split(pool, splits["train"])
        val_counts, _ = count_split(pool, splits["val"])
    print(f"train: {sum(train_counts.values()):,} tokens, {len(train_counts):,} distinct ({train_failed} files failed to tokenize)")

    print("\nOOV on VAL by vocabulary size (vocabulary built from train only):")
    print(f"  {'V':>7s} {'<unk>':>7s} {'<STR>':>7s} {'<NUM>':>7s}")
    oov = {}
    for v in VOCAB_SIZES:
        oov[v] = oov_stats(val_counts, set(build_vocab(train_counts, v)))
        print(f"  {v:>7,} {oov[v]['unk']:7.2%} {oov[v]['str']:7.2%} {oov[v]['num']:7.2%}")

    # Smallest V after which growing the vocabulary stops paying off.
    size = args.vocab_size or next(
        (a for a, b in zip(VOCAB_SIZES, VOCAB_SIZES[1:]) if oov[a]["unk"] - oov[b]["unk"] < MIN_OOV_GAIN),
        VOCAB_SIZES[-1],
    )
    vocab = build_vocab(train_counts, size)
    print(f"chosen V = {size:,}" + ("" if args.vocab_size else f" (next size gains < {MIN_OOV_GAIN:.1%} OOV)"))

    # ---- pass 2: tokenize + normalize + write --------------------------------
    TOKENS.mkdir(parents=True, exist_ok=True)
    (TOKENS / "vocab.json").write_text(json.dumps({"size": size, "tokens": vocab}))
    vocab_set = set(vocab)
    print(f"\n{'split':6s} {'files':>7s} {'tokens':>12s} {'<unk>':>7s} {'<STR>':>7s} {'<NUM>':>7s} {'failed':>7s}")
    with Pool(8, initializer=_init_worker, initargs=(vocab_set,)) as pool:
        for name, records in splits.items():
            stats: Counter = Counter()
            failed = 0
            with (TOKENS / f"{name}.txt").open("w") as txt, (TOKENS / f"{name}.meta.jsonl").open("w") as meta:
                # imap (ordered) keeps line i of .txt aligned with line i of .meta.jsonl
                for rec, toks in zip(records, pool.imap(tokenize_file, (r["code"] for r in records), chunksize=50)):
                    if not toks:
                        failed += 1
                        continue
                    txt.write(" ".join(toks) + "\n")
                    meta.write(json.dumps({"path": rec["path"], "package": rec["package"], "n_tokens": len(toks)}) + "\n")
                    stats["total"] += len(toks)
                    for special in ("<unk>", "<STR>", "<NUM>"):
                        stats[special] += toks.count(special)
            t = stats["total"]
            print(f"{name:6s} {len(records) - failed:>7,} {t:>12,} {stats['<unk>'] / t:7.2%} {stats['<STR>'] / t:7.2%} {stats['<NUM>'] / t:7.2%} {failed:>7}")


# ---- for humans: tokens back to readable code (demos, debugging) -------------
NO_SPACE_BEFORE = {")", "]", "}", ",", ":", ".", ";"}
NO_SPACE_AFTER = {"(", "[", "{", ".", "~"}


def detokenize(tokens: list[str]) -> str:
    """Approximate code from tokens: `json . load ( f )` -> `json.load(f)`."""
    lines, line, depth = [], [], 0
    for tok in tokens:
        if tok == "<EOL>":
            lines.append("    " * depth + _join(line))
            line = []
        elif tok == "<INDENT>":
            depth += 1
        elif tok == "<DEDENT>":
            depth = max(depth - 1, 0)
        else:
            line.append(tok)
    if line:
        lines.append("    " * depth + _join(line))
    return "\n".join(lines)


def _join(toks: list[str]) -> str:
    out = ""
    for i, tok in enumerate(toks):
        prev = toks[i - 1] if i else ""
        # Calls and indexing hug their name: `load(f)`, `x[0]`, but `if (a)` keeps its space.
        call = tok in ("(", "[") and (prev in (")", "]", "<unk>", "<STR>") or (prev.isidentifier() and not keyword.iskeyword(prev)))
        if i and tok not in NO_SPACE_BEFORE and prev not in NO_SPACE_AFTER and not call:
            out += " "
        out += tok
    return out


if __name__ == "__main__":
    main()
