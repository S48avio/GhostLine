"""The two baselines every real model must beat. They also test the harness:
if these get absurd scores, the harness is wrong, not the model.

    unigram   the most common tokens, ignoring context
    bigram    the most common next token given the previous one (self -> .),
              mixed with the unigram so unseen pairs still get some probability:
              P(w | v) = LAMBDA * count(v, w) / count(v) + (1 - LAMBDA) * P_unigram(w)
"""

import numpy as np

from autocomplete.vocab import Vocab, read_ids

LAMBDA = 0.9


class Unigram:
    name = "unigram"
    context_size = 0

    def __init__(self, vocab: Vocab):
        ids, _ = read_ids("train", vocab)
        counts = np.bincount(ids, minlength=len(vocab)).astype(np.float64) + 1  # add-one: no zero probabilities
        self.logp = np.log(counts / counts.sum())

    def next_logprobs(self, context: list[int]) -> np.ndarray:
        return self.logp

    def stream_logprobs(self, ids: np.ndarray) -> np.ndarray:
        return self.logp[ids]


class Bigram:
    name = "bigram"
    context_size = 1

    def __init__(self, vocab: Vocab):
        self.v = len(vocab)
        ids, starts = read_ids("train", vocab)
        uni = np.bincount(ids, minlength=self.v).astype(np.float64) + 1
        self.p_uni = uni / uni.sum()
        # Pairs (prev, next) inside each file, encoded as one int64 so numpy can count them.
        first_of_file = np.zeros(len(ids), dtype=bool)
        first_of_file[starts[:-1]] = True
        prev, nxt = ids[:-1].astype(np.int64), ids[1:].astype(np.int64)
        keep = ~first_of_file[1:]  # don't pair the last token of one file with the first of the next
        codes, counts = np.unique(prev[keep] * self.v + nxt[keep], return_counts=True)
        self.codes, self.counts = codes, counts.astype(np.float64)
        self.prev_totals = np.bincount(codes // self.v, weights=self.counts, minlength=self.v)
        self._cache: dict[int, np.ndarray] = {}

    def next_logprobs(self, context: list[int]) -> np.ndarray:
        if not context:
            return np.log(self.p_uni)
        v = context[-1]
        if v not in self._cache:
            lo, hi = np.searchsorted(self.codes, [v * self.v, (v + 1) * self.v])
            p = (1 - LAMBDA) * self.p_uni
            if self.prev_totals[v] > 0:
                p = p.copy()
                p[self.codes[lo:hi] % self.v] += LAMBDA * self.counts[lo:hi] / self.prev_totals[v]
            else:
                p = self.p_uni
            self._cache[v] = np.log(p)
        return self._cache[v]

    def stream_logprobs(self, ids: np.ndarray) -> np.ndarray:
        ids = ids.astype(np.int64)
        out = np.log(self.p_uni[ids])
        if len(ids) > 1:
            prev, nxt = ids[:-1], ids[1:]
            pos = np.searchsorted(self.codes, prev * self.v + nxt)
            pos = np.minimum(pos, len(self.codes) - 1)
            pair = np.where(self.codes[pos] == prev * self.v + nxt, self.counts[pos], 0.0)
            totals = self.prev_totals[prev]
            p_bi = np.divide(pair, totals, out=np.zeros_like(pair), where=totals > 0)
            mix = np.where(totals > 0, LAMBDA * p_bi + (1 - LAMBDA) * self.p_uni[nxt], self.p_uni[nxt])
            out[1:] = np.log(mix)
        return out
