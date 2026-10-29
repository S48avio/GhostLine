"""Step 6b: the shared decoder. Beam search + the product rules.

Beam search keeps the `beam_size` most probable partial completions, extends
each by one token, and keeps the best `beam_size` again, until they finish.

Rules (same for every model):
    <unk> is never suggested            nobody can type "unknown" for you
    stop at <EOL>                       the line is finished
    stop BEFORE <STR> / <NUM>           we can't know which string or number goes there
    stop after max_len tokens

Finished completions are compared by their total log-probability INCLUDING the
token that ended them (<EOL> or the literal). So "open (" followed by a string
competes fairly with a full line: both are complete events.
"""

from dataclasses import dataclass

import numpy as np

from autocomplete.models.base import LanguageModel
from autocomplete.vocab import Vocab


@dataclass
class Completion:
    ids: list[int]  # suggested tokens; may end with <EOL>; empty = stay silent
    logprob: float  # log P(suggestion and the way it ended)

    @property
    def confidence(self) -> float:
        return float(np.exp(self.logprob))


def complete(model: LanguageModel, context: list[int], vocab: Vocab, beam_size: int = 4, max_len: int = 15) -> Completion:
    beams: list[tuple[list[int], float]] = [([], 0.0)]
    finished: list[tuple[list[int], float]] = []
    for _ in range(max_len):
        candidates = []
        for ids, lp in beams:
            logp = model.next_logprobs(context + ids).copy()
            logp[vocab.unk] = -np.inf  # rule: never suggest <unk>
            top = np.argpartition(-logp, beam_size)[:beam_size]
            candidates.extend((ids + [int(t)], lp + float(logp[t])) for t in top)
        candidates.sort(key=lambda c: -c[1])
        beams = []
        for ids, lp in candidates[:beam_size]:
            last = ids[-1]
            if last == vocab.eol:
                finished.append((ids, lp))  # keep <EOL>: "the line ends here" is part of the suggestion
            elif last in vocab.literals:
                finished.append((ids[:-1], lp))  # rule: stop before the literal
            else:
                beams.append((ids, lp))
        if not beams:
            break
    finished.extend(beams)  # hit max_len
    ids, lp = max(finished, key=lambda c: c[1])
    return Completion(ids, lp)


def complete_fast(model: LanguageModel, context: list[int], vocab: Vocab, beam_size: int = 4, max_len: int = 15) -> Completion:
    """Same answer as `complete`, less work:

    fix 1, early stop: a path's log-probability can only go DOWN as it grows
        (each step adds log p <= 0). Once a finished suggestion beats every
        running beam, none of them can ever catch up, so stop.
    fix 2, top-k only: beam search needs each beam's best `beam_size` next
        tokens, not all 20,000. Models with `top_next` provide exactly that.
    """
    top_next = getattr(model, "top_next", None)
    exclude = {vocab.unk}  # rule: never suggest <unk>
    beams: list[tuple[list[int], float]] = [([], 0.0)]
    finished: list[tuple[list[int], float]] = []
    for _ in range(max_len):
        candidates = []
        for ids, lp in beams:
            if top_next:
                toks, lps = top_next(context + ids, beam_size, exclude)
            else:
                logp = model.next_logprobs(context + ids).copy()
                logp[vocab.unk] = -np.inf
                toks = np.argpartition(-logp, beam_size)[:beam_size].tolist()
                lps = [float(logp[t]) for t in toks]
            candidates.extend((ids + [t], lp + l) for t, l in zip(toks, lps))
        candidates.sort(key=lambda c: -c[1])
        beams = []
        for ids, lp in candidates[:beam_size]:
            last = ids[-1]
            if last == vocab.eol:
                finished.append((ids, lp))
            elif last in vocab.literals:
                finished.append((ids[:-1], lp))
            else:
                beams.append((ids, lp))
        if not beams or (finished and max(lp for _, lp in beams) < max(lp for _, lp in finished)):
            break  # fix 1: nothing still running can win any more
    finished.extend(beams)
    ids, lp = max(finished, key=lambda c: c[1])
    return Completion(ids, lp)
