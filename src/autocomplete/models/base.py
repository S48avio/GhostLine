"""The one contract every model implements (baselines, KenLM, GRU).

Models only answer "what comes next?". Decoding, rules and metrics live in the
shared harness, so a score difference between models comes from the model,
never from different decoding code.
"""

from typing import Protocol

import numpy as np


class LanguageModel(Protocol):
    name: str
    context_size: int  # how many previous tokens the model can use (n-1 for an n-gram)

    def next_logprobs(self, context: list[int]) -> np.ndarray:
        """Natural-log probabilities over the whole vocabulary for the next token."""
        ...

    def stream_logprobs(self, ids: np.ndarray) -> np.ndarray:
        """log P(token_i | tokens before it in the same file), for every token of one file."""
        ...
