"""The shared vocabulary (built in step 5) and readers for the token files."""

import json
from array import array

import numpy as np

from autocomplete import config

TOKENS = config.DATA / "tokens"


class Vocab:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.index = {t: i for i, t in enumerate(tokens)}
        self.unk = self.index["<unk>"]
        self.eol = self.index["<EOL>"]
        self.literals = {self.index["<STR>"], self.index["<NUM>"]}

    @classmethod
    def load(cls) -> "Vocab":
        return cls(json.loads((TOKENS / "vocab.json").read_text())["tokens"])

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.index[t] for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.tokens[i] for i in ids]


def read_files(split: str) -> list[list[str]]:
    """One token list per source file. Fine for val/test (~3M tokens)."""
    with (TOKENS / f"{split}.txt").open() as f:
        return [line.split() for line in f]


def read_meta(split: str) -> list[dict]:
    with (TOKENS / f"{split}.meta.jsonl").open() as f:
        return [json.loads(line) for line in f]


def read_ids(split: str, vocab: Vocab) -> tuple[np.ndarray, np.ndarray]:
    """All files as one int32 stream + the offset where each file starts.

    Streams line by line into a compact array: holding 22M Python strings for
    train would need over a gigabyte of RAM; 22M int32s need 88 MB.
    """
    ids = array("i")
    starts = []
    with (TOKENS / f"{split}.txt").open() as f:
        for line in f:
            starts.append(len(ids))
            ids.extend(vocab.index[t] for t in line.split())
    return np.frombuffer(ids, dtype=np.int32), np.array(starts + [len(ids)], dtype=np.int64)
