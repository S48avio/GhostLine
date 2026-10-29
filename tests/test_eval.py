"""The harness rules, checked on a toy vocabulary and a hand-made model."""

import numpy as np

from autocomplete.data.tokens import detokenize, raw_tokens
from autocomplete.eval.cases import family, logical_lines
from autocomplete.eval.decode import complete, complete_fast
from autocomplete.eval.metrics import edit_similarity, suggestible
from autocomplete.vocab import Vocab

VOCAB = Vocab(["<unk>", "<STR>", "<NUM>", "<EOL>", "<INDENT>", "<DEDENT>", "print", "(", ")", "x"])
I = VOCAB.index


class TableModel:
    """next token = a fixed table on the last token; mostly-certain predictions."""

    name, context_size = "table", 1

    def __init__(self, table: dict[str, str], default: str = ")"):
        self.table, self.default = table, default

    def next_logprobs(self, context):
        last = VOCAB.tokens[context[-1]] if context else "<start>"
        p = np.full(len(VOCAB), 0.01)
        p[I[self.table.get(last, self.default)]] = 1.0
        return np.log(p / p.sum())


def test_beam_search_completes_the_line():
    model = TableModel({"print": "(", "(": "x", "x": ")", ")": "<EOL>"})
    comp = complete(model, [I["print"]], VOCAB)
    assert VOCAB.decode(comp.ids) == ["(", "x", ")", "<EOL>"]
    assert 0 < comp.confidence <= 1


def test_never_suggests_unk():
    model = TableModel({"(": "<unk>"})
    comp = complete(model, [I["("]], VOCAB)
    assert I["<unk>"] not in comp.ids


def test_stops_before_a_literal():
    model = TableModel({"print": "(", "(": "<STR>"})
    comp = complete(model, [I["print"]], VOCAB)
    assert VOCAB.decode(comp.ids) == ["("]


def test_stays_silent_when_a_literal_comes_first():
    model = TableModel({"(": "<STR>"})
    assert complete(model, [I["("]], VOCAB).ids == []


def test_fast_decoder_gives_the_same_answer():
    """Early stop + top-k must never change the suggestion, only the work done."""
    rng = np.random.default_rng(0)

    class RandomModel:
        name, context_size = "random", 2
        tables: dict = {}

        def next_logprobs(self, context):
            key = tuple(context[-2:])
            if key not in self.tables:
                p = rng.dirichlet(np.full(len(VOCAB), 0.3))
                self.tables[key] = np.log(p)
            return self.tables[key]

    model = RandomModel()
    for start in range(len(VOCAB)):
        for beam in (1, 2, 4):
            a = complete(model, [start], VOCAB, beam_size=beam)
            b = complete_fast(model, [start], VOCAB, beam_size=beam)
            assert a.ids == b.ids and abs(a.logprob - b.logprob) < 1e-9


def test_suggestible_cuts_at_first_placeholder():
    assert suggestible(["open", "(", "<STR>", ")", "<EOL>"]) == ["open", "("]
    assert suggestible(["<unk>", "(", ")"]) == []
    assert suggestible(["x", ")", "<EOL>"]) == ["x", ")", "<EOL>"]


def test_edit_similarity():
    assert edit_similarity(["a", "b"], ["a", "b"]) == 1.0
    assert edit_similarity([], []) == 1.0
    assert edit_similarity(["a", "b"], ["a", "c"]) == 0.5


def test_logical_lines_skip_block_markers():
    toks = ["if", "x", ":", "<EOL>", "<INDENT>", "print", "(", "x", ")", "<EOL>", "<DEDENT>"]
    assert [toks[s:e] for s, e in logical_lines(toks)] == [
        ["if", "x", ":", "<EOL>"],
        ["print", "(", "x", ")", "<EOL>"],
    ]


def test_family():
    assert family("pydantic_core") == family("pydantic") == "pydantic"
    assert family("_pytest") == "pytest"


def test_tokenizer_drops_comments_and_collapses_fstrings():
    kinds = raw_tokens('x = f"a{b}c"  # note\n')
    assert ("special", "<STR>") in kinds
    assert all("note" not in text for _, text in kinds)
    assert detokenize(["print", "(", "x", ")", "<EOL>"]) == "print(x)"
