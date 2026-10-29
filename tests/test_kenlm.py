"""KenLM speed tricks must be EXACT. Skipped until the models are built."""

import numpy as np
import pytest

from autocomplete.models.kenlm_model import model_path

pytestmark = pytest.mark.skipif(not model_path(5).exists(), reason="build models/kenlm-5.bin first")

CONTEXTS = [
    ["for", "i", "in", "range", "("],
    ["self", "."],
    [":", "<EOL>", "<INDENT>"],
    ["import", "numpy", "as"],
    ["return", "<unk>", "("],
]


@pytest.fixture(scope="module")
def models():
    from autocomplete.models.kenlm_model import KenLM
    from autocomplete.vocab import Vocab

    vocab = Vocab.load()
    return vocab, KenLM(vocab, 5, fast=True), KenLM(vocab, 5, fast=False)


def test_distribution_sums_to_one(models):
    vocab, fast, _ = models
    assert np.exp(fast.next_logprobs(vocab.encode(CONTEXTS[0]))).sum() == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("ctx", CONTEXTS)
def test_backoff_trick_equals_brute_force(models, ctx):
    vocab, fast, slow = models
    ids = vocab.encode(ctx)
    assert np.allclose(fast._fast(ids), slow._brute(ids), atol=1e-4)


@pytest.mark.parametrize("ctx", CONTEXTS)
def test_top_next_equals_full_ranking(models, ctx):
    vocab, fast, _ = models
    ids = vocab.encode(ctx)
    full = fast.next_logprobs(ids).copy()
    full[vocab.unk] = -np.inf
    expected = np.argsort(-full)[:8].tolist()
    got, lps = fast.top_next(ids, 8, {vocab.unk}, use_table=False)
    assert got == expected
    assert np.allclose(lps, full[got], atol=1e-4)
