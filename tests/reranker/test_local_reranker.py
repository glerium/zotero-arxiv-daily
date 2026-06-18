"""Tests for LocalReranker — requires sentence-transformers, marked slow."""

import numpy as np
import pytest

from zotero_arxiv_daily.reranker.local import LocalReranker


@pytest.mark.slow
def test_local_reranker(config):
    reranker = LocalReranker(config)
    score = reranker.get_similarity_score(["hello", "world"], ["ping"])
    assert score.shape == (2, 1)


def test_local_reranker_fallback_on_model_error(config, monkeypatch):
    import sentence_transformers

    def _raise(*args, **kwargs):
        raise OSError("mock model download failure")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _raise)
    reranker = LocalReranker(config)
    score = reranker.get_similarity_score(["hello", "world"], ["ping"])
    assert score.shape == (2, 1)
    assert np.array_equal(score, np.zeros((2, 1)))
