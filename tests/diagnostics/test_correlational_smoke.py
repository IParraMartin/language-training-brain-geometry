"""Smoke tests for the trainedness correlation (brain-correlate).

All tests are CPU-only with no HuggingFace downloads. Synthetic brain
scores, frequency tables, and token sequences stand in for real data.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from brain_code.diagnostics.correlational_analysis import (
    compute_embedding_trainedness,
    correlational_analysis
)


def _brain_scores(corpora: list[str]) -> dict[str, dict]:
    """Synthetic brain scores with strictly increasing values per corpus."""
    return {c: {"overall": 0.05 * (i + 1)} for i, c in enumerate(corpora)}


def _freq_table(vocab_size: int, seed: int = 0) -> Counter:
    """Uniform-ish random frequency table over a small vocabulary."""
    rng = np.random.default_rng(seed)
    return Counter({i: int(c) for i, c in enumerate(rng.integers(1, 1000, vocab_size))})


def _token_seqs(n: int = 8, length: int = 5, vocab_size: int = 28) -> list[list[int]]:
    """Random token-ID sequences simulating stimulus tokenisation."""
    rng = np.random.default_rng(seed=1)
    return [list(rng.integers(0, vocab_size, length)) for _ in range(n)]


class TestCorrelationalAnalysis:
    """Tests for the pure-computation regression function."""

    def test_runs_with_three_corpora(self, tmp_path):
        corpora = ["alpha", "beta", "gamma"]
        result = correlational_analysis(
            brain_scores=_brain_scores(corpora),
            freq_tables={c: _freq_table(28, seed=i) for i, c in enumerate(corpora)},
            stimulus_token_ids=_token_seqs(),
            out_path=tmp_path / "result.json"
        )
        assert isinstance(result, dict)
        assert "across_models" in result

    def test_writes_valid_json(self, tmp_path):
        corpora = ["alpha", "beta", "gamma"]
        out = tmp_path / "out.json"
        correlational_analysis(
            brain_scores=_brain_scores(corpora),
            freq_tables={c: _freq_table(28, seed=i) for i, c in enumerate(corpora)},
            stimulus_token_ids=_token_seqs(),
            out_path=out
        )
        assert out.exists()
        data = json.loads(out.read_text())
        assert "across_models" in data

    def test_regression_fields_present(self, tmp_path):
        corpora = ["alpha", "beta", "gamma", "delta", "epsilon"]
        result = correlational_analysis(
            brain_scores=_brain_scores(corpora),
            freq_tables={c: _freq_table(28, seed=i) for i, c in enumerate(corpora)},
            stimulus_token_ids=_token_seqs(),
            out_path=tmp_path / "r.json"
        )
        am = result["across_models"]
        for key in ("slope", "intercept", "r", "r_squared", "p_value"):
            assert key in am, f"Missing key: {key}"

    def test_extra_corpus_in_freq_table_is_ignored(self, tmp_path):
        """A corpus in freq_tables but not brain_scores should not cause an error."""
        brain_scores = {"alpha": {"overall": 0.1}, "beta": {"overall": 0.2}}
        freq_tables = {
            "alpha": _freq_table(28, seed=0),
            "beta": _freq_table(28, seed=1),
            "gamma": _freq_table(28, seed=2)  # not in brain_scores
        }
        result = correlational_analysis(
            brain_scores=brain_scores,
            freq_tables=freq_tables,
            stimulus_token_ids=_token_seqs(),
            out_path=tmp_path / "r.json"
        )
        assert isinstance(result, dict)

    def test_single_corpus_returns_empty_regression(self, tmp_path):
        """With fewer than 2 corpora the regression is skipped, not raised."""
        result = correlational_analysis(
            brain_scores={"solo": {"overall": 0.1}},
            freq_tables={"solo": _freq_table(28)},
            stimulus_token_ids=_token_seqs(),
            out_path=tmp_path / "r.json"
        )
        # across_models should be an empty dict (no regression possible)
        assert result["across_models"] == {}


class TestComputeEmbeddingTrainedness:
    """Tests using a local saved micro model — no HuggingFace downloads."""

    def test_returns_dict(
        self,
        micro_model_config,
        tiny_tokenizer,
        tmp_path
    ):
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(micro_model_config)
        model_path = tmp_path / "micro"
        model.save_pretrained(str(model_path))
        tiny_tokenizer.save_pretrained(str(model_path))

        result = compute_embedding_trainedness(
            corpus_name="test",
            model_id=str(model_path),
            freq_table=_freq_table(micro_model_config.vocab_size),
            stimulus_token_ids=_token_seqs(vocab_size=micro_model_config.vocab_size),
            stimulus_sentences=[f"s{i}" for i in range(8)],
            out_dir=tmp_path / "diag"
        )
        assert isinstance(result, dict)

    def test_with_init_embeddings_path(
        self,
        micro_model_config,
        tiny_tokenizer,
        tmp_path
    ):
        """Passing init_embeddings as a .npy file path does not crash.

        Note: the parameter is typed Optional[np.ndarray] but the implementation
        calls np.load() on it, so it must be a file path, not an array.
        """
        torch.manual_seed(1)
        model = AutoModelForCausalLM.from_config(micro_model_config)
        model_path = tmp_path / "micro"
        model.save_pretrained(str(model_path))
        tiny_tokenizer.save_pretrained(str(model_path))

        V = micro_model_config.vocab_size
        init_emb = np.random.default_rng(0).standard_normal(
            (V, micro_model_config.n_embd)
        ).astype(np.float32)
        init_path = tmp_path / "init.npy"
        np.save(str(init_path), init_emb)

        result = compute_embedding_trainedness(
            corpus_name="test_init",
            model_id=str(model_path),
            freq_table=_freq_table(V),
            stimulus_token_ids=_token_seqs(vocab_size=V),
            stimulus_sentences=[f"s{i}" for i in range(8)],
            init_embeddings=str(init_path),   # file path, not array
            out_dir=tmp_path / "diag_init"
        )
        assert isinstance(result, dict)
