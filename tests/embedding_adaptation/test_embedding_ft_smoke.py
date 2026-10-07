"""Smoke tests for embedding-only fine-tuning (brain-embed-ft).

All tests are CPU-only. A micro model is saved to a tmp directory and used
as the ``model_id`` so no HuggingFace downloads are required.
"""

from __future__ import annotations

import torch
import pytest
from transformers import AutoModelForCausalLM

from brain_code.training.module_adaptors import run_embedding_ablation


TEXTS = [
    "abc def ghi jkl mno",
    "pqr stu vwx yza bcd",
    "efg hij klm nop qrs",
    "tuv wxy zab cde fgh",
    "ijk lmn opq rst uvw",
    "xyz abc def ghi jkl",
    "mno pqr stu vwx yza",
    "bcd efg hij klm nop"
]

# Minimal training config that avoids GPU and runs in < 1 s.
_FT_KWARGS = dict(
    n_steps=2,
    max_tokens=300,
    lr=1e-3,
    context_length=4,
    batch_size=2
)


def _save_model(model, tokenizer, path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    return str(path)


class TestRunEmbeddingAblation:

    def test_output_directory_created(
        self,
        micro_model_config,
        tiny_tokenizer,
        tmp_path
    ):
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(micro_model_config)
        src = _save_model(model, tiny_tokenizer, tmp_path / "src")
        out = tmp_path / "ablated"

        run_embedding_ablation(
            model_id=src,
            english_texts=TEXTS,
            out_dir=out,
            **_FT_KWARGS
        )
        assert out.exists()
        assert (out / "config.json").exists()

    def test_only_embedding_weights_change(
        self,
        micro_model_config,
        tiny_tokenizer,
        tmp_path
    ):
        """All weights except wte should be byte-identical after fine-tuning."""
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(micro_model_config)
        wte_before = model.transformer.wte.weight.detach().clone()
        attn_before = model.transformer.h[0].attn.c_attn.weight.detach().clone()

        src = _save_model(model, tiny_tokenizer, tmp_path / "src")
        out = tmp_path / "ablated"

        run_embedding_ablation(
            model_id=src,
            english_texts=TEXTS,
            out_dir=out,
            **_FT_KWARGS
        )

        trained = AutoModelForCausalLM.from_pretrained(str(out))
        wte_after = trained.transformer.wte.weight
        attn_after = trained.transformer.h[0].attn.c_attn.weight

        # Embedding should have changed
        assert not torch.allclose(wte_before, wte_after), "wte did not update"
        # Attention weights should be frozen
        torch.testing.assert_close(attn_before, attn_after)

    def test_returns_path_string(
        self,
        micro_model_config,
        tiny_tokenizer,
        tmp_path
    ):
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(micro_model_config)
        src = _save_model(model, tiny_tokenizer, tmp_path / "src")

        result = run_embedding_ablation(
            model_id=src,
            english_texts=TEXTS,
            out_dir=tmp_path / "out",
            **_FT_KWARGS
        )
        assert isinstance(result, str)

    def test_dataset_too_small_raises(
        self,
        micro_model_config,
        tiny_tokenizer,
        tmp_path
    ):
        """batch_size > dataset chunks should raise ValueError, not silently skip."""
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(micro_model_config)
        src = _save_model(model, tiny_tokenizer, tmp_path / "src")

        with pytest.raises(ValueError, match="batch_size"):
            run_embedding_ablation(
                model_id=src,
                english_texts=["hi"],   # far too few tokens
                out_dir=tmp_path / "out",
                n_steps=2,
                max_tokens=300,
                context_length=64,
                batch_size=32          # impossible with 1 short string
            )
