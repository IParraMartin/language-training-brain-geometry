"""Smoke tests for causal replacement / transplant (brain-patch).

All tests are CPU-only. Donor and receiver are both saved micro models so
no HuggingFace downloads are required.
"""

from __future__ import annotations

import torch
import pytest
from transformers import AutoModelForCausalLM

from brain_code.experiments.causal_replacement import (
    patch_embeddings,
    patch_block,
    save_patched
)


def _save(model, tokenizer, path):
    """Save model + tokenizer to path; return path as str."""
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    return str(path)


class TestPatchEmbeddings:

    def test_wte_matches_donor(
        self,
        micro_model_config,
        tiny_tokenizer,
        tmp_path
    ):
        """After patching, receiver wte must equal donor wte."""
        torch.manual_seed(1)
        donor = AutoModelForCausalLM.from_config(micro_model_config)
        torch.manual_seed(2)
        receiver = AutoModelForCausalLM.from_config(micro_model_config)

        d_path = _save(donor, tiny_tokenizer, tmp_path / "donor")
        r_path = _save(receiver, tiny_tokenizer, tmp_path / "receiver")

        patched = patch_embeddings(
            donor_model=d_path,
            receiver_model=r_path,
            include_position_embeddings=False
        )

        torch.testing.assert_close(
            patched.transformer.wte.weight,
            donor.transformer.wte.weight
        )

    def test_wpe_matches_donor_when_included(
        self,
        micro_model_config,
        tiny_tokenizer,
        tmp_path
    ):
        torch.manual_seed(1)
        donor = AutoModelForCausalLM.from_config(micro_model_config)
        torch.manual_seed(2)
        receiver = AutoModelForCausalLM.from_config(micro_model_config)

        d_path = _save(donor, tiny_tokenizer, tmp_path / "donor")
        r_path = _save(receiver, tiny_tokenizer, tmp_path / "receiver")

        patched = patch_embeddings(
            donor_model=d_path,
            receiver_model=r_path,
            include_position_embeddings=True
        )

        torch.testing.assert_close(
            patched.transformer.wpe.weight,
            donor.transformer.wpe.weight
        )

    def test_transformer_body_unchanged(
        self,
        micro_model_config,
        tiny_tokenizer,
        tmp_path
    ):
        """Attention weights must not be touched by an embedding-only patch."""
        torch.manual_seed(1)
        donor = AutoModelForCausalLM.from_config(micro_model_config)
        torch.manual_seed(2)
        receiver = AutoModelForCausalLM.from_config(micro_model_config)
        attn_before = receiver.transformer.h[0].attn.c_attn.weight.detach().clone()

        d_path = _save(donor, tiny_tokenizer, tmp_path / "donor")
        r_path = _save(receiver, tiny_tokenizer, tmp_path / "receiver")

        patched = patch_embeddings(donor_model=d_path, receiver_model=r_path)
        torch.testing.assert_close(attn_before, patched.transformer.h[0].attn.c_attn.weight)

    def test_shape_mismatch_raises(
        self,
        micro_model_config,
        tiny_tokenizer,
        tmp_path
    ):
        """Donor/receiver with incompatible wte shapes should raise ValueError."""
        import copy

        torch.manual_seed(1)
        donor = AutoModelForCausalLM.from_config(micro_model_config)

        # Build a receiver with a different vocab size
        alt_cfg = copy.deepcopy(micro_model_config)
        alt_cfg.vocab_size = micro_model_config.vocab_size + 10
        torch.manual_seed(2)
        receiver_bad = AutoModelForCausalLM.from_config(alt_cfg)

        # Receiver needs a matching tokenizer to save — use the same tiny one
        d_path = _save(donor, tiny_tokenizer, tmp_path / "donor")
        r_path = _save(receiver_bad, tiny_tokenizer, tmp_path / "receiver_bad")

        with pytest.raises(ValueError, match="wte shape mismatch"):
            patch_embeddings(donor_model=d_path, receiver_model=r_path)


class TestPatchBlock:

    def test_target_block_matches_donor(
        self,
        micro_model_config,
        tiny_tokenizer,
        tmp_path
    ):
        torch.manual_seed(1)
        donor = AutoModelForCausalLM.from_config(micro_model_config)
        torch.manual_seed(2)
        receiver = AutoModelForCausalLM.from_config(micro_model_config)

        d_path = _save(donor, tiny_tokenizer, tmp_path / "donor")
        r_path = _save(receiver, tiny_tokenizer, tmp_path / "receiver")

        patched = patch_block(donor_model=d_path, receiver_model=r_path, layer=0)

        donor_sd = donor.transformer.h[0].state_dict()
        patched_sd = patched.transformer.h[0].state_dict()
        for k in donor_sd:
            torch.testing.assert_close(donor_sd[k], patched_sd[k], msg=f"Mismatch at {k}")

    def test_other_blocks_unchanged(
        self,
        micro_model_config,
        tiny_tokenizer,
        tmp_path
    ):
        """Patching block 0 must leave block 1 identical to the receiver."""
        torch.manual_seed(1)
        donor = AutoModelForCausalLM.from_config(micro_model_config)
        torch.manual_seed(2)
        receiver = AutoModelForCausalLM.from_config(micro_model_config)
        block1_before = {
            k: v.detach().clone()
            for k, v in receiver.transformer.h[1].state_dict().items()
        }

        d_path = _save(donor, tiny_tokenizer, tmp_path / "donor")
        r_path = _save(receiver, tiny_tokenizer, tmp_path / "receiver")

        patched = patch_block(donor_model=d_path, receiver_model=r_path, layer=0)

        for k, v in block1_before.items():
            torch.testing.assert_close(
                v,
                patched.transformer.h[1].state_dict()[k],
                msg=f"block 1 changed at {k}"
            )

    def test_out_of_range_layer_raises(
        self,
        micro_model_config,
        tiny_tokenizer,
        tmp_path
    ):
        torch.manual_seed(1)
        donor = AutoModelForCausalLM.from_config(micro_model_config)
        torch.manual_seed(2)
        receiver = AutoModelForCausalLM.from_config(micro_model_config)

        d_path = _save(donor, tiny_tokenizer, tmp_path / "donor")
        r_path = _save(receiver, tiny_tokenizer, tmp_path / "receiver")

        with pytest.raises(ValueError, match="out of range"):
            patch_block(donor_model=d_path, receiver_model=r_path, layer=999)


class TestSavePatched:

    def test_saves_model_and_tokenizer_files(
        self,
        micro_model_config,
        tiny_tokenizer,
        tmp_path
    ):
        torch.manual_seed(1)
        donor = AutoModelForCausalLM.from_config(micro_model_config)
        torch.manual_seed(2)
        receiver = AutoModelForCausalLM.from_config(micro_model_config)

        d_path = _save(donor, tiny_tokenizer, tmp_path / "donor")
        r_path = _save(receiver, tiny_tokenizer, tmp_path / "receiver")

        patched = patch_embeddings(donor_model=d_path, receiver_model=r_path)
        out = tmp_path / "saved"
        save_patched(model=patched, tokenizer_source=r_path, out_dir=out)

        assert (out / "config.json").exists()
        assert any(out.glob("*.safetensors")) or any(out.glob("pytorch_model*.bin"))

    def test_saved_model_is_loadable(
        self,
        micro_model_config,
        tiny_tokenizer,
        tmp_path
    ):
        torch.manual_seed(1)
        donor = AutoModelForCausalLM.from_config(micro_model_config)
        torch.manual_seed(2)
        receiver = AutoModelForCausalLM.from_config(micro_model_config)

        d_path = _save(donor, tiny_tokenizer, tmp_path / "donor")
        r_path = _save(receiver, tiny_tokenizer, tmp_path / "receiver")

        patched = patch_block(donor_model=d_path, receiver_model=r_path, layer=0)
        out = tmp_path / "saved"
        save_patched(model=patched, tokenizer_source=r_path, out_dir=out)

        reloaded = AutoModelForCausalLM.from_pretrained(str(out))
        assert reloaded is not None
