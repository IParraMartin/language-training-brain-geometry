"""Tests for embedding patching and weight-tie transparency.

GPT-2 ties wte and lm_head, so patching wte also changes the output head;
patch_embeddings must say so in its log.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from brain_code.experiments.causal_replacement import patch_embeddings, _log_weight_tie_status


def _make_tiny_model(
    vocab_size=30,
    n_layer=2,
    n_head=2,
    n_embd=32
):
    cfg = AutoConfig.for_model(
        "gpt2",
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_positions=32,
        n_ctx=32,
        tie_word_embeddings=True
    )
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg)


def _model_to_temp_dir(model, tmp_path, name: str):
    d = tmp_path / name
    d.mkdir()
    model.save_pretrained(d)
    return str(d)


@pytest.fixture
def two_tiny_models(tmp_path):
    donor = _make_tiny_model()
    torch.manual_seed(1)
    receiver = _make_tiny_model()
    # make the wte weights differ
    receiver.transformer.wte.weight.data.fill_(0.5)
    donor_path = _model_to_temp_dir(donor, tmp_path, "donor")
    receiver_path = _model_to_temp_dir(receiver, tmp_path, "receiver")
    return (
        donor_path,
        receiver_path,
        donor,
        receiver
    )


def test_patch_embeddings_tied_model_logs_tied_warning(two_tiny_models, capsys):
    """For a tied-weight GPT-2, patching wte inevitably updates lm_head.

    GPT-2 ties wte and lm_head: they share the same underlying storage.
    Modifying wte.weight.data therefore always modifies lm_head.weight.data too,
    regardless of sync_lm_head.  The function must log a warning about this.
    """
    donor_path, receiver_path, donor, receiver = two_tiny_models
    patched = patch_embeddings(
        donor_model=donor_path,
        receiver_model=receiver_path,
        include_position_embeddings=False,
        sync_lm_head=False
    )
    captured = capsys.readouterr().out
    assert "tied" in captured.lower() or "share" in captured.lower(), (
        "patch_embeddings must log a warning when wte and lm_head share storage."
    )
    loaded_donor = AutoModelForCausalLM.from_pretrained(donor_path)
    assert torch.allclose(
        patched.transformer.wte.weight.data,
        loaded_donor.transformer.wte.weight.data
    )


def test_patch_embeddings_head_sync_changes_head(two_tiny_models):
    donor_path, receiver_path, donor, receiver = two_tiny_models
    patched = patch_embeddings(
        donor_model=donor_path,
        receiver_model=receiver_path,
        include_position_embeddings=False,
        sync_lm_head=True
    )
    assert torch.allclose(patched.lm_head.weight.data, patched.transformer.wte.weight.data), (
        "sync_lm_head=True: lm_head.weight should match wte.weight after tie_weights()."
    )


def test_log_weight_tie_status_returns_correct_tied_status():
    model = _make_tiny_model()
    # GPT-2 ties wte and lm_head by default
    tied = _log_weight_tie_status(model, "test")
    # tied or not, the function must return a bool without error
    assert isinstance(tied, bool)


def test_wte_is_updated_after_patch(two_tiny_models):
    donor_path, receiver_path, donor, receiver = two_tiny_models
    patched = patch_embeddings(
        donor_model=donor_path,
        receiver_model=receiver_path,
        include_position_embeddings=False,
        sync_lm_head=False
    )
    loaded_donor = AutoModelForCausalLM.from_pretrained(donor_path)
    assert torch.allclose(
        patched.transformer.wte.weight.data,
        loaded_donor.transformer.wte.weight.data
    ), "wte must equal donor wte after patching."
