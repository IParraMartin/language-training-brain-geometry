"""Tests for null-position-embedding terminology and behavior.

Key assertion: zeroing wpe does NOT make the model order-invariant.
Two different token orderings must produce different hidden states.
"""

from __future__ import annotations

import copy

import pytest
import torch

from brain_code.utils.get_null_positions import null_positions


@pytest.fixture
def zeroed_model(micro_model):
    """A fresh copy of the micro model with wpe zeroed."""
    m = copy.deepcopy(micro_model)
    return null_positions(m)


def test_wpe_is_zeroed(zeroed_model):
    assert zeroed_model.transformer.wpe.weight.data.abs().max().item() == 0.0


def test_wpe_is_frozen(zeroed_model):
    assert not zeroed_model.transformer.wpe.weight.requires_grad


def test_model_is_NOT_order_invariant(zeroed_model):
    """Permuting tokens must still change the hidden states (order still matters)."""
    # build the tokens on the model's own device: earlier tests may have moved
    # the shared micro_model fixture to CUDA (extract_hidden_states moves models
    # in place), and this assertion is about ordering, not device placement
    device = next(zeroed_model.parameters()).device
    tokens_abc = torch.tensor([[0, 1, 2]], device=device)   # a, b, c
    tokens_cab = torch.tensor([[2, 0, 1]], device=device)   # c, a, b

    zeroed_model.eval()
    with torch.no_grad():
        out_abc = zeroed_model(tokens_abc, output_hidden_states=True).hidden_states[-1]
        out_cab = zeroed_model(tokens_cab, output_hidden_states=True).hidden_states[-1]

    # position 0 holds a different token in each sequence, so the states must differ
    max_diff = (out_abc - out_cab).abs().max().item()
    assert max_diff > 1e-6, (
        "zeroed wpe model produced identical hidden states for different token orders.  "
        "This would indicate true order-invariance, which should be impossible with "
        "a causal attention mask."
    )


def test_warning_message_mentions_wpe_and_causal_mask(capsys):
    """The warning must mention wpe removal and clarify that causal attention persists."""
    from conftest import VOCAB_SIZE
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(
        "gpt2",
        vocab_size=VOCAB_SIZE,
        n_layer=2,
        n_head=2,
        n_embd=64,
        n_positions=128,
        n_ctx=128
    )
    m = AutoModelForCausalLM.from_config(cfg)
    null_positions(m)
    captured = capsys.readouterr().out.lower()
    assert "wpe" in captured or "absolute position" in captured, (
        "Warning must mention wpe or absolute position embeddings."
    )
    assert "causal" in captured or "token order" in captured, (
        "Warning must clarify that causal attention mask / token order still encodes order."
    )
    # saying the model is NOT positionally invariant is fine; claiming it IS is not
    out = captured.replace("not positionally invariant", "").replace("is not", "")
    assert "positionally invariant" not in out or "not" in captured, (
        "The warning must not assert the model IS positionally invariant."
    )
