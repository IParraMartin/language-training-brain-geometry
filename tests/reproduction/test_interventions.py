"""Transplant endpoints, tensor copying, weight tying and the trainable tensors of embedding adaptation."""

from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from brain_code.experiments.progressive_transplant import state_dict_hash, transplant
from brain_code.training.module_adaptors import freeze_all_but_token_embeddings


def _model(seed: int) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=40,
        n_positions=16,
        n_embd=16,
        n_layer=4,
        n_head=2
    )
    return GPT2LMHeadModel(config).eval()


@pytest.fixture(scope="module")
def pair():
    return _model(0), _model(1)


def test_zero_blocks_without_embeddings_is_the_receiver(pair):
    receiver, donor = pair
    assert state_dict_hash(transplant(
        receiver,
        donor,
        0,
        False
    )) == state_dict_hash(receiver)


def test_all_blocks_with_embeddings_is_the_donor(pair):
    receiver, donor = pair
    assert state_dict_hash(transplant(
        receiver,
        donor,
        4,
        True
    )) == state_dict_hash(donor)


def test_partial_transplant_copies_exactly_the_requested_parts(pair):
    receiver, donor = pair
    model = transplant(
        receiver,
        donor,
        2,
        False
    )
    for i in range(4):
        source = donor if i < 2 else receiver
        copied = model.transformer.h[i].state_dict().values()
        expected = source.transformer.h[i].state_dict().values()
        for a, b in zip(copied, expected):
            assert torch.equal(a, b)
    assert torch.equal(model.transformer.wte.weight, receiver.transformer.wte.weight)
    assert torch.equal(model.transformer.ln_f.weight, receiver.transformer.ln_f.weight)


def test_full_blocks_only_arm_copies_final_norm_but_keeps_receiver_embeddings(pair):
    receiver, donor = pair
    model = transplant(
        receiver,
        donor,
        4,
        False
    )
    assert torch.equal(model.transformer.ln_f.weight, donor.transformer.ln_f.weight)
    assert torch.equal(model.transformer.wte.weight, receiver.transformer.wte.weight)
    assert torch.equal(model.transformer.wpe.weight, receiver.transformer.wpe.weight)


def test_embedding_arm_keeps_the_output_head_tied(pair):
    receiver, donor = pair
    model = transplant(
        receiver,
        donor,
        1,
        True
    )
    assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()
    assert torch.equal(model.lm_head.weight, donor.transformer.wte.weight)


def test_transplant_leaves_its_inputs_unchanged(pair):
    receiver, donor = pair
    before = state_dict_hash(receiver), state_dict_hash(donor)
    transplant(
        receiver,
        donor,
        3,
        True
    )
    assert (state_dict_hash(receiver), state_dict_hash(donor)) == before


def test_embedding_adaptation_trains_only_the_tied_token_matrix():
    model = _model(2)
    emb = freeze_all_but_token_embeddings(model)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable == ["transformer.wte.weight"]
    assert model.lm_head.weight is emb.weight
    assert not model.transformer.wpe.weight.requires_grad
    loss = model(input_ids=torch.tensor([[
        1,
        2,
        3,
        4
    ]]), labels=torch.tensor([[1, 2, 3, 4]])).loss
    loss.backward()
    assert emb.weight.grad is not None
    assert all(p.grad is None for name, p in model.named_parameters() if name != "transformer.wte.weight")
