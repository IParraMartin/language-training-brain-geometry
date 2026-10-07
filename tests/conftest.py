"""Shared fixtures for the brain-code test suite.

All fixtures here are CPU-only and require no HuggingFace downloads,
no neural benchmark data, and no GPU.  They use in-memory toy models
and datasets so the suite can run on CI with minimal resources.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace


VOCAB = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
VOCAB["[UNK]"] = 26
VOCAB["[EOS]"] = 27
VOCAB_SIZE = len(VOCAB)


@pytest.fixture(scope="session")
def tiny_tokenizer():
    """A minimal fast tokenizer over 26 lower-case ASCII characters."""
    tok = Tokenizer(BPE(vocab=dict(VOCAB), merges=[]))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        eos_token="[EOS]",
        pad_token="[EOS]"
    )


@pytest.fixture(scope="session")
def micro_model_config():
    """GPT-2 config small enough for CPU unit tests (2 layers, 64-d, 128 ctx)."""
    return AutoConfig.for_model(
        "gpt2",
        vocab_size=VOCAB_SIZE,
        n_layer=2,
        n_head=2,
        n_embd=64,
        n_positions=128,
        n_ctx=128,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0
    )


@pytest.fixture(scope="session")
def micro_model(micro_model_config):
    """A fresh GPT-2 micro model with no pretrained weights."""
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(micro_model_config)


@pytest.fixture(scope="session")
def toy_texts():
    """Ten short synthetic documents with distinct token patterns."""
    return [
        "abc def ghi",
        "jkl mno pqr",
        "stu vwx yza",
        "bcd efg hij",
        "klm nop qrs",
        "tuv wxy zab",
        "cde fgh ijk",
        "lmn opq rst",
        "uvw xyz abc",
        "def ghi jkl"
    ]
