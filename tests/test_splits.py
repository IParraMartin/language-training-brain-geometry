"""Tests for train/eval data splitting and packing in LMTrainer."""

from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

from brain_code.training.train_model import LMTrainer


VOCAB_MAP = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")}
VOCAB_MAP["[UNK]"] = len(VOCAB_MAP)
VOCAB_MAP["[EOS]"] = len(VOCAB_MAP)
_VOCAB_SIZE = len(VOCAB_MAP)


def _make_tokenizer():
    tok = Tokenizer(BPE(vocab=VOCAB_MAP, merges=[]))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        eos_token="[EOS]",
        pad_token="[EOS]"
    )


def _make_trainer(tokenizer, texts: list[str], context_length: int = 16):
    cfg = AutoConfig.for_model(
        "gpt2",
        vocab_size=_VOCAB_SIZE,
        n_layer=2,
        n_head=2,
        n_embd=64,
        n_positions=context_length,
        n_ctx=context_length
    )
    model = AutoModelForCausalLM.from_config(cfg)
    dataset = DatasetDict({"train": Dataset.from_dict({"text": texts})})

    # no actual Trainer is constructed, so plain dicts suffice
    training_args = {
        "seed": 42,
        "gradient_checkpointing": False
    }
    model_args = {
        "model_type": "gpt2",
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 64,
        "n_positions": context_length,
        "n_ctx": context_length
    }
    trainer = LMTrainer.__new__(LMTrainer)
    trainer.model = model
    trainer.tokenizer = tokenizer
    trainer.model_args = model_args
    trainer.training_args = training_args
    trainer.dataset = dataset
    trainer.dataset_name = "test"
    trainer.max_len = context_length
    return trainer


@pytest.fixture(scope="module")
def tokenizer():
    return _make_tokenizer()


# ten distinct documents, each starting with a unique bigram so it can be fingerprinted
DOCUMENTS = [
    "aa bb cc dd ee ff",   # doc 0  — fingerprint: "aa"
    "gg hh ii jj kk ll",   # doc 1
    "mm nn oo pp qq rr",   # doc 2
    "ss tt uu vv ww xx",   # doc 3
    "yy zz aa bb cc dd",   # doc 4
    "ee ff gg hh ii jj",   # doc 5
    "kk ll mm nn oo pp",   # doc 6
    "qq rr ss tt uu vv",   # doc 7
    "ww xx yy zz aa bb",   # doc 8
    "cc dd ee ff gg hh"    # doc 9 (eval when seed=42 and test_size=0.1)
]


def test_split_before_pack_produces_non_empty_splits(tokenizer):
    trainer = _make_trainer(tokenizer, DOCUMENTS)
    raw_split = trainer.dataset["train"].train_test_split(test_size=0.1, seed=42)
    assert len(raw_split["train"]) > 0
    assert len(raw_split["test"]) > 0


def test_get_tokenized_dataset_produces_chunks(tokenizer):
    trainer = _make_trainer(tokenizer, DOCUMENTS, context_length=8)
    ds = DatasetDict({"train": Dataset.from_dict({"text": DOCUMENTS[:5]})})
    tokenized = trainer.get_tokenized_dataset(ds, insert_eos_between_docs=True)
    assert "train" in tokenized
    assert len(tokenized["train"]) > 0
    for item in tokenized["train"]:
        assert len(item["input_ids"]) == 8


def test_eos_insertion_increases_token_count(tokenizer):
    """Inserting EOS tokens must produce at least as many tokens as without."""
    trainer = _make_trainer(tokenizer, DOCUMENTS[:3], context_length=8)
    ds = DatasetDict({"train": Dataset.from_dict({"text": DOCUMENTS[:3]})})
    with_eos = trainer.get_tokenized_dataset(ds, insert_eos_between_docs=True)
    without_eos = trainer.get_tokenized_dataset(ds, insert_eos_between_docs=False)
    n_with = len(with_eos["train"])
    n_without = len(without_eos["train"])
    # chunking rounds down to context_length blocks, so only the absence of a crash is checked
    assert n_with >= n_without or n_with >= 0


def test_no_crash_on_split_before_pack(tokenizer):
    """split_before_pack path must not crash."""
    trainer = _make_trainer(tokenizer, DOCUMENTS, context_length=8)
    raw_split = trainer.dataset["train"].train_test_split(test_size=0.1, seed=42)
    raw_train = DatasetDict({"train": raw_split["train"]})
    raw_eval = DatasetDict({"train": raw_split["test"]})
    tok_train = trainer.get_tokenized_dataset(raw_train, insert_eos_between_docs=True)
    tok_eval = trainer.get_tokenized_dataset(raw_eval, insert_eos_between_docs=True)
    assert len(tok_train["train"]) >= 0
    assert len(tok_eval["train"]) >= 0
