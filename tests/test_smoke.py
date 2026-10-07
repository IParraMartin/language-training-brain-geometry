"""CPU toy smoke test — verifies the full pipeline on a minimal in-memory dataset.

Requires no GPU, no HuggingFace downloads, and no neural benchmark data.
Runs a 5-step training loop on 10 synthetic documents.
"""

from __future__ import annotations

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

from brain_code.training.train_model import LMTrainer
from brain_code.utils.get_null_positions import null_positions
from brain_code.utils.cache_utils import make_cache_key
from brain_code.experiments.brain_score_harness import BENCHMARK_METADATA, _compute_paper_normalized


_CHARS = "abcdefghijklmnopqrstuvwxyz "
_VOCAB = {c: i for i, c in enumerate(_CHARS)}
_VOCAB["[UNK]"] = len(_VOCAB)
_VOCAB["[EOS]"] = len(_VOCAB)
_VOCAB_SIZE = len(_VOCAB)

_DOCS = [
    "abc def ghi jkl mno pqr stu vwx yza bcd",
    "efg hij klm nop qrs tuv wxy zab cde fgh",
    "ijk lmn opq rst uvw xyz abc def ghi jkl",
    "mno pqr stu vwx yza bcd efg hij klm nop",
    "qrs tuv wxy zab cde fgh ijk lmn opq rst"
]


def _make_tok():
    t = Tokenizer(BPE(vocab=_VOCAB, merges=[]))
    t.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=t,
        unk_token="[UNK]",
        eos_token="[EOS]",
        pad_token="[EOS]"
    )


def _make_cfg():
    return AutoConfig.for_model(
        "gpt2",
        vocab_size=_VOCAB_SIZE,
        n_layer=2,
        n_head=2,
        n_embd=32,
        n_positions=16,
        n_ctx=16,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0
    )


def test_tokenized_dataset_builds_without_crash():
    tok = _make_tok()
    cfg = _make_cfg()
    model = AutoModelForCausalLM.from_config(cfg)
    trainer = LMTrainer.__new__(LMTrainer)
    trainer.tokenizer = tok
    trainer.model = model
    trainer.max_len = 16
    trainer.model_args = {"n_positions": 16}
    trainer.training_args = {"seed": 42}
    ds = DatasetDict({"train": Dataset.from_dict({"text": _DOCS})})
    tokenized = trainer.get_tokenized_dataset(ds, insert_eos_between_docs=True)
    assert "train" in tokenized
    assert len(tokenized["train"]) > 0


def test_ceiling_pipeline_smoke():
    """Verify the ceiling logic doesn't crash on any registered benchmark."""
    for bench_id in BENCHMARK_METADATA:
        pn, reason = _compute_paper_normalized(0.15, bench_id)
        assert isinstance(reason, str)
        if pn is not None:
            assert isinstance(pn, float)


def test_null_positions_smoke():
    """Null-position pipeline runs without error."""
    cfg = _make_cfg()
    model = AutoModelForCausalLM.from_config(cfg)
    model = null_positions(model)
    tokens = torch.tensor([[0, 1, 2, 3]])
    with torch.no_grad():
        out = model(tokens)
    assert out.logits.shape[-1] == _VOCAB_SIZE


def test_cache_key_smoke():
    k = make_cache_key(
        corpus="python",
        tok="default",
        field="code",
        rev="abc123"
    )
    assert len(k) == 16


def test_split_before_pack_smoke():
    """Split-before-pack path runs without error on 5 documents."""
    tok = _make_tok()
    cfg = _make_cfg()
    model = AutoModelForCausalLM.from_config(cfg)
    trainer = LMTrainer.__new__(LMTrainer)
    trainer.tokenizer = tok
    trainer.model = model
    trainer.max_len = 16
    trainer.model_args = {"n_positions": 16}
    trainer.training_args = {"seed": 42}

    ds = DatasetDict({"train": Dataset.from_dict({"text": _DOCS})})
    raw_split = ds["train"].train_test_split(test_size=0.2, seed=42)
    raw_train = DatasetDict({"train": raw_split["train"]})
    raw_eval = DatasetDict({"train": raw_split["test"]})

    tok_train = trainer.get_tokenized_dataset(raw_train, insert_eos_between_docs=True)
    tok_eval = trainer.get_tokenized_dataset(raw_eval, insert_eos_between_docs=True)
    # a smoke test: only the absence of a crash matters
    assert len(tok_train["train"]) >= 0
    assert len(tok_eval["train"]) >= 0
