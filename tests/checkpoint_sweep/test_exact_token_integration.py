"""Exact-token integration tests for the checkpoint sweep.

Covers the pieces that turn nominal token estimates into exact, checkpoint-bound
accounting, all runnable on CPU without Brain-Score data:

* hash_checkpoint_dir determinism / sensitivity
* TokenCountingTrainer non-padding reduction (single-process path + distributed guard)
* ExactTokenMilestoneCallback milestone crossing, checkpoint hashing, and
  fail-closed vs bound neural-alignment status
* item 9: same-seed initial parameters AND first optimizer update match;
  different seed diverges
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from brain_code.training.checkpoint_logger import (
    ExactTokenMilestoneCallback,
    hash_checkpoint_dir
)
from brain_code.training.exact_token_accounting import ExactTokenCounter
from brain_code.training.token_counting_trainer import TokenCountingTrainer
from brain_code.training.reproducibility import parameter_hash, seed_everything


def test_hash_checkpoint_dir_is_deterministic(tmp_path):
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir()
    (ckpt / "model.bin").write_bytes(b"weights")
    (ckpt / "optimizer.pt").write_bytes(b"opt-state")
    first = hash_checkpoint_dir(ckpt)
    assert first == hash_checkpoint_dir(ckpt)
    assert len(first) == 64  # full sha256 hex


def test_hash_checkpoint_dir_changes_with_contents(tmp_path):
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir()
    (ckpt / "model.bin").write_bytes(b"weights")
    before = hash_checkpoint_dir(ckpt)
    (ckpt / "model.bin").write_bytes(b"weights-updated")
    assert hash_checkpoint_dir(ckpt) != before


def test_hash_checkpoint_dir_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        hash_checkpoint_dir(tmp_path / "does-not-exist")


def test_reduced_tokens_single_process_returns_local():
    trainer = TokenCountingTrainer.__new__(TokenCountingTrainer)
    trainer.args = SimpleNamespace(world_size=1, device="cpu")
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    assert trainer._globally_reduced_tokens({"attention_mask": mask}) == 5


def test_reduced_tokens_distributed_without_group_raises():
    trainer = TokenCountingTrainer.__new__(TokenCountingTrainer)
    trainer.args = SimpleNamespace(world_size=2, device="cpu")
    with pytest.raises(RuntimeError, match="not initialized"):
        trainer._globally_reduced_tokens({"attention_mask": torch.ones((1, 4), dtype=torch.long)})


def _control():
    return SimpleNamespace(should_save=False, should_evaluate=False)


def test_callback_triggers_save_on_exact_milestone(tmp_path):
    counter = ExactTokenCounter([100, 500])
    cb = ExactTokenMilestoneCallback(
        counter,
        [100, 500],
        log_dir=tmp_path,
        seed=7
    )
    args = SimpleNamespace(output_dir=str(tmp_path))
    control = _control()

    # Below the first milestone → no trigger.
    counter.observe_batch({"attention_mask": np.ones((1, 60))})
    cb.on_step_end(args, SimpleNamespace(global_step=1), control)
    assert control.should_save is False

    # Cross 100 → trigger save + evaluate.
    counter.observe_batch({"attention_mask": np.ones((1, 60))})  # tokens_seen=120
    cb.on_step_end(args, SimpleNamespace(global_step=2), control)
    assert control.should_save is True
    assert control.should_evaluate is True
    assert cb._pending == [100]


def test_callback_records_not_run_alignment_by_default(tmp_path):
    counter = ExactTokenCounter([100])
    cb = ExactTokenMilestoneCallback(
        counter,
        [100],
        log_dir=tmp_path,
        seed=7
    )
    args = SimpleNamespace(output_dir=str(tmp_path))
    # Cross the milestone.
    counter.observe_batch({"attention_mask": np.ones((1, 150))})
    cb.on_step_end(args, SimpleNamespace(global_step=5), _control())
    # The Trainer would have written this checkpoint; emulate it.
    ckpt = tmp_path / "checkpoint-5"
    ckpt.mkdir()
    (ckpt / "model.bin").write_bytes(b"w")
    state = SimpleNamespace(global_step=5, log_history=[{"eval_loss": 2.5}])
    cb.on_save(args, state, _control())

    records = cb.load_log()
    assert len(records) == 1
    rec = records[0]
    assert rec["milestone_tokens"] == 100
    assert rec["actual_nonpadding_tokens"] == 150
    assert rec["neural_alignment_status"] == "not_run"
    assert rec["neural_alignment"] is None
    assert rec["eval_loss"] == 2.5
    assert len(rec["checkpoint_hash"]) == 64


def test_callback_binds_alignment_when_fn_supplied(tmp_path):
    counter = ExactTokenCounter([100])
    seen_args = {}

    def fake_alignment(checkpoint_dir, checkpoint_hash):
        seen_args["dir"] = checkpoint_dir
        seen_args["hash"] = checkpoint_hash
        return {"benchmark": "toy", "raw_score": 0.123}

    cb = ExactTokenMilestoneCallback(
        counter,
        [100],
        log_dir=tmp_path,
        seed=7,
        alignment_fn=fake_alignment
    )
    args = SimpleNamespace(output_dir=str(tmp_path))
    counter.observe_batch({"attention_mask": np.ones((1, 120))})
    cb.on_step_end(args, SimpleNamespace(global_step=3), _control())
    ckpt = tmp_path / "checkpoint-3"
    ckpt.mkdir()
    (ckpt / "model.bin").write_bytes(b"w")
    cb.on_save(
        args,
        SimpleNamespace(global_step=3, log_history=[{"step": 3, "eval_loss": 1.5}]),
        _control()
    )

    rec = cb.load_log()[0]
    assert rec["neural_alignment_status"] == "bound"
    assert rec["neural_alignment"]["benchmark"] == "toy"
    assert rec["neural_alignment"]["raw_score"] == 0.123
    assert rec["neural_alignment"]["checkpoint_hash"] == rec["checkpoint_hash"]
    assert rec["token_overshoot"] == 20
    # Alignment was bound to the exact checkpoint hash that was recorded.
    assert seen_args["hash"] == rec["checkpoint_hash"]


def test_callback_records_failed_alignment_and_raises(tmp_path):
    counter = ExactTokenCounter([100])

    def boom(checkpoint_dir, checkpoint_hash):
        raise ValueError("alignment blew up")

    cb = ExactTokenMilestoneCallback(
        counter,
        [100],
        log_dir=tmp_path,
        alignment_fn=boom
    )
    args = SimpleNamespace(output_dir=str(tmp_path))
    counter.observe_batch({"attention_mask": np.ones((1, 120))})
    cb.on_step_end(args, SimpleNamespace(global_step=3), _control())
    ckpt = tmp_path / "checkpoint-3"
    ckpt.mkdir()
    (ckpt / "model.bin").write_bytes(b"w")
    with pytest.raises(RuntimeError, match="Neural alignment failed"):
        cb.on_save(
            args,
            SimpleNamespace(global_step=3, log_history=[{"step": 3, "eval_loss": 1.5}]),
            _control()
        )
    rec = cb.load_log()[0]
    assert rec["neural_alignment_status"].startswith("failed")
    assert rec["neural_alignment"] is None


def _micro_model():
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(
        "gpt2",
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_embd=32,
        n_positions=32,
        n_ctx=32,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0  # deterministic forward
    )
    return AutoModelForCausalLM.from_config(cfg)


def _one_update(model):
    """Deterministic single optimizer step on a fixed batch; return param hash."""
    torch.manual_seed(0)  # optimizer/step determinism independent of init seed
    ids = torch.arange(16).reshape(2, 8) % 64
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    model.train()
    opt.zero_grad()
    out = model(**batch)
    out.loss.backward()
    opt.step()
    return parameter_hash(model)


def test_same_seed_matches_initial_and_first_update():
    seed_everything(11, deterministic=True)
    model_a = _micro_model()
    init_a = parameter_hash(model_a)

    seed_everything(11, deterministic=True)
    model_b = _micro_model()
    init_b = parameter_hash(model_b)

    assert init_a == init_b, "same seed must give identical initial parameters"
    assert _one_update(model_a) == _one_update(model_b), (
        "same seed + identical batch must give identical parameters after one update"
    )


def test_different_seed_diverges():
    seed_everything(11, deterministic=True)
    init_11 = parameter_hash(_micro_model())
    seed_everything(12, deterministic=True)
    init_12 = parameter_hash(_micro_model())
    assert init_11 != init_12, "different seeds must give different initial parameters"
