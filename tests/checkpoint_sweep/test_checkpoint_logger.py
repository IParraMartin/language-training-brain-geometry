"""Tests for the checkpoint-sweep logger infrastructure."""

from __future__ import annotations

import json
import tempfile

import numpy as np

from brain_code.training.checkpoint_logger import (
    TokenMilestoneCallback,
    CheckpointMetrics,
    embedding_displacement,
    embedding_anisotropy,
    embedding_effective_rank,
    wte_hash,
    compute_embedding_metrics
)


def _make_wte(vocab=50, d=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((vocab, d)).astype(np.float32)


def test_displacement_zero_for_identical_wte():
    wte = _make_wte()
    assert embedding_displacement(wte, wte) == 0.0


def test_displacement_positive_for_different_wte():
    wte_a = _make_wte(seed=0)
    wte_b = _make_wte(seed=1)
    assert embedding_displacement(wte_a, wte_b) > 0.0


def test_displacement_increases_with_noise():
    wte_init = _make_wte()
    rng = np.random.default_rng(99)
    wte_small = wte_init + 0.01 * rng.standard_normal(wte_init.shape).astype(np.float32)
    wte_large = wte_init + 1.00 * rng.standard_normal(wte_init.shape).astype(np.float32)
    assert embedding_displacement(wte_small, wte_init) < embedding_displacement(wte_large, wte_init)


def test_anisotropy_in_unit_interval():
    wte = _make_wte(vocab=100, d=16)
    a = embedding_anisotropy(wte)
    assert 0.0 < a < 1.0, f"anisotropy should be in (0, 1): got {a}"


def test_anisotropy_high_for_rank1():
    # Rank-1 matrix: outer product — all spectral mass in one direction
    u = np.random.default_rng(0).standard_normal(64).astype(np.float32)
    v = np.random.default_rng(1).standard_normal(16).astype(np.float32)
    rank1 = np.outer(u / np.linalg.norm(u), v / np.linalg.norm(v)).astype(np.float32)
    a = embedding_anisotropy(rank1)
    assert a > 0.95, f"rank-1 matrix should have high anisotropy: got {a}"


def test_effective_rank_upper_bounded():
    wte = _make_wte(vocab=50, d=16)
    er = embedding_effective_rank(wte)
    min_dim = min(wte.shape)
    assert 1.0 <= er <= min_dim + 0.01, (
        f"effective_rank should be in [1, min_dim={min_dim}]: got {er}"
    )


def test_effective_rank_one_for_rank1():
    u = np.ones((64, 1), dtype=np.float32) / 8
    v = np.ones((1, 16), dtype=np.float32) / 4
    rank1 = (u @ v).astype(np.float32)
    er = embedding_effective_rank(rank1)
    assert er < 2.0, f"rank-1 matrix should have effective_rank ≈ 1: got {er}"


def test_wte_hash_is_deterministic():
    wte = _make_wte()
    assert wte_hash(wte) == wte_hash(wte)


def test_wte_hash_changes_with_wte():
    wte_a = _make_wte(seed=0)
    wte_b = _make_wte(seed=1)
    assert wte_hash(wte_a) != wte_hash(wte_b)


def test_wte_hash_is_eight_hex_chars():
    h = wte_hash(_make_wte())
    assert len(h) == 8
    assert all(c in "0123456789abcdef" for c in h)


def test_compute_embedding_metrics_returns_triple():
    wte_init = _make_wte(seed=0)
    wte_curr = _make_wte(seed=1)
    disp, aniso, effr = compute_embedding_metrics(wte_curr, wte_init)
    assert disp > 0
    assert 0 < aniso < 1
    assert effr >= 1


def test_milestone_step_conversion():
    with tempfile.TemporaryDirectory() as log_dir:
        cb = TokenMilestoneCallback(
            milestones_tokens=[1_000, 5_000],
            batch_size=10,
            seq_len=10,
            log_dir=log_dir,
            save_checkpoints=False
        )
        # tokens_per_step = 10 × 10 = 100
        # milestone 1_000 → step 10
        # milestone 5_000 → step 50
        assert 10 in cb.milestone_steps
        assert 50 in cb.milestone_steps
        assert cb.milestone_steps[10] == 1_000
        assert cb.milestone_steps[50] == 5_000


def test_milestone_steps_not_zero():
    # Even a very small milestone should map to step ≥ 1
    with tempfile.TemporaryDirectory() as log_dir:
        cb = TokenMilestoneCallback(
            milestones_tokens=[1],
            batch_size=1000,
            seq_len=1000,
            log_dir=log_dir,
            save_checkpoints=False
        )
        assert all(s >= 1 for s in cb.milestone_steps)


def test_log_written_to_jsonl():
    wte_init = _make_wte()
    wte_curr = _make_wte(seed=5)
    with tempfile.TemporaryDirectory() as log_dir:
        cb = TokenMilestoneCallback(
            milestones_tokens=[1_000],
            batch_size=10,
            seq_len=10,
            log_dir=log_dir,
            save_checkpoints=False
        )
        cb._wte_init = wte_init.copy()
        cb._init_hash = wte_hash(wte_init)

        # Manually trigger a record
        disp, aniso, effr = compute_embedding_metrics(wte_curr, wte_init)
        from dataclasses import asdict
        rec = CheckpointMetrics(
            milestone_tokens=1_000,
            step=10,
            train_loss=2.5,
            eval_loss=2.6,
            embedding_displacement=disp,
            anisotropy=aniso,
            effective_rank=effr,
            init_hash=cb._init_hash
        )
        with open(cb._log_path, "a") as fh:
            fh.write(json.dumps(asdict(rec)) + "\n")

        records = cb.load_log()
        assert len(records) == 1
        r = records[0]
        assert r["milestone_tokens"] == 1_000
        assert r["step"] == 10
        assert r["init_hash"] == cb._init_hash
        assert r["embedding_displacement"] > 0


def test_load_log_returns_empty_when_no_file():
    with tempfile.TemporaryDirectory() as log_dir:
        cb = TokenMilestoneCallback(
            milestones_tokens=[1_000],
            batch_size=10,
            seq_len=10,
            log_dir=log_dir,
            save_checkpoints=False
        )
        assert cb.load_log() == []


def test_multiple_milestones_appended():
    wte_init = _make_wte()
    with tempfile.TemporaryDirectory() as log_dir:
        cb = TokenMilestoneCallback(
            milestones_tokens=[1_000, 2_000, 3_000],
            batch_size=10,
            seq_len=10,
            log_dir=log_dir,
            save_checkpoints=False
        )
        cb._wte_init = wte_init.copy()
        cb._init_hash = wte_hash(wte_init)
        from dataclasses import asdict
        for i, tok in enumerate([1_000, 2_000, 3_000]):
            wte_t = _make_wte(seed=i + 10)
            disp, aniso, effr = compute_embedding_metrics(wte_t, wte_init)
            rec = CheckpointMetrics(
                milestone_tokens=tok,
                step=i * 10,
                train_loss=3.0 - i * 0.2,
                eval_loss=3.1 - i * 0.2,
                embedding_displacement=disp,
                anisotropy=aniso,
                effective_rank=effr,
                init_hash=cb._init_hash
            )
            with open(cb._log_path, "a") as fh:
                fh.write(json.dumps(asdict(rec)) + "\n")
        records = cb.load_log()
        assert len(records) == 3
        assert [r["milestone_tokens"] for r in records] == [1_000, 2_000, 3_000]


def test_toy_pipeline_runs():
    import sys
    import subprocess
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_checkpoint_sweep",
            "--toy-mode"
        ],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, (
        f"toy_mode exited {result.returncode}.\nstderr:\n{result.stderr}"
    )
    assert "TOY MODE COMPLETE" in result.stdout


def test_checkpoint_metrics_includes_seed():
    """CheckpointMetrics must include the seed field."""
    from brain_code.training.checkpoint_logger import CheckpointMetrics
    import dataclasses
    fields = {f.name for f in dataclasses.fields(CheckpointMetrics)}
    assert "seed" in fields, "CheckpointMetrics must have a 'seed' field"
    assert "tokens_seen" in fields, "CheckpointMetrics must have a 'tokens_seen' field"


def test_token_milestone_callback_stores_seed():
    """TokenMilestoneCallback must accept and store a seed parameter."""
    with tempfile.TemporaryDirectory() as log_dir:
        cb = TokenMilestoneCallback(
            milestones_tokens=[1_000],
            batch_size=4,
            seq_len=128,
            log_dir=log_dir,
            save_checkpoints=False,
            seed=99
        )
        assert cb.seed == 99


def test_tokens_seen_matches_step_times_tokens_per_step():
    """tokens_seen recorded in CheckpointMetrics must equal step × tokens_per_step."""
    wte_init = _make_wte()
    with tempfile.TemporaryDirectory() as log_dir:
        cb = TokenMilestoneCallback(
            milestones_tokens=[1_000],
            batch_size=10,
            seq_len=10,
            log_dir=log_dir,
            save_checkpoints=False,
            seed=7
        )
        cb._wte_init = wte_init.copy()
        cb._init_hash = wte_hash(wte_init)
        # milestone 1_000 → step 10, tokens_per_step=100, so tokens_seen=1000
        from brain_code.training.checkpoint_logger import CheckpointMetrics, compute_embedding_metrics
        wte_curr = _make_wte(seed=5)
        disp, aniso, effr = compute_embedding_metrics(wte_curr, wte_init)
        metrics = CheckpointMetrics(
            milestone_tokens=1_000,
            step=10,
            train_loss=2.5,
            eval_loss=None,
            embedding_displacement=disp,
            anisotropy=aniso,
            effective_rank=effr,
            init_hash=cb._init_hash,
            seed=cb.seed,
            tokens_seen=10 * cb.tokens_per_step  # step × tokens_per_step
        )
        assert metrics.tokens_seen == 1_000  # 10 steps × 100 tokens/step
        assert metrics.seed == 7
