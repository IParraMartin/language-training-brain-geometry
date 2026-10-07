"""Checkpoint callbacks for the token-milestone checkpoint sweep.

Checkpoints are taken at token-count milestones rather than step milestones, so evaluations are comparable
across corpora with different token densities. At each milestone the callbacks record lightweight geometric
metrics on the embedding matrix that need no neural benchmark data:

    eval_loss               held-out language-modelling loss (nats)
    embedding_displacement  ||wte(t) - wte(0)||_F / sqrt(n_params)
    anisotropy              lambda_1 / sum(lambda), 0 isotropic and 1 rank-1
    effective_rank          exp(H(normalised singular values))

Neural alignment is an offline step that needs Brain-Score benchmarks; the JSONL log is meant to be extended
in post-processing.

Temporal co-occurrence of a metric and brain alignment does not imply mediation: a mechanistic claim needs a
causal intervention, not a time-series correlation. The warning is also printed when the
callback starts so it appears in every training log.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from transformers import TrainerCallback

from brain_code.training.exact_token_accounting import ExactTokenCounter


@dataclass
class CheckpointMetrics:
    milestone_tokens: int
    step: int
    train_loss: float | None
    eval_loss: float | None
    embedding_displacement: float | None
    anisotropy: float | None
    effective_rank: float | None
    init_hash: str
    seed: int | None = None
    # nominal step * batch_size * seq_len tokens up to this step
    tokens_seen: int | None = None


def embedding_displacement(wte_current: np.ndarray, wte_init: np.ndarray) -> float:
    """Normalised Frobenius distance from initialisation.

    Divides by sqrt(n_params) so the metric is comparable across model sizes.
    """
    diff = wte_current.astype(np.float64) - wte_init.astype(np.float64)
    return float(np.linalg.norm(diff) / np.sqrt(wte_init.size))


def embedding_anisotropy(wte: np.ndarray) -> float:
    """Fraction of spectral mass in the leading singular direction (0 isotropic, 1 rank-1)."""
    _, sv, _ = np.linalg.svd(wte.astype(np.float32), full_matrices=False)
    sv = sv.astype(np.float64)
    total = sv.sum()
    if total <= 0:
        return float("nan")
    return float(sv[0] / total)


def embedding_effective_rank(wte: np.ndarray) -> float:
    """exp(Shannon entropy of normalised singular values).

    Ranges from 1 (rank-1) to min(n_vocab, d) (perfectly isotropic).
    """
    _, sv, _ = np.linalg.svd(wte.astype(np.float32), full_matrices=False)
    sv = sv.astype(np.float64)
    total = sv.sum()
    if total <= 0:
        return float("nan")
    probs = sv / total
    h = float(-np.sum(probs * np.log(probs + 1e-15)))
    return float(np.exp(h))


def wte_hash(wte: np.ndarray) -> str:
    """Deterministic 8-char sha256 hex digest of wte (float32 bytes)."""
    return hashlib.sha256(wte.astype(np.float32).tobytes()).hexdigest()[:8]


def compute_embedding_metrics(wte_current: np.ndarray, wte_init: np.ndarray) -> tuple[float, float, float]:
    """Returns (displacement, anisotropy, effective_rank), all nan if the SVD fails."""
    try:
        disp = embedding_displacement(wte_current, wte_init)
        aniso = embedding_anisotropy(wte_current)
        effr = embedding_effective_rank(wte_current)
    except np.linalg.LinAlgError:
        disp, aniso, effr = float("nan"), float("nan"), float("nan")
    return disp, aniso, effr


class TokenMilestoneCallback(TrainerCallback):
    """Trainer callback that fires at nominal token-count milestones.

    Saves checkpoints and appends a CheckpointMetrics record to a JSONL log at each milestone. Milestones are
    given in total tokens consumed, so runs on corpora with different token densities are comparable.

    Args:
        milestones_tokens: Token-count milestones, e.g. [100_000, 500_000, 1_000_000].
        batch_size: Per-device batch size times number of devices; gradient accumulation is not included,
            so pass the effective batch size if it matters.
        seq_len: Sequence length in tokens.
        log_dir: Directory where checkpoint_metrics.jsonl is written.
        save_checkpoints: Whether to also save model weights at each milestone; False for quick debugging.
        seed: Training seed, recorded with every milestone.
    """

    def __init__(
        self,
        milestones_tokens: list[int],
        batch_size: int,
        seq_len: int,
        log_dir: str | Path,
        save_checkpoints: bool = True,
        seed: int | None = None
    ) -> None:
        self.milestones_tokens = sorted(milestones_tokens)
        self.tokens_per_step = batch_size * seq_len
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_checkpoints = save_checkpoints
        self.seed = seed

        # milestones that map to the same step collapse onto the last one
        self.milestone_steps: dict[int, int] = {
            max(1, int(np.ceil(m / self.tokens_per_step))): m
            for m in self.milestones_tokens
        }
        self._triggered: set[int] = set()
        self._wte_init: np.ndarray | None = None
        self._init_hash: str = ""
        self._log_path = self.log_dir / "checkpoint_metrics.jsonl"

        print(
            "[CheckpointLogger] INTERPRETATION WARNING: temporal association between "
            "a metric and brain alignment does NOT imply mediation. "
            "Causal claims require intervention experiments."
        )
        print(f"[CheckpointLogger] {len(self.milestones_tokens)} milestones: {self.milestones_tokens}")

    def on_train_begin(
        self,
        args,
        state,
        control,
        model=None,
        **kwargs
    ):
        """Records the initial embedding matrix and its hash."""
        if model is not None:
            try:
                wte = model.transformer.wte.weight.detach().cpu().float().numpy()
                self._wte_init = wte.copy()
                self._init_hash = wte_hash(wte)
                print(f"[CheckpointLogger] Init hash: {self._init_hash}  wte shape: {wte.shape}")
            except AttributeError as exc:
                print(f"[CheckpointLogger] WARNING: could not access wte: {exc}")

    def on_step_end(
        self,
        args,
        state,
        control,
        model=None,
        **kwargs
    ):
        step = state.global_step
        milestone_tokens = self.milestone_steps.get(step)
        if milestone_tokens is None or step in self._triggered:
            return control
        self._triggered.add(step)
        self._record(
            step,
            milestone_tokens,
            state,
            model,
            args
        )
        return control

    def _record(
        self,
        step: int,
        milestone_tokens: int,
        state,
        model,
        args
    ) -> None:
        print(f"[CheckpointLogger] Milestone {milestone_tokens:,} tokens (step {step})")
        train_loss = None
        for entry in reversed(state.log_history):
            if "loss" in entry:
                train_loss = entry["loss"]
                break

        metrics = CheckpointMetrics(
            milestone_tokens=milestone_tokens,
            step=step,
            train_loss=train_loss,
            eval_loss=None,
            embedding_displacement=None,
            anisotropy=None,
            effective_rank=None,
            init_hash=self._init_hash,
            seed=self.seed,
            tokens_seen=step * self.tokens_per_step
        )

        if model is not None and self._wte_init is not None:
            # a metric failure should not stop training; it is logged and the fields stay None
            try:
                wte = model.transformer.wte.weight.detach().cpu().float().numpy()
                disp, aniso, effr = compute_embedding_metrics(wte, self._wte_init)
                metrics.embedding_displacement = disp
                metrics.anisotropy = aniso
                metrics.effective_rank = effr
            except Exception as exc:  # noqa: BLE001
                print(f"[CheckpointLogger] WARNING: embedding metrics failed: {exc}")

        with open(self._log_path, "a") as fh:
            fh.write(json.dumps(asdict(metrics)) + "\n")

        if self.save_checkpoints and model is not None and hasattr(args, "output_dir"):
            ckpt_dir = Path(args.output_dir) / f"checkpoint-tokens-{milestone_tokens}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            print(f"[CheckpointLogger] Saved → {ckpt_dir}")

    def load_log(self) -> list[dict]:
        """Returns all milestone records from the JSONL log."""
        if not self._log_path.exists():
            return []
        return [
            json.loads(line)
            for line in self._log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


def hash_checkpoint_dir(checkpoint_dir: str | Path) -> str:
    """Deterministic sha256 over every file in a saved checkpoint directory.

    Hashes relative paths and byte contents in sorted order, so a full checkpoint (model, optimizer,
    scheduler, scaler, RNG, trainer_state) maps to one stable identifier that a neural-alignment result can
    be bound to.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    root = Path(checkpoint_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {root}")
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass
class ExactCheckpointMetrics:
    """Milestone record bound to exact token accounting and a checkpoint hash.

    ``neural_alignment_status`` stays ``"not_run"`` (and ``neural_alignment`` None) unless an alignment
    function is supplied and succeeds, so the record never carries a brain-alignment number that was not
    measured.
    """

    milestone_tokens: int
    actual_nonpadding_tokens: int
    token_overshoot: int
    global_step: int
    data_position: int | None
    eval_loss: float | None
    checkpoint_dir: str | None
    checkpoint_hash: str | None
    neural_alignment_status: str
    neural_alignment: dict | None
    error: str | None
    seed: int | None
    init_hash: str | None


class ExactTokenMilestoneCallback(TrainerCallback):
    """Checkpoints at the first optimizer step after a token threshold is crossed.

    Where :class:`TokenMilestoneCallback` uses the nominal step * batch * seq count, this reads a shared
    :class:`ExactTokenCounter` fed by :class:`TokenCountingTrainer`. The cumulative non-padding count is
    exact, but a checkpoint can overshoot a threshold by the tokens of the final optimizer step; the
    overshoot is recorded. The callback asks the Trainer to save its complete checkpoint (model, tokenizer,
    optimizer, scheduler, scaler, RNG and trainer state), evaluates it, hashes it and binds a
    neural-alignment result to that hash.

    Args:
        counter: The shared ExactTokenCounter, the same instance attached to the trainer.
        milestones_tokens: Token-count milestones; should match ``counter.milestones``.
        log_dir: Directory for ``exact_checkpoint_metrics.jsonl``.
        seed: Training seed, recorded for provenance.
        alignment_fn: Optional ``(checkpoint_dir, checkpoint_hash) -> dict`` that runs one frozen
            neural-alignment protocol. Without it the record is written with
            ``neural_alignment_status="not_run"``, so nothing downstream can read an alignment that never
            happened.
    """

    def __init__(
        self,
        counter: ExactTokenCounter,
        milestones_tokens: list[int],
        log_dir: str | Path,
        seed: int | None = None,
        alignment_fn: Callable[[str, str], dict] | None = None
    ) -> None:
        self.counter = counter
        self.milestones_tokens = sorted(int(m) for m in milestones_tokens)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.alignment_fn = alignment_fn
        self._handled: set[int] = set()
        self._pending: list[int] = []
        self._wte_init: np.ndarray | None = None
        self._init_hash: str = ""
        self._log_path = self.log_dir / "exact_checkpoint_metrics.jsonl"

    def on_train_begin(
        self,
        args,
        state,
        control,
        model=None,
        **kwargs
    ):
        if model is not None:
            try:
                wte = model.transformer.wte.weight.detach().cpu().float().numpy()
                self._wte_init = wte.copy()
                self._init_hash = wte_hash(wte)
            except AttributeError:
                pass
        return control

    def on_step_end(
        self,
        args,
        state,
        control,
        **kwargs
    ):
        """Requests a save and an evaluation once the exact token count crosses a milestone."""
        seen = self.counter.tokens_seen
        crossed = [m for m in self.milestones_tokens if m <= seen and m not in self._handled]
        if crossed:
            # defer to the Trainer's own complete checkpoint save and evaluation
            self._pending.extend(crossed)
            self._handled.update(crossed)
            control.should_evaluate = True
            control.should_save = True
        return control

    def on_save(
        self,
        args,
        state,
        control,
        model=None,
        **kwargs
    ):
        """Hashes the checkpoint the Trainer just wrote and binds the alignment result to it."""
        if not self._pending:
            return control
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.is_dir():
            raise RuntimeError(
                "Trainer requested a milestone save but the expected complete checkpoint "
                f"does not exist: {checkpoint_dir}"
            )
        ckpt_hash = hash_checkpoint_dir(checkpoint_dir)

        eval_loss = None
        for entry in reversed(state.log_history):
            if "eval_loss" in entry and (
                "step" not in entry or int(entry["step"]) == int(state.global_step)
            ):
                eval_loss = float(entry["eval_loss"])
                break
        if eval_loss is None:
            raise RuntimeError(
                "Milestone checkpoint was saved without a recorded eval_loss. "
                "The checkpoint sweep requires evaluation at every checkpoint."
            )

        alignment_status = "not_run"
        alignment_result: dict | None = None
        alignment_error: str | None = None
        if self.alignment_fn is not None:
            # the failure is recorded in the log first and re-raised below
            try:
                alignment_result = dict(self.alignment_fn(str(checkpoint_dir), ckpt_hash))
                returned_hash = alignment_result.get("checkpoint_hash")
                if returned_hash is not None and returned_hash != ckpt_hash:
                    raise ValueError(
                        "Neural-alignment result returned a checkpoint_hash that does "
                        "not match the checkpoint it scored."
                    )
                alignment_result["checkpoint_hash"] = ckpt_hash
                alignment_status = "bound"
            except Exception as exc:  # noqa: BLE001
                alignment_status = f"failed: {type(exc).__name__}"
                alignment_error = f"{type(exc).__name__}: {exc}"

        # the generic Trainer state exposes no stable sampler offset, and global_step is not a data position
        data_position = None

        for milestone in self._pending:
            record = ExactCheckpointMetrics(
                milestone_tokens=milestone,
                actual_nonpadding_tokens=int(self.counter.tokens_seen),
                token_overshoot=int(self.counter.tokens_seen) - int(milestone),
                global_step=int(state.global_step),
                data_position=data_position,
                eval_loss=eval_loss,
                checkpoint_dir=str(checkpoint_dir),
                checkpoint_hash=ckpt_hash,
                neural_alignment_status=alignment_status,
                neural_alignment=alignment_result,
                error=alignment_error,
                seed=self.seed,
                init_hash=self._init_hash or None
            )
            with open(self._log_path, "a") as fh:
                fh.write(json.dumps(asdict(record)) + "\n")
            print(
                f"[ExactCheckpoint] milestone={milestone:,} "
                f"actual_tokens={self.counter.tokens_seen:,} step={state.global_step} "
                f"hash={ckpt_hash[:8]} alignment={alignment_status}"
            )
        self._pending.clear()
        if alignment_error is not None:
            raise RuntimeError("Neural alignment failed for a milestone checkpoint: " + alignment_error)
        return control

    def load_log(self) -> list[dict]:
        """Returns all milestone records from the JSONL log."""
        if not self._log_path.exists():
            return []
        return [
            json.loads(line)
            for line in self._log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
