"""Trains one model and checkpoints it at exact token milestones (pseudoword and factorial controls).

The seed is applied before the trainer builds any parameters, and every milestone checkpoint is bound
to the result of --alignment_entrypoint. Without an entrypoint the run writes RUN_BLOCKED.json to
--output-dir and stops, unless --allow_unbound_alignment is given; --toy-mode runs a synthetic check.

Example: brain-checkpoint-sweep --dataset-path data/factorial_corpora/condition_A.txt --alignment-entrypoint <mod:fn>
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from brain_code.training.reproducibility import (
    file_or_mapping,
    override_training_seed,
    seed_everything
)


def run_toy(args: argparse.Namespace) -> None:
    """Seeds everything and prints a checksum of synthetic data."""
    state = seed_everything(args.seed, deterministic=args.deterministic)
    rng = np.random.default_rng(args.seed)
    values = rng.standard_normal((16, 8)).astype(np.float32)
    output = {
        "result_status": "pipeline_validation_only",
        "seed": state.seed,
        "deterministic": state.deterministic,
        "toy_checksum": float(values.sum()),
        "scientific_interpretation_allowed": False
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    print("TOY MODE COMPLETE")


def _resolve_named_or_path(value: str, mapping: dict[str, str]) -> str:
    return mapping.get(value, value)


def run_full(args: argparse.Namespace) -> None:
    """Trains with milestone checkpoints, or writes a blocked-run record without an alignment entrypoint."""
    from brain_code.config import PATHS

    model_config_source = _resolve_named_or_path(args.model_args, PATHS["model_configs"])
    training_config_source = _resolve_named_or_path(args.training_args, PATHS["train_configs"])
    training_config = override_training_seed(file_or_mapping(training_config_source), args.seed)
    model_config = file_or_mapping(model_config_source)

    # must run before any model or sampler is constructed
    seed_state = seed_everything(args.seed, deterministic=args.deterministic)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # the command-line directory wins: otherwise the trainer checkpoints to the yaml's
    # shared directory while the callback logs here, breaking provenance and resume
    training_config["output_dir"] = str(output_dir)
    # milestone checkpoints are the only scientific ones; periodic saves with
    # save_total_limit could rotate away one that a neural result is bound to
    training_config["save_strategy"] = "no"
    training_config["save_total_limit"] = None
    training_config["load_best_model_at_end"] = False
    # early stopping requires best-model tracking, which the sweep turns off
    training_config["patience"] = None
    if args.batch_size is not None:
        training_config["per_device_train_batch_size"] = int(args.batch_size)
    if args.seq_len is not None:
        if int(args.seq_len) < 2:
            raise ValueError("--seq-len must be at least 2.")
        model_config["n_positions"] = int(args.seq_len)
        if "n_ctx" in model_config:
            model_config["n_ctx"] = int(args.seq_len)

    # a milestone is only scientific once a frozen alignment protocol is scored on it
    # and bound to its hash; that needs brain-score data this runner cannot supply,
    # so the run stops unless an entrypoint is given
    alignment_fn = _load_alignment_fn(args.alignment_entrypoint) if args.alignment_entrypoint else None

    if alignment_fn is None and not args.allow_unbound_alignment:
        status = {
            "result_status": "blocked_external_dependency",
            "conclusion_status": "rerun_required",
            "seed_applied_before_model_initialization": True,
            "seed": seed_state.seed,
            "deterministic": seed_state.deterministic,
            "actual_nonpadding_token_accounting": True,
            "validation_loss_at_every_milestone": True,
            "neural_alignment_at_every_milestone": False,
            "reason": (
                "Exact token accounting and full checkpoint saving are integrated, but no "
                "--alignment_entrypoint was supplied, so no frozen neural-alignment result can "
                "be bound to the checkpoints. Provide "
                "--alignment_entrypoint module:function on the data box, or pass "
                "--allow_unbound_alignment to record checkpoints with neural_alignment_status="
                "'not_run' for infrastructure debugging only."
            )
        }
        (output_dir / "RUN_BLOCKED.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise RuntimeError(status["reason"])

    from datasets import load_dataset

    from brain_code.training.checkpoint_logger import ExactTokenMilestoneCallback
    from brain_code.training.exact_token_accounting import ExactTokenCounter
    from brain_code.training.token_counting_trainer import TokenCountingTrainer
    from brain_code.training.train_model import LMTrainer

    counter = ExactTokenCounter(
        milestones=args.milestones,
        world_size=int(training_config.get("world_size", 1) or 1)
    )
    dataset = load_dataset("text", data_files=args.dataset_path)
    trainer = LMTrainer(
        dataset=dataset,
        training_args=training_config,
        model_args=model_config,
        dataset_name="checkpoint_sweep"
    )
    callback = ExactTokenMilestoneCallback(
        counter=counter,
        milestones_tokens=args.milestones,
        log_dir=output_dir,
        seed=args.seed,
        alignment_fn=alignment_fn
    )
    trainer.train(
        extra_callbacks=[callback],
        trainer_cls=TokenCountingTrainer,
        trainer_kwargs={"token_counter": counter},
        timestamp_dir=False
    )


def _load_alignment_fn(entrypoint: str):
    """Resolves a ``module:function`` string to a callable ``(ckpt_dir, ckpt_hash) -> dict``."""
    if ":" not in entrypoint:
        raise ValueError("--alignment_entrypoint must be 'module.path:function'.")
    module_path, _, func_name = entrypoint.partition(":")
    module = importlib.import_module(module_path)
    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise ValueError(f"Alignment entrypoint {entrypoint!r} is not callable.")
    return fn


def main() -> None:
    """Runs the toy check or the full sweep."""
    parser = argparse.ArgumentParser(description="Token-milestone checkpoint sweep")
    parser.add_argument("--dataset-path", "--dataset_path", dest="dataset_path")
    parser.add_argument(
        "--model-args",
        "--model_args",
        dest="model_args",
        default="small"
    )
    parser.add_argument(
        "--training-args",
        "--training_args",
        dest="training_args",
        default="default"
    )
    parser.add_argument(
        "--milestones",
        nargs="+",
        type=int,
        default=[100_000, 500_000, 1_000_000]
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int
    )
    parser.add_argument(
        "--seq-len",
        "--seq_len",
        dest="seq_len",
        type=int
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default="results/checkpoint_sweep"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--save-checkpoints",
        "--save_checkpoints",
        dest="save_checkpoints",
        action="store_true",
        help="Accepted and ignored: milestone checkpoints are always saved."
    )
    parser.add_argument(
        "--toy-mode",
        "--toy_mode",
        dest="toy_mode",
        action="store_true"
    )
    parser.add_argument(
        "--alignment-entrypoint",
        "--alignment_entrypoint",
        dest="alignment_entrypoint",
        default=None,
        help="'module.path:function' resolving to (checkpoint_dir, checkpoint_hash) -> dict. "
        "Runs one frozen neural-alignment protocol per milestone and binds it to the "
        "checkpoint hash."
    )
    parser.add_argument(
        "--allow-unbound-alignment",
        "--allow_unbound_alignment",
        dest="allow_unbound_alignment",
        action="store_true",
        help="Infrastructure debugging only: run without an alignment entrypoint; "
        "checkpoints are recorded with neural_alignment_status='not_run'. Never a "
        "scientific result."
    )
    args = parser.parse_args()
    if args.toy_mode:
        run_toy(args)
        return
    if not args.dataset_path:
        parser.error("--dataset-path is required outside toy mode")
    run_full(args)


if __name__ == "__main__":
    main()
