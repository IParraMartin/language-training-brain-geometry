"""Pretrains a GPT-2 model on a hub corpus or a local plain-text file.

--dataset takes a key from DATASETS in src/brain_code/config.py and --dataset_path a file with one
document per line, such as the factorial corpora. --output_dir, --seed, --max_steps and
--run_name override the training YAML, so a sweep needs no config file per cell.

Example: brain-train --dataset python --model_args small --training_args python
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from brain_code.training.train_model import LMTrainer  # sets HF_HOME before any HF imports
from brain_code.config import PATHS, DATASETS

from datasets import load_dataset, DatasetDict


def parse_args() -> argparse.Namespace:
    """Parses the command-line arguments."""
    p = argparse.ArgumentParser(description="Pretraining script.")
    p.add_argument(
        "--tokenizer",
        type=lambda k: PATHS["tokenizer"][k],
        default="default",
        help=f"Tokenizer key from config.py. One of: {list(PATHS['tokenizer'])}."
    )
    p.add_argument(
        "--dataset",
        type=lambda k: DATASETS[k],
        default=None,
        help=f"Dataset key from config.py DATASETS. One of: {list(DATASETS)}. "
             "Ignored when --dataset_path is given; defaults to 'tinystories' "
             "if neither is supplied."
    )
    p.add_argument(
        "--dataset-path",
        "--dataset_path",
        dest="dataset_path",
        default=None,
        help="Path to a local plain-text corpus (one document per line). Takes "
             "precedence over --dataset. Used for the factorial conditions."
    )
    p.add_argument(
        "--training-args",
        "--training_args",
        dest="training_args",
        type=lambda k: PATHS["train_configs"][k],
        default="default",
        help=f"Training config key from config.py. One of: {list(PATHS['train_configs'])}."
    )
    p.add_argument(
        "--model-args",
        "--model_args",
        dest="model_args",
        type=lambda k: PATHS["model_configs"][k],
        default="small",
        help=f"Model size key from config.py. One of: {list(PATHS['model_configs'])}."
    )
    p.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default=None,
        help="Override output_dir from the training config (per-run checkpoints)."
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed from the training config (for multi-seed sweeps)."
    )
    p.add_argument(
        "--max-steps",
        "--max_steps",
        dest="max_steps",
        type=int,
        default=None,
        help="Override max_steps from the training config."
    )
    p.add_argument(
        "--run-name",
        "--run_name",
        dest="run_name",
        default=None,
        help="Override run_name from the training config (the W&B run label). "
             "Without this, every experiment sharing a training config reports "
             "under that config's name."
    )
    p.add_argument(
        "--null-pos",
        "--null_pos",
        dest="null_pos",
        action="store_true",
        help="If set, makes the transformer positionally invariant by nulling position embeddings."
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the latest checkpoint-N/ in --output-dir instead of restarting from step 0; "
             "starts fresh when the directory holds no checkpoint."
    )
    return p.parse_args()


def _resolve_training_args(
    config_path: str,
    output_dir: str | None,
    seed: int | None,
    max_steps: int | None,
    run_name: str | None
) -> dict:
    """Load the training YAML and apply CLI overrides.

    Args:
        config_path: Path to the training YAML.
        output_dir: Replacement ``output_dir``, or ``None`` to keep the config's.
        seed: Replacement ``seed``, or ``None`` to keep the config's.
        max_steps: Replacement ``max_steps``, or ``None`` to keep the config's.
        run_name: Replacement ``run_name``, or ``None`` to keep the config's.

    Returns:
        Training arguments with overrides applied.
    """
    cfg = yaml.safe_load(Path(config_path).read_text()) or {}
    overrides = (
        ("output_dir", output_dir),
        ("seed", seed),
        ("max_steps", max_steps),
        ("run_name", run_name)
    )
    for key, value in overrides:
        if value is not None:
            cfg[key] = value
    return cfg


def main() -> None:
    """Load the corpus, build the trainer, and run pretraining."""
    args = parse_args()

    if args.dataset_path:
        path = Path(args.dataset_path)
        if not path.exists():
            raise SystemExit(f"--dataset_path not found: {path}")
        ds = load_dataset("text", data_files=str(path))
        dataset_name = path.stem
    else:
        # tinystories when neither source is given
        dataset_spec = args.dataset or DATASETS["tinystories"]
        ds = load_dataset(dataset_spec[0], dataset_spec[1], split="train")
        dataset_name = dataset_spec[1]

    if not isinstance(ds, DatasetDict):
        ds = DatasetDict({"train": ds})

    training_args = _resolve_training_args(
        args.training_args,
        args.output_dir,
        args.seed,
        args.max_steps,
        args.run_name
    )

    trainer = LMTrainer(
        tokenizer=args.tokenizer,
        dataset=ds,
        training_args=training_args,
        model_args=args.model_args,
        dataset_name=dataset_name,
        null_pos=args.null_pos
    )

    # an explicit --output_dir is a path the caller scores right after training, so no
    # timestamped subdirectory; without it, timestamps keep runs from colliding
    if args.resume and args.output_dir is None:
        raise SystemExit("--resume requires --output-dir: a timestamped directory never holds a checkpoint.")
    out_dir = trainer.train(timestamp_dir=args.output_dir is None, resume=args.resume)
    print(f"Model directory: {out_dir}")


if __name__ == "__main__":
    main()
