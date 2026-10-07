"""Fine-tunes only the input embeddings of a causal LM on another corpus.

Saves the adapted model under --out_dir (results/models/ablated_model/ by default) and can push it
to the Hub.

Example: brain-embed-ft --model_id IParraMartin/gpt2-python --dataset tinystories
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from brain_code.config import DATASETS
from brain_code.training.module_adaptors import run_embedding_ablation


def parse_args() -> argparse.Namespace:
    """Parses the command-line arguments."""
    p = argparse.ArgumentParser(
        description="Fine-tune only the input embeddings of a causal LM on English text."
    )
    p.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace Hub model ID (e.g. 'IParraMartin/gpt2-python') or local checkpoint path."
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Name of the training dataset to use for embedding fine-tuning. "
            f"Available options: {', '.join(DATASETS.keys())}."
        )
    )
    p.add_argument(
        "--text_column",
        type=str,
        default="code",
        help="Column name to extract from the dataset (default: 'code')."
    )
    p.add_argument(
        "--n_steps",
        type=int,
        default=2000,
        help="Number of gradient update steps (default: 2000)."
    )
    p.add_argument(
        "--max_tokens",
        type=int,
        default=2_000_000,
        help="Number of tokens to use (default: 2M)."
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate (default: 1e-3)."
    )
    p.add_argument(
        "--context_length",
        type=int,
        default=64,
        help="Token sequence length per training chunk (default: 64)."
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size (default: 32)."
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for the saved model (default: results/models/ablated_model/)."
    )
    p.add_argument(
        "--push",
        action="store_true",
        help="Push the ablated model and tokenizer to the HuggingFace Hub after saving."
    )
    return p.parse_args()


def main() -> None:
    """Loads the training dataset and runs the adaptation."""
    args = parse_args()
    repo_id, config_name = DATASETS[args.dataset]
    dataset = load_dataset(repo_id, config_name)
    run_embedding_ablation(
        model_id=args.model_id,
        english_texts=dataset,
        n_steps=args.n_steps,
        max_tokens=args.max_tokens,
        lr=args.lr,
        context_length=args.context_length,
        batch_size=args.batch_size,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        text_column=args.text_column,
        split="train",
        push_to_hub_=args.push
    )


if __name__ == "__main__":
    main()
