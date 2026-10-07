"""Trains the shared BPE tokenizer on a byte-capped sample of each corpus.

Streams each dataset, trains a byte-level BPE tokenizer and saves it to
tokenizer/ in the working directory.

Example: brain-train-tokenizer --vocab_size 50000 --bytes_per_dataset 50_000_000
"""

from __future__ import annotations

import argparse

from brain_code.training.train_tokenizer import TrainTokenizer
from brain_code.config import DATASETS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tokenizer pretraining script")
    p.add_argument(
        "--datasets",
        default=[
            DATASETS["python"],
            DATASETS["tinystories"],
            DATASETS["assembly"],
            DATASETS["binary"],
            DATASETS["source"]
        ],
        help="Datasets the BPE tokenizer is trained on."
    )
    p.add_argument(
        "--vocab_size",
        type=int,
        default=50000,
        help="Vocab size of the tokenizer"
    )
    p.add_argument(
        "--bytes_per_dataset",
        type=int,
        default=50_000_000,
        help=(
            "Max bytes drawn from each dataset (shuffled subset). "
            "Lower this to reduce memory and disk usage."
        )
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    trainer = TrainTokenizer(
        args.datasets,
        args.vocab_size,
        args.bytes_per_dataset
    )

    trainer.train()


if __name__ == "__main__":
    main()
