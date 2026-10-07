"""Writes one corpus of the pinned dataset revision as a one-document-per-line text file.

brain-build-corpora and brain-checkpoint-sweep read local text files, so the pseudoword and factorial
corpora start from this dump of TinyStories. The parquet shards of the pinned revision are read in
sorted order; whitespace inside a document is collapsed so a newline never splits it, and empty
documents are skipped. Fails when the dump is smaller than --min-bytes.

Example: python -m scripts.dump_corpus --dataset-key tinystories --out data/factorial_corpora/tinystories_corpus.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from brain_code.config import DATASETS
from brain_code.reproduction.config import DATASET_REVISION


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-key", required=True, choices=sorted(DATASETS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--target-bytes",
        type=int,
        default=0,
        help="stop after about this many bytes; 0 dumps everything"
    )
    p.add_argument("--min-bytes", type=int, default=60_000_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    repo_id, config = DATASETS[args.dataset_key]
    snapshot = Path(snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=DATASET_REVISION,
        allow_patterns=[f"data/{config}-*.parquet"]
    ))
    shards = sorted(str(p) for p in (snapshot / "data").glob(f"{config}-*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet shards for config {config!r} under {snapshot / 'data'}")
    print(f"{repo_id}:{config} -> {len(shards)} shards", flush=True)
    dataset = load_dataset(
        "parquet",
        data_files=shards,
        split="train",
        streaming=True
    )

    # the schema is (idx, language, code): documents are in "code", and "language" is a per-row label
    field = "code"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = n_docs = n_skipped = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for row in dataset:
            line = " ".join((row.get(field) or "").split())
            if not line:
                n_skipped += 1
                continue
            handle.write(line + "\n")
            written += len(line) + 1
            n_docs += 1
            if args.target_bytes and written >= args.target_bytes:
                break
    print(f"wrote {n_docs:,} documents, {written / 1e6:.1f} MB -> {args.out}; skipped {n_skipped:,} empty", flush=True)
    if written < args.min_bytes:
        raise SystemExit(f"only {written / 1e6:.1f} MB, below the {args.min_bytes / 1e6:.0f} MB floor")


if __name__ == "__main__":
    main()
