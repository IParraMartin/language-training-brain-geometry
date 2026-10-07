"""Recomputes the packed training-split sizes of the primary GPT-2 runs from the public corpus.

Tokenizes each corpus with the shared tokenizer and replays both packing pipelines (seed 42 and seeds
43-46), writing per-corpus length summaries and per-seed packed sizes to one JSON; nothing is trained.

Example: python scripts/reproduction/recover_packed_splits.py --out results/recovered/packed_splits.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

DATASET_REPO = "IParraMartin/BrainScore-Code-English"
DATASET_REVISION = "cc83566aebbaa2eeb76104cc956e883a722b9ded"
CORPORA = ["tinystories", "python", "source", "assembly", "binary"]
CONTEXT = 1024
MAP_BATCH = 1000
VALIDATION_FRACTION = 0.1
SEED_42_PIPELINE = [42]
SEED_REPLICATE_PIPELINE = [43, 44, 45, 46]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokenizer", default="data/tokenizers/50k_50mb")
    p.add_argument(
        "--corpora",
        nargs="+",
        default=CORPORA,
        choices=CORPORA
    )
    p.add_argument("--num-proc", type=int, default=12)
    return p.parse_args()


def document_lengths(corpus: str, tokenizer_dir: str, num_proc: int) -> np.ndarray:
    """Returns each document's token count after the training-time truncation to 1,024 tokens.

    Args:
        corpus: Dataset configuration name.
        tokenizer_dir: Directory of the shared tokenizer.
        num_proc: Worker processes for tokenization.

    Returns:
        Integer array, one entry per document in dataset order.
    """
    ds = load_dataset(
        DATASET_REPO,
        corpus,
        split="train",
        revision=DATASET_REVISION
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    def lengths(batch: dict) -> dict:
        enc = tokenizer(batch["code"], truncation=True, max_length=CONTEXT)
        return {"n_tokens": [len(ids) for ids in enc["input_ids"]]}

    out = ds.map(
        lengths,
        batched=True,
        batch_size=MAP_BATCH,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        desc=f"tokenizing {corpus}"
    )
    return np.asarray(out["n_tokens"], dtype=np.int64)


def packed_blocks(lengths: np.ndarray, eos: bool) -> int:
    """Counts 1,024-token blocks when documents are packed in consecutive 1,000-document map batches.

    The trainer's batched ``map`` drops each batch's remainder shorter than one block, so packing
    happens per batch rather than over the whole stream.
    """
    per_doc = lengths + (1 if eos else 0)
    n_batches = int(np.ceil(len(per_doc) / MAP_BATCH))
    totals = np.add.reduceat(per_doc, np.arange(n_batches) * MAP_BATCH)
    return int(np.sum(totals // CONTEXT))


def split_sizes(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns the train and validation row indices produced by ``Dataset.train_test_split``."""
    index = Dataset.from_dict({"i": np.arange(n)})
    split = index.train_test_split(test_size=VALIDATION_FRACTION, seed=seed)
    return np.asarray(split["train"]["i"]), np.asarray(split["test"]["i"])


def corpus_record(corpus: str, lengths: np.ndarray) -> dict:
    """Summarizes one corpus under both packing pipelines."""
    record = {
        "n_documents": int(len(lengths)),
        "tokens_after_truncation": int(lengths.sum()),
        "documents_truncated": int(np.sum(lengths >= CONTEXT)),
        "empty_documents": int(np.sum(lengths == 0)),
        "length_quantiles": {
            str(q): float(np.quantile(lengths, q))
            for q in (0.1, 0.5, 0.9, 0.99)
        },
        "lengths_sha256": hashlib.sha256(lengths.tobytes()).hexdigest(),
        "seeds": {}
    }
    all_blocks = packed_blocks(lengths, eos=False)
    for seed in SEED_42_PIPELINE:
        train_idx, test_idx = split_sizes(all_blocks, seed)
        record["seeds"][str(seed)] = {
            "pipeline": "pack_then_split_no_eos",
            "packed_blocks_total": all_blocks,
            "train_blocks": int(len(train_idx)),
            "validation_blocks": int(len(test_idx))
        }
    for seed in SEED_REPLICATE_PIPELINE:
        train_idx, test_idx = split_sizes(len(lengths), seed)
        record["seeds"][str(seed)] = {
            "pipeline": "split_documents_then_pack_with_eos",
            "train_documents": int(len(train_idx)),
            "validation_documents": int(len(test_idx)),
            "train_blocks": packed_blocks(lengths[train_idx], eos=True),
            "validation_blocks": packed_blocks(lengths[test_idx], eos=True)
        }
    return record


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = json.loads(args.out.read_text()) if args.out.exists() else {}
    result.update({
        "dataset": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "tokenizer": args.tokenizer,
        "context_length": CONTEXT,
        "map_batch_size": MAP_BATCH,
        "validation_fraction": VALIDATION_FRACTION
    })
    result.setdefault("corpora", {})
    for corpus in args.corpora:
        if corpus in result["corpora"]:
            print(f"{corpus}: already recovered, skipping")
            continue
        start = time.time()
        lengths = document_lengths(corpus, args.tokenizer, args.num_proc)
        result["corpora"][corpus] = corpus_record(corpus, lengths)
        result["corpora"][corpus]["wall_seconds"] = round(time.time() - start, 1)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"{corpus}: {json.dumps(result['corpora'][corpus]['seeds'])}", flush=True)


if __name__ == "__main__":
    main()
