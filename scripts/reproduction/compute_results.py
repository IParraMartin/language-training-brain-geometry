"""Computes every reported statistic from the results bundle.

Needs only the bundle (no checkpoints, GPUs or neural data). Writes results.json with every statistic,
training_exposure.csv with one row per primary training run, and corpus_provenance.json with the corpus
audit and the recovered packed-split sizes. Existing outputs in --out are overwritten.

Example: python scripts/reproduction/compute_results.py --data data/results_bundle --out results/summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from brain_code.reproduction.analysis import all_results, to_json
from brain_code.reproduction.bundle import Bundle
from brain_code.reproduction.config import DATASET_REVISION, load_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="results bundle (default: results_bundle in configs/paths.yaml)"
    )
    p.add_argument("--out", type=Path, default=Path("results/summary"))
    return p.parse_args()


def exposure_rows(results: dict, bundle: Bundle) -> list[dict]:
    """One row per primary training run, with the recipe, recovered exposure and missing fields."""
    provenance = bundle.table("training_provenance").set_index("run_id")
    rows = []
    for run in results["exposure"]["runs"]:
        key = f"{run['corpus']}_seed{run['seed']}"
        logged = key in provenance.index
        record = provenance.loc[key] if logged else None
        rows.append({
            **run,
            "architecture": "GPT-2 small, 12 blocks, 12 heads, width 768, context 1024, dropout 0.1",
            "optimizer": "AdamW lr 1e-4, betas (0.9, 0.999), weight decay 0.01, clip 1.0, 1000 warmup, cosine",
            "precision": "bf16" if not logged or bool(record["bf16"]) else "fp32",
            "batch": "16 sequences x 1024 tokens, no gradient accumulation",
            "validation": "every 1000 updates, eval_loss, patience 5, best weights restored",
            "evaluated_checkpoint": "best validation checkpoint",
            "initial_parameter_hash": record["initial_parameter_hash"] if logged else None,
            "final_parameter_hash": record["final_parameter_hash"] if logged else None,
            "tokenizer_hash": record["tokenizer_hash"] if logged else None,
            "dataset_revision": DATASET_REVISION,
            "device": "not recorded in the provenance file",
            "gpu_hours": "not recorded",
            "missing": "" if logged else (
                "training log, step count, selected checkpoint and parameter hashes not retained; "
                "pass equivalents assume the 30,000-update checkpoint"
            )
        })
    return rows


def main() -> None:
    args = parse_args()
    data = args.data or load_paths().results_bundle
    bundle = Bundle(data)
    results = all_results(bundle)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(to_json(results) + "\n")
    pd.DataFrame(exposure_rows(results, bundle)).to_csv(args.out / "training_exposure.csv", index=False)
    provenance = {
        "audit": bundle.json("corpus_audit"),
        "packed_splits": bundle.json("packed_splits")
    }
    (args.out / "corpus_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote results.json, training_exposure.csv and corpus_provenance.json to {args.out}")


if __name__ == "__main__":
    main()
