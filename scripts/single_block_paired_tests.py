"""Tests each single-block or embedding transplant against its untouched receiver, paired across benchmark folds.

Reads per-fold scores of the transplanted models (scripts/score_local_checkpoints.py on the brain-patch
outputs) and of the receivers (scripts/extract_layer_cis.py), runs a two-sided paired t-test per readout
block and patch position, and applies Holm correction across the 13 patch positions within each
(receiver, readout block). An optional noise band marks differences larger than it. Writes one JSON.

Example: python -m scripts.single_block_paired_tests --transplants t.json --baseline layer_cis.json --output p.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from brain_code.reproduction.inference import holm

RECEIVERS = ["assembly", "binary", "python", "source"]
SHORT = {
    "assembly": "Assembly",
    "binary": "Binary",
    "python": "Python",
    "source": "C"
}
POSITIONS = ["embeddings"] + [str(i) for i in range(12)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--transplants",
        type=Path,
        required=True,
        help="per-fold scores of the transplanted models"
    )
    p.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="per-fold scores of the seed-42 receivers"
    )
    p.add_argument(
        "--band-tests",
        type=Path,
        default=None,
        help="JSON with an empirical_noise_band entry, applied as an external reference band"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def transplant_label(receiver: str, position: str) -> str:
    """Model label of one transplant, as brain-patch names its output directory."""
    if position == "embeddings":
        return f"exp11_gpt2-{receiver}_patch_embeddings"
    return f"exp11_gpt2-{receiver}_patch_block_L{position}"


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired two-sided t-test of a minus b across folds, with the paired effect size."""
    diff = a - b
    result = stats.ttest_rel(a, b)
    sd = float(np.std(diff, ddof=1))
    return {
        "mean_difference": float(np.mean(diff)),
        "t": float(result.statistic),
        "p_uncorrected": float(result.pvalue),
        "cohens_d_paired": float(np.mean(diff) / sd) if sd > 0 else float("nan"),
        "n_folds": int(a.size)
    }


def main() -> None:
    args = parse_args()
    transplants = json.loads(args.transplants.read_text())["models"]
    baseline = json.loads(args.baseline.read_text())["models"]
    band = None
    out: dict = {
        "alpha": args.alpha,
        "test": "paired two-sided t-test across benchmark CV folds",
        "correction": "Holm-Bonferroni across the 13 patch positions within each (receiver, readout layer)",
        "contrast": "patched model (no LayerNorm re-adaptation) minus its own receiver baseline",
        "receivers": {}
    }
    if args.band_tests is not None:
        band_source = json.loads(args.band_tests.read_text())["empirical_noise_band"]
        band = float(band_source["value"])
        out["band_provenance"] = {
            "value": band,
            "source": "measured on the LayerNorm-adapted transplant controls, imported here",
            **{k: v for k, v in band_source.items() if k != "value"}
        }
    layer_names = sorted(next(iter(transplants.values())), key=lambda name: int(name.rsplit(".", 1)[-1]))

    summary = []
    for receiver in RECEIVERS:
        receiver_folds = baseline[f"IParraMartin/gpt2-{receiver}"]
        per_layer = {}
        counts = {
            "significant": 0,
            "negative": 0,
            "over_band": 0
        }
        for layer in layer_names:
            rows = []
            for position in POSITIONS:
                a = np.asarray(transplants[transplant_label(receiver, position)][layer]["fold_scores"], dtype=float)
                b = np.asarray(receiver_folds[layer]["fold_scores"], dtype=float)
                if a.shape != b.shape:
                    raise SystemExit(f"fold mismatch, {receiver} {position} {layer}")
                rows.append({"patch_position": position, **paired(a, b)})
            for row, p_adjusted in zip(rows, holm([r["p_uncorrected"] for r in rows])):
                row["p_holm"] = float(p_adjusted)
                row["significant"] = bool(p_adjusted < args.alpha)
                if band is not None:
                    row["exceeds_band"] = abs(row["mean_difference"]) > band
                    counts["over_band"] += row["exceeds_band"]
                if row["significant"]:
                    counts["significant"] += 1
                    counts["negative"] += row["mean_difference"] < 0
            per_layer[layer] = rows
        out["receivers"][SHORT[receiver]] = per_layer
        summary.append({
            "receiver": SHORT[receiver],
            "cells_significant": counts["significant"],
            "of_which_negative": counts["negative"],
            "cells_exceeding_band": counts["over_band"] if band is not None else None,
            "cells_total": len(layer_names) * len(POSITIONS)
        })
    out["summary"] = summary
    out["summary_total"] = {
        "cells_significant": sum(s["cells_significant"] for s in summary),
        "of_which_negative": sum(s["of_which_negative"] for s in summary),
        "cells_total": sum(s["cells_total"] for s in summary)
    }
    if band is not None:
        out["summary_total"]["cells_exceeding_band"] = sum(s["cells_exceeding_band"] for s in summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    total = out["summary_total"]
    significant, negative = total["cells_significant"], total["of_which_negative"]
    print(f"{significant}/{total['cells_total']} cells significant ({negative} negative)")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
