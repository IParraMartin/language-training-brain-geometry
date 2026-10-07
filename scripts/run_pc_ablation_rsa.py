"""Tests whether a model's brain RSA rides on a few dominant directions by removing leading principal components.

RSA can reward a collapsed manifold (binary, participation ratio about 3) whose distances track one
variable. For every model and block this removes the top m principal directions (fitted on all stimuli,
as the comparison is descriptive) and recomputes RSA against the brain. Writes one JSON to --out.

Example: python scripts/run_pc_ablation_rsa.py --models-dir <seed_models> --benchmark pereira_243 --out pc_ablation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from brain_code.experiments.representational_similarity import column_center, rdm, rsa_from_rdms
from scripts.run_latent_geometry import (
    BENCHMARKS,
    build_passage_contexts,
    discover_models,
    extract_last_token_states,
    load_stimuli
)

DEFAULT_M = (0, 1, 2, 3, 5)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", type=Path, required=True)
    p.add_argument(
        "--hf-model",
        action="append",
        default=[],
        metavar="NAME=HUB_ID"
    )
    p.add_argument("--benchmark", default="pereira_243", choices=sorted(BENCHMARKS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument(
        "--remove",
        type=int,
        nargs="+",
        default=list(DEFAULT_M),
        help="Counts of leading principal directions to remove."
    )
    return p.parse_args(argv)


def remove_top_pcs(X: np.ndarray, m: int) -> np.ndarray:
    """Project out the m leading principal directions of X."""
    Xc = column_center(X)
    if m <= 0:
        return Xc
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    directions = Vt[:m]
    return Xc - (Xc @ directions.T) @ directions


def corpus_of(model: str) -> str:
    return model.rsplit("_seed", 1)[0]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = BENCHMARKS[args.benchmark]
    sentences, brain, presentation, _neuroid = load_stimuli(args.benchmark)
    contexts = build_passage_contexts(presentation, spec["group"])
    brain_metric = "correlation" if brain.shape[1] >= 3 else "euclidean"
    brain_rdm = rdm(brain, brain_metric)
    print(
        f"benchmark {args.benchmark}: {len(sentences)} stimuli, "
        f"{brain.shape[1]} neuroids, brain RDM metric {brain_metric}",
        flush=True
    )

    models = discover_models(args.models_dir, args.hf_model, None)
    results: dict[str, dict[str, dict[str, float]]] = {}
    for name, ref in models:
        states = extract_last_token_states(
            ref,
            contexts,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        per_layer = {}
        for layer, X in states.items():
            per_layer[layer] = {str(m): rsa_from_rdms(rdm(remove_top_pcs(X, m)), brain_rdm) for m in args.remove}
        results[name] = per_layer
        peak = {m: max(v[str(m)] for v in per_layer.values()) for m in args.remove}
        print("  " + name.ljust(22) + "  ".join(f"m={m}:{peak[m]:+.4f}" for m in args.remove), flush=True)

    layers = sorted(next(iter(results.values())), key=lambda s: int(s.rsplit(".", 1)[1]))
    corpora = sorted({corpus_of(m) for m in results})
    print("\npeak RSA over layers, by corpus (mean over seeds)")
    print("corpus".ljust(14) + "".join(f"m={m}".rjust(10) for m in args.remove))
    summary = {}
    for c in corpora:
        ms = [m for m in results if corpus_of(m) == c]
        row = {}
        for m in args.remove:
            per_model = [max(results[k][layer][str(m)] for layer in layers) for k in ms]
            row[str(m)] = float(np.mean(per_model))
        summary[c] = row
        print(c.ljust(14) + "".join(f"{row[str(m)]:10.4f}" for m in args.remove))

    eng, bin_ = summary.get("tinystories"), summary.get("binary")
    if eng and bin_:
        print("\nEnglish minus binary")
        for m in args.remove:
            d = eng[str(m)] - bin_[str(m)]
            print(f"  remove top {m}: {d:+.4f}  ({'English leads' if d > 0 else 'binary leads'})")

    payload = {
        "schema": "pc_ablation_rsa",
        "benchmark": spec["id"],
        "benchmark_key": args.benchmark,
        "brain_rdm_metric": brain_metric,
        "removed_counts": list(args.remove),
        "result_status": "exploratory_only",
        "note": (
            "Projector fitted on all stimuli, which is legitimate here: this "
            "is a descriptive geometry comparison with no held-out prediction, "
            "so there is no test set for an unsupervised fit to leak into."
        ),
        "per_model": results,
        "peak_by_corpus": summary
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
