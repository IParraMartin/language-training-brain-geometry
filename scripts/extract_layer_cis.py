"""Computes per-layer raw Brain-Score with per-fold scores and cross-fold 95% intervals for Hub models.

Scores each layer through the official source assembly and the pinned metric's recording wrapper, keeps
each CV fold's mean neuroid correlation and reports a t-based 95% interval across folds, raw and
ceiling-normalized. Benchmarks without a ceiling (Tuckute2024) get null normalized values. Writes JSON
to --output; this is a reporting aid, not a certified result.

Example: python -m scripts.extract_layer_cis --output layer_cis.json --models IParraMartin/gpt2-python=<sha>
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# student-t 0.975 quantiles by df = n_folds - 1 (pereira uses 10 folds); other df fall back to 1.96
_T975 = {
    4: 2.776,
    8: 2.306,
    9: 2.262,
    14: 2.145,
    19: 2.093
}


def benchmark_ceiling(benchmark) -> float | None:
    """The benchmark's ceiling, or None for a benchmark that has none."""
    raw = getattr(benchmark, "ceiling", None)
    return None if raw is None else float(np.asarray(raw.values).reshape(-1)[0])


def score_layers(
    benchmark,
    model,
    tokenizer,
    label: str,
    n_layers: int,
    ceiling: float | None
) -> dict[str, dict]:
    """Scores every block of one model on the benchmark's own folds.

    Args:
        benchmark: A loaded Brain-Score benchmark.
        model: Causal LM in evaluation mode.
        tokenizer: Its tokenizer, loaded with ``truncation_side="left"`` as the harness does.
        label: Model identifier passed to the subject and printed.
        n_layers: Number of transformer blocks to score.
        ceiling: Benchmark ceiling, or None to leave normalized values null.

    Returns:
        Block name -> raw and normalized center and interval, fold count and per-fold scores (the
        mean over neuroids within each fold).
    """
    from brainscore_language import ArtificialSubject
    from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

    import scripts.export_pereira_parity as ex
    from brain_code.experiments.pereira_parity_export import build_recording_metric

    def normalized(x: float) -> float | None:
        return None if ceiling in (None, 0) else x / ceiling

    per_layer: dict[str, dict] = {}
    for i in range(n_layers):
        layer = f"transformer.h.{i}"
        candidate = HuggingfaceSubject(
            model_id=label,
            model=model,
            tokenizer=tokenizer,
            region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: [layer]}
        )
        source = ex._official_source(benchmark, candidate)
        recording, records = build_recording_metric(benchmark.metric)
        recording(source, benchmark.data)
        fold_scores = np.asarray([
            float(np.nanmean(np.asarray(f, dtype=float)))
            for f in records.per_split_correlations
        ])
        n = len(fold_scores)
        center = float(np.mean(fold_scores))
        sem = float(np.std(fold_scores, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        half = _T975.get(n - 1, 1.96) * sem if n > 1 else float("nan")
        per_layer[layer] = {
            "raw_center": center,
            "raw_ci_low": center - half,
            "raw_ci_high": center + half,
            "ceiling_normalized_center": normalized(center),
            "ceiling_normalized_ci_low": normalized(center - half),
            "ceiling_normalized_ci_high": normalized(center + half),
            "n_folds": n,
            # every model is scored on the same folds, so keeping them allows paired comparisons
            "fold_scores": [float(x) for x in fold_scores]
        }
        print(f"{label} {layer}: raw={center:.4f} ±{half:.4f} (n={n})", flush=True)
    return per_layer


def main() -> None:
    """Scores every layer of every model and writes the intervals."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", default="Pereira2018.243sentences-ridge")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="repeated 'model_id=revision' entries"
    )
    args = parser.parse_args()

    from brainscore_language import load_benchmark

    import scripts.export_pereira_parity as ex

    benchmark = load_benchmark(args.benchmark)
    ceiling = benchmark_ceiling(benchmark)
    out: dict = {
        "benchmark": args.benchmark,
        "ceiling": ceiling,
        "models": {}
    }
    for entry in args.models:
        model_id, revision = entry.split("=", 1)
        model, tokenizer = ex._load_model_and_tokenizer(model_id, revision)
        out["models"][model_id] = score_layers(
            benchmark,
            model,
            tokenizer,
            model_id,
            args.layers,
            ceiling
        )

    with open(args.output, "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
