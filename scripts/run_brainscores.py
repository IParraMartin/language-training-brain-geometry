"""Scores one model on an official Brain-Score benchmark without test-set layer selection.

Either sweeps every layer descriptively or scores one prespecified layer, and writes the result
JSON under --results_dir.

Example: brain-score --hf_model_id IParraMartin/gpt2-python --model_revision <sha> --experiment pereira_243
"""

from __future__ import annotations

import argparse

from brain_code.config import EXPERIMENTS
from brain_code.experiments.brain_score_harness import run_experiment


def main() -> None:
    """Parses the arguments and runs the benchmark."""
    parser = argparse.ArgumentParser(
        description="Run official Brain-Score without test-set peak-layer selection."
    )
    parser.add_argument(
        "--hf_model_id",
        required=True,
        help=(
            "One model identifier per invocation. The immutable model/tokenizer "
            "revision applies to this model only."
        )
    )
    parser.add_argument("--model_revision", required=True, help="Immutable HF commit SHA.")
    parser.add_argument("--tokenizer_revision")
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument(
        "--layer_selection_mode",
        choices=["descriptive_grid", "prespecified"],
        default="descriptive_grid"
    )
    parser.add_argument("--prespecified_layer")
    parser.add_argument("--is_test", action="store_true")
    parser.add_argument("--smoke_layer", default="last")
    parser.add_argument("--save_results", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--results_dir", default="results/official_brainscore")
    parser.add_argument("--selection_evidence_path")
    parser.add_argument("--unit_test_evidence_path")
    args = parser.parse_args()
    if args.layer_selection_mode == "prespecified" and not (
        args.prespecified_layer or args.is_test
    ):
        parser.error("--prespecified_layer is required in prespecified mode.")
    benchmark = EXPERIMENTS[args.experiment]
    result = run_experiment(
        args.hf_model_id,
        benchmark,
        is_test=args.is_test,
        save_results=args.save_results,
        smoke_layer=args.smoke_layer,
        layer_selection_mode=args.layer_selection_mode,
        prespecified_layer=args.prespecified_layer,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        results_dir=args.results_dir,
        selection_evidence_path=args.selection_evidence_path,
        unit_test_evidence_path=args.unit_test_evidence_path
    )
    if args.is_test:
        print(
            f"{args.hf_model_id}: "
            f"official_brainscore_raw_pearson={result:.6f}"
        )


if __name__ == "__main__":
    main()
