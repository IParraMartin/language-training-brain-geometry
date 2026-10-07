"""Compares a custom and an official Pereira parity bundle.

Writes the comparison report to --output and exits with status 2 when the bundles disagree beyond
tolerance.

Example: python scripts/run_pereira_parity.py --official-bundle a.npz --custom-bundle b.npz --output report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from brain_code.experiments.pereira_parity import (
    ParityTolerance,
    compare_pereira_parity,
    load_parity_bundle
)


def main() -> None:
    """Loads both bundles, compares them and writes the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-bundle",
        "--official_bundle",
        dest="official_bundle",
        required=True
    )
    parser.add_argument(
        "--custom-bundle",
        "--custom_bundle",
        dest="custom_bundle",
        required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--activation-atol",
        "--activation_atol",
        dest="activation_atol",
        type=float,
        default=1e-6
    )
    parser.add_argument(
        "--activation-rtol",
        "--activation_rtol",
        dest="activation_rtol",
        type=float,
        default=1e-5
    )
    parser.add_argument("--target-atol", type=float, default=0.0)
    parser.add_argument("--target-rtol", type=float, default=0.0)
    parser.add_argument(
        "--prediction-atol",
        "--prediction_atol",
        dest="prediction_atol",
        type=float,
        default=1e-7
    )
    parser.add_argument(
        "--prediction-rtol",
        "--prediction_rtol",
        dest="prediction_rtol",
        type=float,
        default=1e-6
    )
    parser.add_argument(
        "--score-atol",
        "--score_atol",
        dest="score_atol",
        type=float,
        default=1e-8
    )
    parser.add_argument(
        "--score-rtol",
        "--score_rtol",
        dest="score_rtol",
        type=float,
        default=1e-7
    )
    args = parser.parse_args()

    tolerance = ParityTolerance(
        activation_atol=args.activation_atol,
        activation_rtol=args.activation_rtol,
        target_atol=args.target_atol,
        target_rtol=args.target_rtol,
        prediction_atol=args.prediction_atol,
        prediction_rtol=args.prediction_rtol,
        score_atol=args.score_atol,
        score_rtol=args.score_rtol
    )
    report = compare_pereira_parity(
        load_parity_bundle(args.official_bundle),
        load_parity_bundle(args.custom_bundle),
        tolerance=tolerance
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": report["validated_against_official"]}))
    if not report["validated_against_official"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
