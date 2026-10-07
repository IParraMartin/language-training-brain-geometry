"""Computes per-layer raw Brain-Score with per-fold scores for local checkpoint directories.

Scores seeds 43-46, the random-weight models and adapted or transplanted models through the same official
loop, metric and fold intervals as scripts/extract_layer_cis.py. Writes after every model; a rerun with
the same --output skips the models already scored.

Example: python -m scripts.score_local_checkpoints --benchmark Blank2014-ridge --output s.json --checkpoints a=ckpt/a
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from brain_code.config import PATHS
from scripts.extract_layer_cis import benchmark_ceiling, score_layers

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--tokenizer-fallback",
        default=PATHS["tokenizer"]["default"],
        help="tokenizer used when a checkpoint directory carries none"
    )
    p.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="repeated 'label=path/to/checkpoint_dir' entries"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from brainscore_language import load_benchmark
    from transformers import AutoModelForCausalLM, AutoTokenizer

    benchmark = load_benchmark(args.benchmark)
    ceiling = benchmark_ceiling(benchmark)
    if ceiling is None:
        print(f"  NOTE: {args.benchmark} has no ceiling; ceiling_normalized_* will be null", flush=True)
    out: dict = {
        "benchmark": args.benchmark,
        "ceiling": ceiling,
        "source": "score_local_checkpoints.py (local checkpoints)",
        "models": {}
    }
    if args.output.is_file():
        try:
            prior = json.loads(args.output.read_text())
        except ValueError:
            prior = {}
        if prior.get("benchmark") == args.benchmark:
            out["models"] = prior.get("models", {})
            print(f"  resuming: {len(out['models'])} models already scored", flush=True)

    for entry in args.checkpoints:
        label, path = entry.split("=", 1)
        if label in out["models"]:
            print(f"  skip {label} (already scored)", flush=True)
            continue
        checkpoint = Path(path)
        if not (checkpoint / "model.safetensors").is_file():
            raise SystemExit(f"no model.safetensors in {checkpoint}")
        tokenizer_dir = checkpoint if (checkpoint / "tokenizer.json").is_file() else Path(args.tokenizer_fallback)
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), truncation_side="left")
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint), use_safetensors=True).eval()
        per_layer = score_layers(
            benchmark,
            model,
            tokenizer,
            label,
            args.layers,
            ceiling
        )
        for record in per_layer.values():
            record["checkpoint_path"] = str(checkpoint)
        out["models"][label] = per_layer
        # written after every model, so a walltime kill costs one model rather than all of them
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + "\n")
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWROTE {args.output} ({len(out['models'])} models)", flush=True)


if __name__ == "__main__":
    main()
