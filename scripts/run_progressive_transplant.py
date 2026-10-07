"""Scores progressive transplants of the TinyStories blocks, with and without its embeddings, into each code model.

For every receiver and k it builds the transplant in memory (nothing is written to disk) and scores all
twelve blocks on the benchmark's own folds. Both ends of each curve are checked as parameter-hash
equalities: k = 0 without embeddings is the receiver and k = 12 with embeddings is the donor. Writes one
JSON to --output and exits with status 2 when an endpoint check fails.

Example: python -m scripts.run_progressive_transplant --output layer_cis_progressive.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from brain_code.experiments.progressive_transplant import state_dict_hash, transplant
from brain_code.reproduction.config import HUB_REVISIONS
from scripts.extract_layer_cis import benchmark_ceiling, score_layers

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="Pereira2018.243sentences-ridge")
    p.add_argument("--donor", default="tinystories")
    p.add_argument(
        "--receivers",
        nargs="+",
        default=["python", "source", "binary", "assembly"]
    )
    p.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[0, 1, 2, 4, 6, 8, 12]
    )
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def load(corpus: str):
    """The pinned seed-42 Hub model and tokenizer of one corpus."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hub_id = f"IParraMartin/gpt2-{corpus}"
    revision = HUB_REVISIONS[corpus]
    model = AutoModelForCausalLM.from_pretrained(hub_id, revision=revision).eval()
    tokenizer = AutoTokenizer.from_pretrained(hub_id, revision=revision, truncation_side="left")
    return hub_id, model, tokenizer


def main() -> None:
    args = parse_args()

    import torch
    from brainscore_language import load_benchmark

    benchmark = load_benchmark(args.benchmark)
    ceiling = benchmark_ceiling(benchmark)
    donor_id, donor, donor_tokenizer = load(args.donor)
    donor_hash = state_dict_hash(donor)
    n_layer = len(donor.transformer.h)
    out: dict = {
        "benchmark": args.benchmark,
        "ceiling": ceiling,
        "source": "run_progressive_transplant.py (progressive block transplant, in-memory)",
        "donor": donor_id,
        "donor_parameter_hash": donor_hash,
        "k_values": args.k_values,
        "design": (
            "arm '<receiver>|k<k>|<blocks|blocksemb>' = donor's transformer.h[0:k] "
            "copied into the receiver; 'blocksemb' also copies wte/wpe and re-ties "
            "the head; ln_f is copied only at k=n_layer so the k=12 arm is exactly "
            "the donor."
        ),
        "endpoint_checks": {},
        "models": {}
    }

    for corpus in args.receivers:
        receiver_id, receiver, receiver_tokenizer = load(corpus)
        receiver_hash = state_dict_hash(receiver)
        # an arm with transplanted embeddings is read with the donor's tokenizer, so the design needs a
        # shared vocabulary
        if receiver_tokenizer.get_vocab() != donor_tokenizer.get_vocab():
            raise SystemExit(f"{receiver_id} and the donor do not share a vocabulary")
        for with_embeddings in (False, True):
            family = "blocksemb" if with_embeddings else "blocks"
            for k in args.k_values:
                label = f"{corpus}|k{k}|{family}"
                patched = transplant(
                    receiver,
                    donor,
                    k,
                    with_embeddings
                )
                patched_hash = state_dict_hash(patched)
                if k == 0 and not with_embeddings:
                    out["endpoint_checks"][f"{corpus}|k0|blocks==receiver"] = patched_hash == receiver_hash
                if k == n_layer and with_embeddings:
                    out["endpoint_checks"][f"{corpus}|k{n_layer}|blocksemb==donor"] = patched_hash == donor_hash
                tokenizer = donor_tokenizer if with_embeddings else receiver_tokenizer
                out["models"][label] = {
                    "receiver": receiver_id,
                    "k": k,
                    "with_embeddings": with_embeddings,
                    "parameter_hash": patched_hash,
                    "layers": score_layers(
                        benchmark,
                        patched,
                        tokenizer,
                        label,
                        args.layers,
                        ceiling
                    )
                }
                del patched
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        del receiver

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"WROTE {args.output}", flush=True)
    failed = [name for name, ok in out["endpoint_checks"].items() if not ok]
    if failed:
        print(f"endpoint identity failed for: {', '.join(failed)}", flush=True)
        raise SystemExit(2)
    print(f"all {len(out['endpoint_checks'])} endpoint identities hold", flush=True)


if __name__ == "__main__":
    main()
