"""Builds the random-weight reference models, identical to a training run's initialization.

Constructs LMTrainer exactly as brain-train does and saves its model before any update, so the seed
reaches initialization the same way and the tokenizer's special-token override (bos = eos = pad = 0)
is applied. Writes one directory per seed under --out-root and a manifest with parameter hashes; nothing
is trained and no data is read. Score the directories with scripts/score_local_checkpoints.py.

Example: python -m scripts.make_random_init_models --out-root checkpoints --manifest checkpoints/random_init.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from brain_code.config import PATHS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44]
    )
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    return p.parse_args()


def parameter_hash(model) -> str:
    """sha256 over the sorted state dict's names and raw bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()

    from brain_code.training.train_model import LMTrainer

    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "model_config": "configs/models/gpt2_small.yaml",
        "tokenizer": "data/tokenizers/50k_50mb",
        "models": {}
    }
    for seed in args.seeds:
        dest = args.out_root / f"random_init_seed{seed}"
        # gradient checkpointing stays off: it changes the module for backprop and this model only runs forward
        trainer = LMTrainer(
            tokenizer=PATHS["tokenizer"]["default"],
            dataset=None,
            training_args={
                "seed": seed,
                "gradient_checkpointing": False,
                "output_dir": str(dest)
            },
            model_args=PATHS["model_configs"]["small"],
            dataset_name="random_init"
        )
        model, tokenizer = trainer.model, trainer.tokenizer
        dest.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(dest), safe_serialization=True)
        tokenizer.save_pretrained(str(dest))
        digest = parameter_hash(model)
        manifest["models"][f"random-init-seed{seed}"] = {
            "path": str(dest),
            "seed": seed,
            "n_parameters": sum(p.numel() for p in model.parameters()),
            "parameter_hash_sha256": digest
        }
        print(f"  seed {seed}: hash {digest[:16]} -> {dest}", flush=True)

    # identical weights for different seeds would mean the seed never reached initialization
    hashes = {m["parameter_hash_sha256"] for m in manifest["models"].values()}
    manifest["distinct_initialisations"] = len(hashes)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if len(hashes) != len(manifest["models"]):
        raise SystemExit(f"only {len(hashes)} distinct initialisations for {len(manifest['models'])} seeds")
    print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()
