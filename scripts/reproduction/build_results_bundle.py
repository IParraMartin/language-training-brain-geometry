"""Builds the results bundle: tidy CSV and JSON tables extracted from the raw result archives.

Reads the exported results tree, the analysis-campaign artifacts, the training provenance records and
the recovered metadata, and writes every table the statistics need, plus MANIFEST.json with
the sha256 of each input and output. Raw archives are only read, never modified; an existing bundle is
overwritten.

Example: python scripts/reproduction/build_results_bundle.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

from brain_code.config import ROOT
from brain_code.reproduction.config import (
    BENCHMARKS,
    CODE_CORPORA,
    CORPORA,
    LAYERS,
    load_paths,
    require
)

_INPUTS: dict[str, str] = {}
PRIMARY_MODEL = re.compile(r"^(tinystories|python|source|assembly|binary|random_init)_seed\d+$")
# named roots, so the manifest records archive-relative paths rather than machine paths
_ROOTS: dict[str, Path] = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="alternative paths yaml")
    return p.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_input(path: Path) -> None:
    """Records an input's checksum under a root-relative name such as ``results_archive:blimp/x.txt``."""
    resolved = path.resolve()
    for name, root in _ROOTS.items():
        try:
            _INPUTS[f"{name}:{resolved.relative_to(root.resolve())}"] = sha256(path)
            return
        except ValueError:
            continue
    raise ValueError(f"input {path} is outside every configured root")


def read_json(path: Path, label: str) -> dict:
    """Reads one input, recording its checksum for the manifest."""
    require(path, label)
    record_input(path)
    return json.loads(path.read_text())


def read_jsonl(path: Path, label: str) -> list[dict]:
    require(path, label)
    record_input(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    """Writes rows with a header taken from the first row; refuses ragged rows."""
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    fields = list(rows[0])
    for row in rows:
        if list(row) != fields:
            raise ValueError(f"inconsistent columns in {path.name}: {list(row)} vs {fields}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def model_identity(key: str) -> tuple[str, int]:
    """Maps an archive model key to ``(corpus, seed)``; random-weight models map to corpus 'random'."""
    if key.startswith("IParraMartin/gpt2-"):
        return key.split("gpt2-", 1)[1], 42
    if key.startswith("random-init-seed") or key.startswith("random_init_seed"):
        return "random", int(re.findall(r"\d+$", key)[0])
    corpus, seed = key.rsplit("_seed", 1)
    return corpus, int(seed)


def layer_scores(archive: Path) -> list[dict]:
    """Per-fold re-extracted scores (neuroid mean per fold, mean over folds) for every model and block."""
    rows = []
    for bench, spec in BENCHMARKS.items():
        seen = set()
        for rel in spec["files"]:
            payload = read_json(archive / rel, f"{bench} layer scores")
            for key, layers in payload["models"].items():
                corpus, seed = model_identity(key)
                if (corpus, seed) in seen:
                    raise ValueError(f"{bench}: duplicate model {corpus} seed {seed} in {rel}")
                seen.add((corpus, seed))
                for block, layer in enumerate(LAYERS):
                    cell = layers[layer]
                    rows.append({
                        "benchmark": bench,
                        "corpus": corpus,
                        "seed": seed,
                        "run_id": key,
                        "block": block,
                        "raw_r": cell["raw_center"],
                        "n_folds": cell["n_folds"],
                        "fold_scores": json.dumps(cell["fold_scores"]),
                        "source": rel
                    })
        expected = {(c, s) for c in CORPORA for s in (
            42,
            43,
            44,
            45,
            46
        )} | {("random", s) for s in (42, 43, 44)}
        if seen != expected:
            raise ValueError(f"{bench}: models {sorted(seen ^ expected)} missing or unexpected")
    return rows


def official_seed42(archive: Path) -> list[dict]:
    """Benchmark-native scores (median over neuroids) of the five published seed-42 models."""
    rows = []
    for bench, folder in (("pereira_243", "official_brainscore"), ("pereira_384", "official_brainscore_384")):
        for path in sorted((archive / folder).glob("brain-code-*.json")):
            payload = read_json(path, "official score")
            corpus, _ = model_identity(payload["model_identifier"])
            for block, layer in enumerate(LAYERS):
                cell = payload["raw_layer_results"][layer]
                rows.append({
                    "benchmark": bench,
                    "corpus": corpus,
                    "block": block,
                    "raw_r": cell["official_brainscore_raw_pearson"],
                    "ceiling": cell["official_benchmark_ceiling"],
                    "model_revision": payload["model_revision"],
                    "run_id": payload["run_id"]
                })
    return rows


def geometry(campaign: Path) -> tuple[list[dict], list[dict]]:
    """Multi-seed spectrum summaries, brain RSA and permutation-null results, and pairwise CKA."""
    per_model, cka_rows = [], []
    for bench in (
        "pereira_243",
        "pereira_384",
        "tuckute",
        "blank",
        "fedorenko"
    ):
        payload = read_json(campaign / "experiments" / "19" / f"geometry_{bench}.json", f"{bench} geometry")
        for model in payload["models"]:
            if not PRIMARY_MODEL.match(model):
                continue
            corpus, seed = model_identity(model)
            for block, layer in enumerate(LAYERS):
                g = payload["geometry"][model][layer]
                null = payload.get("rsa_permutation_null", {}).get(model, {}).get(layer, {})
                per_model.append({
                    "benchmark": bench,
                    "corpus": corpus,
                    "seed": seed,
                    "block": block,
                    "participation_ratio": g["participation_ratio"],
                    "effective_rank": g["effective_rank"],
                    "anisotropy": g["anisotropy"],
                    "spectral_decay": g["spectral_decay"],
                    "rsa_pooled": payload["brain_rsa"][model][layer]["pooled"],
                    "rsa_null_z": null.get("z_vs_null"),
                    "rsa_null_p": null.get("p_value"),
                    "brain_rdm_metric": payload["brain_rdm_metric"],
                    "representation": payload["representation"]
                })
        if bench == "pereira_243":
            for block, layer in enumerate(LAYERS):
                for pair, value in payload["cka"][layer].items():
                    a, b = pair.split("|")
                    cka_rows.append({
                        "benchmark": bench,
                        "block": block,
                        "model_a": a,
                        "model_b": b,
                        "cka": value
                    })
    return per_model, cka_rows


def pc_ablation(campaign: Path) -> list[dict]:
    """RSA after removing leading principal directions (column-centered representations)."""
    rows = []
    for bench in ("pereira_243", "pereira_384", "fedorenko"):
        payload = read_json(campaign / "experiments" / "19" / f"pc_ablation_rsa_{bench}.json", "PC ablation")
        for model, layers in payload["per_model"].items():
            if not PRIMARY_MODEL.match(model):
                continue
            corpus, seed = model_identity(model)
            for block, layer in enumerate(LAYERS):
                for m, value in layers[layer].items():
                    rows.append({
                        "benchmark": bench,
                        "corpus": corpus,
                        "seed": seed,
                        "block": block,
                        "removed_pcs": int(m),
                        "rsa": value
                    })
    return rows


def progressive_transplant(archive: Path) -> tuple[list[dict], dict]:
    """Per-fold scores of every progressive-transplant arm, plus the design and endpoint checks."""
    payload = read_json(archive / "official_brainscore" / "layer_cis_e2b_progressive.json", "progressive transplant")
    rows = []
    for arm_key, arm in payload["models"].items():
        receiver, k, arm_name = arm_key.split("|")
        for block, layer in enumerate(LAYERS):
            cell = arm["layers"][layer]
            rows.append({
                "receiver": receiver,
                "k": int(k[1:]),
                "arm": "blocks_and_embeddings" if arm_name == "blocksemb" else "blocks_only",
                "block": block,
                "raw_r": cell["raw_center"],
                "fold_scores": json.dumps(cell["fold_scores"]),
                "parameter_hash": arm["parameter_hash"]
            })
    meta = {key: payload[key] for key in (
        "donor",
        "donor_parameter_hash",
        "k_values",
        "design",
        "endpoint_checks"
    )}
    return rows, meta


def embedding_adaptation(archive: Path) -> list[dict]:
    """Receiver, English-adapted and own-corpus-adapted scores on both scoring paths."""
    rows = []
    folds = read_json(archive / "official_brainscore" / "layer_cis_embedft.json", "embedding FT folds")["models"]
    base_folds = read_json(archive / "official_brainscore" / "layer_cis.json", "seed-42 folds")["models"]
    official = {}
    for folder, arm in (("experiments/09", "english"), ("experiments/13", "own_corpus")):
        for path in sorted((archive / folder).glob("brain-code-*.json")):
            payload = read_json(path, "embedding FT official")
            receiver = re.search(r"gpt2-(python|source|assembly|binary)", payload["model_identifier"]).group(1)
            official[(receiver, arm)] = payload["raw_layer_results"]
    for path in sorted((archive / "official_brainscore").glob("brain-code-*.json")):
        payload = read_json(path, "official score")
        corpus, _ = model_identity(payload["model_identifier"])
        official[(corpus, "receiver")] = payload["raw_layer_results"]
    for receiver in CODE_CORPORA:
        fold_sources = {
            "receiver": base_folds[f"IParraMartin/gpt2-{receiver}"],
            "english": folds[f"exp09_gpt2-{receiver}_embed_ft"],
            "own_corpus": folds[f"exp13_gpt2-{receiver}_embed_ft_own"]
        }
        for arm, layers in fold_sources.items():
            for block, layer in enumerate(LAYERS):
                rows.append({
                    "receiver": receiver,
                    "arm": arm,
                    "block": block,
                    "raw_r_fold_mean": layers[layer]["raw_center"],
                    "fold_scores": json.dumps(layers[layer]["fold_scores"]),
                    "raw_r_official": official[(receiver, arm)][layer]["official_brainscore_raw_pearson"]
                })
    return rows


def single_block_transplant(archive: Path) -> list[dict]:
    """Fold-paired tests of each single-block (or embedding) transplant against its receiver."""
    payload = read_json(archive / "official_brainscore" / "exp11_paired_tests.json", "single-block transplant tests")
    rows = []
    names = {
        "Assembly": "assembly",
        "Binary": "binary",
        "Python": "python",
        "C": "source"
    }
    for receiver, layers in payload["receivers"].items():
        for layer, cells in layers.items():
            for cell in cells:
                position = cell["patch_position"]
                rows.append({
                    "receiver": names[receiver],
                    "readout_block": int(layer.rsplit(".", 1)[1]),
                    "patch": "embeddings" if position == "embeddings" else f"block_{int(position)}",
                    "mean_difference": cell["mean_difference"],
                    "cohens_d_paired": cell.get("cohens_d_paired"),
                    "p_uncorrected": cell.get("p_uncorrected"),
                    "p_holm": cell.get("p_holm"),
                    "significant": cell.get("significant"),
                    "exceeds_band": cell.get("exceeds_band")
                })
    return rows


def factorial(archive: Path, campaign: Path) -> list[dict]:
    """Milestone scores of the pseudoword, 2x2 factorial and micro-scale diagnostic runs."""
    sources = [
        ("pseudoword_gpt2_small", archive / "experiments" / "17"),
        ("factorial_gpt2_small", campaign / "savio" / "experiments" / "19"),
        ("factorial_micro_diagnostic", archive / "experiments" / "06")
    ]
    rows = []
    for study, folder in sources:
        for run in sorted(require(folder, study).glob("run_cond*_seed*")):
            condition, seed = re.match(r"run_cond(\w+?)_seed(\d+)", run.name).groups()
            provenance = read_json(run / "TRAINING_PROVENANCE.json", f"{study} provenance")
            for record in read_jsonl(run / "exact_checkpoint_metrics.jsonl", f"{study} milestones"):
                for layer, cell in record["neural_alignment"]["layers"].items():
                    rows.append({
                        "study": study,
                        "condition": condition,
                        "seed": int(seed),
                        "milestone_tokens": record["milestone_tokens"],
                        "actual_nonpadding_tokens": record["actual_nonpadding_tokens"],
                        "global_step": record["global_step"],
                        "eval_loss": record["eval_loss"],
                        "block": int(layer.rsplit(".", 1)[1]),
                        "raw_r": cell["official_brainscore_raw_pearson"],
                        "n_layer": provenance["model_args"]["n_layer"],
                        "n_embd": provenance["model_args"]["n_embd"]
                    })
    return rows


def corpus_diagnostics(archive: Path) -> list[dict]:
    """Corpus size, stimulus coverage and frequency, tokenizer fertility and BLiMP for each corpus."""
    counts = read_json(archive / "confounds" / "token_counts.json", "token counts")
    coverage = read_json(archive / "confounds" / "coverage.json", "coverage")
    fertility = read_json(archive / "confounds" / "fertility_results.json", "fertility")
    frequency = read_json(archive / "confounds" / "correlational_analysis.json", "frequency")["across_models"]
    mean_frequency = dict(zip(frequency["models"], frequency["mean_stimulus_freq"]))
    rows = []
    for corpus in CORPORA:
        blimp_path = archive / "blimp" / f"{corpus}.txt"
        require(blimp_path, "BLiMP output")
        record_input(blimp_path)
        line = next(x for x in blimp_path.read_text().splitlines() if x.startswith("|blimp "))
        cells = [x.strip() for x in line.split("|")]
        rows.append({
            "corpus": corpus,
            "full_split_tokens": counts[corpus],
            "stimulus_type_coverage": coverage[corpus]["type_cov_ge1"],
            "stimulus_token_coverage": coverage[corpus]["tok_cov_ge1"],
            "mean_corpus_count_per_stimulus_token": mean_frequency[corpus],
            "tokens_per_byte": fertility[corpus]["tokens_per_byte"],
            "blimp_accuracy": float(cells[7]),
            "blimp_stderr": float(cells[9])
        })
    return rows


def single_checkpoint_spectra(root: Path) -> list[dict]:
    """The one-checkpoint, isolated-sentence spectrum audit (custom grouped-CV encoder)."""
    import numpy as np

    rows = []
    for corpus in CORPORA:
        path = root / "results" / "spectrum" / f"uncertainty_{corpus}.npz"
        spectra_path = root / "results" / "spectrum" / f"spectra_{corpus}.npz"
        for item in (path, spectra_path):
            require(item, "single-checkpoint spectrum data")
            record_input(item)
        z = np.load(path)
        spectra = np.load(spectra_path)["spectra"]
        name = f"IParraMartin-gpt2-{corpus}_Pereira2018.243sentences-ridge_manifold.json"
        run = root / "results" / "spectrum" / name
        record = read_json(run, "single-checkpoint spectrum result")
        by_block = {g["layer"]: g for g in record["layer_geometry"]}
        coupling = record["geometry_alignment_coupling"]
        for block in range(z["fold_r"].shape[0]):
            eigen = np.clip(spectra[block], 0, None)
            cumulative = np.cumsum(eigen) / eigen.sum()
            rows.append({
                "corpus": corpus,
                "block": block,
                "fold_r": json.dumps(z["fold_r"][block].tolist()),
                "participation_ratio": float(z["full_participation_ratio"][block]),
                "effective_rank": float(z["full_effective_rank"][block]),
                "spectral_decay": float(z["full_spectral_decay"][block]),
                "jackknife_se_spectral_decay": float(z["se_spectral_decay"][block]),
                "dims_for_90pct_variance": int(np.searchsorted(cumulative, 0.9) + 1),
                "anisotropy": by_block[block]["anisotropy"],
                "alignment_r": by_block[block]["alignment_r"],
                "coupling_rho": json.dumps({k: v["rho"] for k, v in coupling.items()}),
                "context_mode": record["context_mode"],
                "outer_cv": record["outer_cv_definition"],
                "cumulative_variance": json.dumps(cumulative.tolist())
            })
    return rows


def training_provenance(archive: Path, seed_models: Path) -> list[dict]:
    """Recorded training configuration of every primary, pseudoword and factorial run with a provenance file."""
    rows = []
    for path in sorted(require(seed_models, "seed checkpoints").glob("*_seed*/TRAINING_PROVENANCE.json")):
        payload = read_json(path, "training provenance")
        corpus, seed = model_identity(path.parent.name)
        args = payload["training_args"]
        best = payload.get("best_model_checkpoint")
        rows.append({
            "run_id": path.parent.name,
            "corpus": corpus,
            "seed": seed,
            "status": payload["status"],
            "global_step": payload["global_step"],
            "max_steps": args["max_steps"],
            "best_checkpoint_step": int(best.rsplit("-", 1)[1]) if best else None,
            "load_best_model_at_end": args["load_best_model_at_end"],
            "eval_steps": args["eval_steps"],
            "per_device_train_batch_size": args["per_device_train_batch_size"],
            "gradient_accumulation_steps": args["gradient_accumulation_steps"],
            "learning_rate": args["learning_rate"],
            "warmup_steps": args["warmup_steps"],
            "bf16": args["bf16"],
            "split_before_pack": payload["split_before_pack"],
            "insert_eos_between_docs": payload["insert_eos_between_docs"],
            "initial_parameter_hash": payload["initial_parameter_hash"],
            "final_parameter_hash": payload["final_parameter_hash"],
            "tokenizer_hash": payload["tokenizer_hash"]
        })
    return rows


def main() -> None:
    args = parse_args()
    paths = load_paths(args.config)
    archive = require(paths.results_archive, "results archive")
    campaign = require(paths.campaign_artifacts, "campaign artifacts")
    _ROOTS.update({
        "results_archive": archive,
        "campaign_artifacts": campaign,
        "seed_models": paths.seed_models,
        "recovered": paths.recovered,
        "repository": ROOT
    })
    out = paths.results_bundle
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    tables = {
        "layer_scores.csv": layer_scores(archive),
        "official_seed42.csv": official_seed42(archive),
        "pc_ablation_rsa.csv": pc_ablation(campaign),
        "embedding_adaptation.csv": embedding_adaptation(archive),
        "single_block_transplant.csv": single_block_transplant(archive),
        "factorial_milestones.csv": factorial(archive, campaign),
        "corpus_diagnostics.csv": corpus_diagnostics(archive),
        "spectrum_single_checkpoint.csv": single_checkpoint_spectra(ROOT),
        "training_provenance.csv": training_provenance(archive, paths.seed_models)
    }
    tables["geometry_multiseed.csv"], tables["cka_pairs.csv"] = geometry(campaign)
    tables["progressive_transplant.csv"], transplant_meta = progressive_transplant(archive)
    for name, rows in tables.items():
        write_csv(out / name, rows)
    (out / "progressive_transplant_design.json").write_text(json.dumps(transplant_meta, indent=2) + "\n")
    for name in ("packed_splits.json", "benchmarks.json", "corpus_audit.json"):
        source = require(paths.recovered / name, f"recovered {name}")
        record_input(source)
        shutil.copy2(source, out / name)

    manifest = {
        "inputs": dict(sorted(_INPUTS.items())),
        "outputs": {p.name: sha256(p) for p in sorted(out.iterdir())}
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(manifest['outputs'])} files to {out} from {len(manifest['inputs'])} inputs")


if __name__ == "__main__":
    main()
