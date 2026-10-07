"""Measures how the training corpus shapes representational geometry, on already-trained checkpoints.

For every model and transformer block it extracts the last-token state of each stimulus's cumulative
within-group context, exactly as the benchmark presents it, and reports the covariance-spectrum profile,
linear CKA between every pair of models and RSA against the brain (pooled and per participant, with an
optional within-group permutation null). Writes one JSON to --out.

Example: python scripts/run_latent_geometry.py --models-dir <seed_models> --benchmark pereira_243 --out geometry.json
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import squareform

from brain_code.experiments.manifold_pressure import spectral_profile
from brain_code.experiments.representational_similarity import (
    grouped_permutation_null,
    linear_cka,
    permutation_p_value,
    rdm,
    rsa_from_rdms,
    square_rdm
)

# Brain-Score's HuggingfaceSubject represents a stimulus by the hidden state at its last token
# (values[:, -1:, :]), and each benchmark feeds stimulus k of a group as prepare_context(text[:k + 1]).
# Isolated sentences would describe a different object from the one the encoder scores, so this script
# reuses prepare_context instead of reimplementing the join. Block N is hidden_states[N + 1];
# hidden_states[0] is the embedding output, which is not scored.
DEFAULT_COLLAPSE_TOKENIZER = "IParraMartin/gpt2-tinystories"

# identifier and the coordinate the benchmark groups stimuli by when it feeds the model; every benchmark
# sorts the groups and accumulates context within a group, so only the coordinate differs
BENCHMARKS = {
    "pereira_243": {
        "id": "Pereira2018.243sentences-ridge",
        "group": "passage_label"
    },
    "pereira_384": {
        "id": "Pereira2018.384sentences-ridge",
        "group": "passage_label"
    },
    "blank": {
        "id": "Blank2014-ridge",
        "group": "story"
    },
    # word-level ECoG: context accumulates word by word within a sentence
    "fedorenko": {
        "id": "Fedorenko2016-ridge",
        "group": "sentence_id"
    },
    # one stimulus per group, so context is the stimulus alone and no within-group permutation exists
    "tuckute": {
        "id": "Tuckute2024-ridge",
        "group": "stimulus_id"
    }
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--models-dir",
        type=Path,
        required=True,
        help="Directory of local checkpoints, one subdirectory per model."
    )
    p.add_argument(
        "--hf-model",
        action="append",
        default=[],
        metavar="NAME=HUB_ID",
        help="Additional Hub checkpoint, repeatable."
    )
    p.add_argument("--benchmark", default="pereira_243", choices=sorted(BENCHMARKS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument(
        "--limit-models",
        type=int,
        default=None,
        help="Use only the first N models; for smoke runs."
    )
    p.add_argument(
        "--permutations",
        type=int,
        default=0,
        help="Within-passage permutation draws for the RSA null. 0 skips the null entirely."
    )
    return p.parse_args(argv)


def load_stimuli(benchmark_key: str):
    """Return (sentences, brain matrix, per-presentation frame, per-neuroid frame)."""
    from brainscore_language import load_benchmark

    bench = load_benchmark(BENCHMARKS[benchmark_key]["id"])
    data = bench.data
    presentation = data.indexes["presentation"].to_frame(index=False)
    neuroid = data.indexes["neuroid"].to_frame(index=False)
    # "stimulus" is the only text coordinate every benchmark carries. Pereira also
    # has "sentence", Blank has "stimulus_sentence", Fedorenko has "word" -- and
    # reading any of those KeyErrors on the others. It is also the coordinate the
    # benchmarks themselves feed to the model (self.data['stimulus']).
    sentences = presentation["stimulus"].astype(str).tolist()
    brain = np.asarray(data.values, dtype=np.float64)
    if brain.shape[0] != len(sentences):
        raise SystemExit(f"benchmark shape {brain.shape} does not match {len(sentences)} stimuli")
    return (
        sentences,
        brain,
        presentation,
        neuroid
    )


def discover_models(models_dir: Path, hf_models: list[str], limit: int | None):
    found: list[tuple[str, str]] = []
    for child in sorted(models_dir.iterdir()) if models_dir.is_dir() else []:
        if (child / "config.json").is_file():
            found.append((child.name, str(child)))
    for spec in hf_models:
        if "=" not in spec:
            raise SystemExit(f"--hf-model expects NAME=HUB_ID, got {spec!r}")
        name, hub_id = spec.split("=", 1)
        found.append((name, hub_id))
    if not found:
        raise SystemExit(f"no checkpoints found under {models_dir}")
    return found[:limit] if limit else found


def build_passage_contexts(presentation, group_coord: str = "passage_label") -> list[str]:
    """Cumulative within-group context per stimulus, mirroring the benchmark.

    Every benchmark here uses the same shape in ``__call__``: groups taken in
    sorted order, stimuli within a group in assembly order, and stimulus ``k``
    represented by ``prepare_context`` over that group's stimuli ``[0..k]``.
    Only the grouping coordinate differs — ``passage_label`` for Pereira,
    ``story`` for Blank, ``sentence_id`` for Fedorenko (word-level, so context
    accumulates word by word), ``stimulus_id`` for Tuckute (singleton groups, so
    the context is the stimulus alone).

    Args:
        presentation: Frame of the benchmark's presentation MultiIndex; needs
            ``stimulus`` and ``group_coord``.
        group_coord: The coordinate the benchmark groups by.

    Returns:
        list[str]: One context string per stimulus, in assembly order.

    Raises:
        SystemExit: If the grouping coordinate is absent.
    """
    from brainscore_language.model_helpers.preprocessing import prepare_context

    if group_coord not in presentation.columns:
        raise SystemExit(
            f"grouping coordinate {group_coord!r} not in presentation ({list(presentation.columns)})"
        )
    stimuli = presentation["stimulus"].astype(str).tolist()
    passages = presentation[group_coord].astype(str).tolist()
    contexts: list[str | None] = [None] * len(stimuli)
    for passage in sorted(set(passages)):
        positions = [i for i, label in enumerate(passages) if label == passage]
        parts = [stimuli[i] for i in positions]
        for k, index in enumerate(positions):
            contexts[index] = prepare_context(parts[:k + 1])
    missing = [i for i, c in enumerate(contexts) if c is None]
    if missing:
        raise SystemExit(f"{len(missing)} stimuli were not assigned a passage context")
    return [c for c in contexts if c is not None]


def extract_last_token_states(
    model_ref: str,
    sentences: list[str],
    *,
    device: str,
    batch_size: int,
    max_length: int
):
    """Last-token hidden state per stimulus, for every transformer block.

    Returns:
        dict[str, np.ndarray]: block name -> ``(n_stimuli, hidden)`` float32.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_ref)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    # Pin the side rather than inherit it. The last-real-token index below is
    # derived for right padding; a checkpoint whose tokenizer_config happens to
    # say "left" would silently shift every index and the error would look like
    # a corpus effect.
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_ref).to(device).eval()

    n_blocks = model.config.n_layer
    chunks: dict[int, list[np.ndarray]] = {i: [] for i in range(n_blocks)}
    n_truncated = 0

    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start:start + batch_size]
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)
            if int(enc["attention_mask"].sum(dim=1).max()) >= max_length:
                n_truncated += int((enc["attention_mask"].sum(dim=1) >= max_length).sum())
            out = model(**enc, output_hidden_states=True)
            # Right-padding puts the final real token at attention_mask.sum()-1,
            # NOT at position -1. Indexing -1 would read a pad state for every
            # sentence shorter than the batch maximum, which is most of them.
            last_idx = enc["attention_mask"].sum(dim=1) - 1
            rows = torch.arange(len(batch), device=out.hidden_states[0].device)
            for block in range(n_blocks):
                hs = out.hidden_states[block + 1]  # +1: [0] is the embedding output
                chunks[block].append(hs[rows, last_idx].float().cpu().numpy())

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    if n_truncated:
        # Do NOT abort. The benchmark itself truncates -- HuggingfaceSubject calls
        # tokenizer(context, truncation=True) with no max_length, so it falls back
        # to model_max_length (1024) with truncation_side="right". Matching that is
        # the whole point, so this reports rather than raises.
        #
        # It matters because on a benchmark whose context grows monotonically
        # within a group (Blank2014's story listening), right-truncation keeps the
        # FIRST 1024 tokens, which stop changing once the limit is passed. Every
        # later stimulus in that group then receives a byte-identical input and so
        # an identical representation. See report_input_collapse().
        print(
            f"  NOTE: {n_truncated} contexts hit max_length={max_length} and were "
            "right-truncated, exactly as the benchmark does",
            flush=True
        )
    return {f"transformer.h.{i}": np.concatenate(chunks[i], axis=0) for i in range(n_blocks)}


def report_input_collapse(contexts: list[str], model_ref: str, max_length: int) -> dict:
    """How many stimuli receive an input identical to an earlier stimulus.

    A validity check, not a diagnostic. Where the cumulative context outgrows the
    model's window, right-truncation freezes the input, and every stimulus past
    that point in the group is handed the same tokens. Its representation cannot
    vary with the stimulus, so an encoder can only predict a constant for those
    rows -- which floors the achievable correlation and compresses differences
    between models. A benchmark where this happens for a large fraction of
    stimuli cannot cleanly rank training corpora, and the number belongs in the
    result rather than in a footnote.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_ref)
    ids = [tuple(tok(c, truncation=True, max_length=max_length)["input_ids"]) for c in contexts]
    distinct = len(set(ids))
    return {
        "n_stimuli": len(ids),
        "n_distinct_model_inputs": distinct,
        "n_collapsed": len(ids) - distinct,
        "fraction_collapsed": (len(ids) - distinct) / max(len(ids), 1),
        "max_length": max_length
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sentences, brain, presentation, neuroid = load_stimuli(args.benchmark)

    # The subject coordinate is named differently on every benchmark: "subject"
    # on Pereira, "subject_id" on Blank, "subject_UID" on Fedorenko and Tuckute.
    # Resolved once, here, because it is used both in the banner and in the
    # per-subject RSA below.
    subject_col = next((c for c in ("subject", "subject_id", "subject_UID") if c in neuroid.columns), None)
    n_subjects = neuroid[subject_col].nunique() if subject_col else 0
    print(
        f"benchmark {args.benchmark}: {len(sentences)} stimuli, {brain.shape[1]} neuroids, "
        f"{n_subjects} subjects (subject coord: {subject_col!r})",
        flush=True
    )

    group_coord = BENCHMARKS[args.benchmark]["group"]
    contexts = build_passage_contexts(presentation, group_coord)
    n_passages = presentation[group_coord].nunique()
    longest = max(len(c.split()) for c in contexts)
    print(
        f"contexts: {n_passages} groups by {group_coord!r}, longest {longest} "
        f"whitespace tokens (cumulative within-group, matching the benchmark)",
        flush=True
    )

    collapse = report_input_collapse(contexts, DEFAULT_COLLAPSE_TOKENIZER, args.max_length)
    if collapse["n_collapsed"]:
        print(
            f"  WARNING: {collapse['n_collapsed']}/{collapse['n_stimuli']} stimuli "
            f"({100 * collapse['fraction_collapsed']:.1f}%) receive an input identical "
            "to an earlier stimulus after right-truncation, so their representation "
            "cannot vary with the stimulus",
            flush=True
        )

    models = discover_models(args.models_dir, args.hf_model, args.limit_models)
    print(f"{len(models)} checkpoints", flush=True)

    states: dict[str, dict[str, np.ndarray]] = {}
    for name, ref in models:
        states[name] = extract_last_token_states(
            ref,
            contexts,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        print(f"  {name}: {len(states[name])} blocks {next(iter(states[name].values())).shape}", flush=True)

    layers = sorted(next(iter(states.values())), key=lambda s: int(s.rsplit(".", 1)[1]))

    subjects = sorted(neuroid[subject_col].astype(str).unique()) if subject_col else []
    if subject_col is None:
        print("  no subject coordinate found; reporting pooled RSA only", flush=True)

    # shape: covariance-spectrum profile of every (model, block)
    geometry = {
        name: {layer: spectral_profile(states[name][layer]) for layer in layers}
        for name in states
    }

    # similarity: linear CKA between every pair of models at each block
    cka: dict[str, dict[str, float]] = {}
    for layer in layers:
        per_layer = {}
        for a, b in combinations(sorted(states), 2):
            per_layer[f"{a}|{b}"] = linear_cka(states[a][layer], states[b][layer])
        cka[layer] = per_layer
    print(f"CKA: {len(layers)} layers x {len(next(iter(cka.values())))} pairs", flush=True)

    # correspondence: the brain RDMs do not depend on the model, so they are built once
    subject_masks = (
        {s: (neuroid[subject_col].astype(str) == s).to_numpy() for s in subjects}
        if subject_col else {}
    )
    # A single-neuroid benchmark (Tuckute) has a one-dimensional representational
    # space, where correlation distance -- which profiles each stimulus ACROSS
    # neuroids -- is undefined. Euclidean is well defined at any width but the
    # resulting rho is not comparable to a correlation-based one, so record it.
    brain_metric = "correlation" if brain.shape[1] >= 3 else "euclidean"
    brain_rdms = {"pooled": rdm(brain, brain_metric)}
    for subject, mask in subject_masks.items():
        if mask.sum() >= 3:
            brain_rdms[f"subject_{subject}"] = rdm(brain[:, mask], brain_metric)
    print(f"brain RDMs: {len(brain_rdms)} (pooled + {len(subject_masks)} subjects)", flush=True)

    brain_rsa: dict[str, dict[str, dict[str, float]]] = {}
    for name in states:
        per_model = {}
        for layer in layers:
            model_rdm = rdm(states[name][layer])
            per_model[layer] = {key: rsa_from_rdms(model_rdm, target) for key, target in brain_rdms.items()}
        brain_rsa[name] = per_model

    # whether the correspondence exceeds a within-passage permutation null
    rsa_null: dict[str, dict[str, dict[str, float]]] = {}
    passage_labels = presentation[group_coord].astype(str).tolist()
    group_sizes: dict[str, int] = {}
    for label in passage_labels:
        group_sizes[label] = group_sizes.get(label, 0) + 1
    null_possible = bool(group_sizes) and max(group_sizes.values()) > 1
    if args.permutations > 0 and not null_possible:
        print(
            f"  permutation null SKIPPED: grouping {group_coord!r} gives singleton "
            "groups, so a within-group permutation cannot move anything",
            flush=True
        )
    if args.permutations > 0 and null_possible:
        brain_square = squareform(brain_rdms["pooled"])
        rng = np.random.default_rng(20260819)
        for name in sorted(states):
            per_model = {}
            for layer in layers:
                observed = brain_rsa[name][layer]["pooled"]
                null = grouped_permutation_null(
                    square_rdm(states[name][layer]),
                    brain_square,
                    passage_labels,
                    rng=rng,
                    n_permutations=args.permutations
                )
                per_model[layer] = {
                    "observed": observed,
                    "null_mean": float(np.nanmean(null)),
                    "null_sd": float(np.nanstd(null)),
                    "null_p95": float(np.nanpercentile(null, 95)),
                    "z_vs_null": float((observed - np.nanmean(null)) / np.nanstd(null))
                    if np.nanstd(null) > 0 else float("nan"),
                    "p_value": permutation_p_value(observed, null)
                }
            rsa_null[name] = per_model
            peak = max(v["z_vs_null"] for v in per_model.values())
            print(f"  null {name}: peak z={peak:+.2f}", flush=True)

    payload = {
        "schema": "latent_geometry",
        "n_permutations": args.permutations,
        "rsa_permutation_null": rsa_null,
        "benchmark": BENCHMARKS[args.benchmark]["id"],
        "benchmark_key": args.benchmark,
        "group_coord": group_coord,
        "n_stimuli": len(sentences),
        "representation": "last_token_of_cumulative_within_group_context",
        "brain_rdm_metric": brain_metric,
        "rsa_null_possible": null_possible,
        "model_input_collapse": collapse,
        "representation_note": (
            "Matches the benchmark on both axes: the last token "
            "(HuggingfaceSubject.output_to_representations, values[:, -1:, :]) of the "
            "cumulative within-passage context (_Pereira2018Experiment.__call__ feeds "
            "prepare_context(text[:k+1]) per passage). Isolated-sentence extraction "
            "would describe a different object from the one the encoder scores."
        ),
        "n_passages": int(n_passages),
        "result_status": "exploratory_only",
        "models": sorted(states),
        "layers": layers,
        "subjects": subjects,
        "geometry": geometry,
        "cka": cka,
        "brain_rsa": brain_rsa,
        # Record whichever grouping coordinates this benchmark actually has, not
        # a fixed Pereira pair. Downstream analyses need the grouping the
        # benchmark cross-validates over, and its name differs per benchmark.
        "stimulus_grouping": {
            name: presentation[name].astype(str).tolist()
            for name in dict.fromkeys([
                group_coord,
                "story",
                "passage_label",
                "sentence_id",
                "stimulus_id"
            ])
            if name and name in presentation.columns
        }
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
