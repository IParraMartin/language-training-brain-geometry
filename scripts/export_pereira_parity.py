"""Export one strict Pereira parity bundle from the official or custom path.

Run both sides with the same immutable model/tokenizer revision and layer, then
compare them with ``scripts/run_pereira_parity.py``.  This command requires the
pinned ``brainscore_language`` submodule, benchmark data, and model weights.
It never marks parity as passed by itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_model_and_tokenizer(model_id: str, revision: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from brain_code.schema import require_immutable_revision

    revision = require_immutable_revision(revision, field_name="--model-revision")
    if revision.startswith("local-sha256:"):
        raise ValueError("The parity exporter currently requires a Hugging Face commit SHA.")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, truncation_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, use_safetensors=True
    )
    return model, tokenizer


def _build_candidate(model_id: str, revision: str, layer: str):
    from brainscore_language import ArtificialSubject
    from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

    model, tokenizer = _load_model_and_tokenizer(model_id, revision)
    candidate = HuggingfaceSubject(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        region_layer_mapping={
            ArtificialSubject.RecordingTarget.language_system: [layer]
        }
    )
    return candidate


def _official_source(benchmark, candidate):
    """Execute the pinned benchmark's passage loop without reordering globally."""
    import xarray as xr
    from brainscore_language import ArtificialSubject

    candidate.start_neural_recording(
        recording_target=ArtificialSubject.RecordingTarget.language_system,
        recording_type=ArtificialSubject.RecordingType.fMRI
    )
    stimuli = benchmark.data["stimulus"]
    # group the way each benchmark's own __call__ does, because the grouping decides how much preceding
    # context a stimulus is scored with: passage_label (Pereira2018), story (Blank2014), sentence_id
    # (Fedorenko2016), stimulus_id (Tuckute2024)
    for group_coord in ("passage_label", "story", "sentence_id", "stimulus_id"):
        try:
            passage_labels = benchmark.data[group_coord].values
            break
        except (KeyError, TypeError):
            continue
    else:
        raise KeyError(
            "benchmark.data carries none of passage_label/story/sentence_id/stimulus_id; "
            "cannot reproduce the official grouping"
        )
    predictions = []
    for passage in sorted(set(passage_labels)):
        passage_indexer = np.asarray(passage_labels) == passage
        passage_stimuli = stimuli[passage_indexer]
        passage_predictions = candidate.digest_text(passage_stimuli.values)["neural"]
        # Fedorenko2016's metric splits on sentence_id and requires it on both source and target
        for coord in (
            "stimulus_id",
            "passage_label",
            "passage_index",
            "story",
            "sentence_id",
            "run_id"
        ):
            # Brain-Score assemblies store these as MultiIndex levels, which are
            # retrievable via benchmark.data[coord] but absent from .coords under
            # the pinned xarray; probe retrievability so the source actually
            # carries stimulus_id/story/passage_label for parity alignment.
            try:
                coord_values = benchmark.data[coord].values
            except (KeyError, TypeError):
                continue
            passage_predictions[coord] = "presentation", coord_values[passage_indexer]
        predictions.append(passage_predictions)
    # Do not sort by part_number after concatenation: part_number restarts at 0
    # in every passage, so a global sort would interleave passages.
    return xr.concat(predictions, dim="presentation")


def _custom_target_assembly(data):
    """Package custom targets with the exact official stimulus/group identities."""
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

    neuroid_ids = np.asarray(data.neuroid_ids)
    coords = {
        "stimulus_id": ("presentation", np.asarray(data.raw_stimulus_ids)),
        "stimulus": ("presentation", np.asarray(data.sentences, dtype="U")),
        "story": ("presentation", np.asarray(data.groups)),
        "passage_label": ("presentation", np.asarray(data.context_groups)),
        "neuroid_id": ("neuroid", neuroid_ids),
        # The pinned official metric's XarrayRegression._package_prediction has a
        # fragile single-neuroid-coordinate branch: when the target carries exactly
        # one neuroid coord it renames the `neuroid` dim to that coord and re-stacks,
        # which raises "cannot create a new dimension with the same name as an
        # existing dimension". Official Pereira assemblies carry ~10 neuroid coords
        # and never hit it; give the custom target a second neuroid coordinate so the
        # metric packages predictions on the `neuroid` dim exactly like the official
        # path. `neuroid_number` is positional metadata the comparator ignores (it
        # keys neuroids on `neuroid_id`).
        "neuroid_number": ("neuroid", np.arange(len(neuroid_ids), dtype=np.int64))
    }
    return NeuroidAssembly(
        data.responses.astype(np.float64),
        dims=("presentation", "neuroid"),
        coords=coords
    )


def _custom_source_assembly(
    data,
    model_id: str,
    revision: str,
    layer: str
):
    """Extract custom block activations under official Pereira context behavior."""
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

    from brain_code.experiments.data_loading import build_contexts, extract_hidden_states

    model, tokenizer = _load_model_and_tokenizer(model_id, revision)
    contexts = build_contexts(
        data.sentences,
        np.asarray(data.context_groups),
        context_mode="cumulative_passage"
    )
    representations = extract_hidden_states(
        model,
        tokenizer,
        data.sentences,
        layer=layer,
        pooling="last",
        groups=np.asarray(data.context_groups),
        context_mode="cumulative_passage"
    )
    n_features = representations.shape[1]
    neuroid_ids = np.asarray([f"{layer}--{index}" for index in range(n_features)], dtype="U")
    return (
        NeuroidAssembly(
            representations.astype(np.float64),
            dims=("presentation", "neuroid"),
            coords={
                "stimulus_id": ("presentation", np.asarray(data.raw_stimulus_ids)),
                "stimulus": ("presentation", np.asarray(data.sentences, dtype="U")),
                "story": ("presentation", np.asarray(data.groups)),
                "passage_label": ("presentation", np.asarray(data.context_groups)),
                "context": ("presentation", np.asarray(contexts, dtype="U")),
                "layer": ("neuroid", np.asarray([layer] * n_features, dtype="U")),
                "neuron_number_in_layer": ("neuroid", np.arange(n_features)),
                "neuroid_id": ("neuroid", neuroid_ids)
            }
        ),
        tokenizer
    )


def _scalar_ceiling(benchmark) -> float | None:
    ceiling = getattr(benchmark, "ceiling", None)
    if ceiling is None:
        return None
    values = np.asarray(getattr(ceiling, "values", ceiling), dtype=np.float64)
    if values.size != 1 or not np.isfinite(values).all() or float(values.reshape(-1)[0]) <= 0:
        return None
    return float(values.reshape(-1)[0])


def _write_bundle(bundle: dict, output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bundle)
    print(f"Wrote parity bundle: {path}")
    print(f"  fields: {sorted(bundle.keys())}")
    print(
        f"  n_stimuli: {len(bundle['stimulus_ids'])}, "
        f"raw_score: {np.asarray(bundle['raw_score']).tolist()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=["official", "custom"], required=True)
    parser.add_argument("--benchmark", default="Pereira2018.243sentences-ridge")
    parser.add_argument(
        "--model-id",
        "--model_id",
        dest="model_id",
        required=True
    )
    parser.add_argument(
        "--model-revision",
        "--model_revision",
        dest="model_revision",
        required=True
    )
    parser.add_argument(
        "--tokenizer-revision",
        "--tokenizer_revision",
        dest="tokenizer_revision",
        default=None,
        help="Currently must equal --model-revision; separate tokenizer repositories are not supported."
    )
    parser.add_argument("--layer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from brain_code.schema import require_immutable_revision

    args.model_revision = require_immutable_revision(
        args.model_revision, field_name="--model-revision"
    )
    tokenizer_revision = require_immutable_revision(
        args.tokenizer_revision or args.model_revision,
        field_name="--tokenizer-revision"
    )
    if tokenizer_revision != args.model_revision:
        parser.error(
            "The parity exporter currently loads tokenizer and model from the same repository; "
            "--tokenizer-revision must equal --model-revision."
        )

    try:
        from brainscore_language import load_benchmark
    except ImportError as exc:
        raise SystemExit(
            f"brainscore_language not importable ({exc}). Initialize/install the pinned "
            "language/ submodule before exporting parity artifacts."
        ) from exc

    from brain_code.experiments.pereira_parity_export import (
        bundle_from_source_and_target
    )

    benchmark = load_benchmark(args.benchmark)
    if args.side == "official":
        candidate = _build_candidate(args.model_id, args.model_revision, args.layer)
        source = _official_source(benchmark, candidate)
        target = benchmark.data
        tokenizer = candidate.tokenizer
    else:
        from brain_code.experiments.data_loading import load_neural_benchmark

        data = load_neural_benchmark(args.benchmark)
        target = _custom_target_assembly(data)
        source, tokenizer = _custom_source_assembly(
            data,
            args.model_id,
            args.model_revision,
            args.layer
        )

    metadata = {
        "model_identifier": args.model_id,
        "model_revision": args.model_revision,
        "tokenizer_revision": tokenizer_revision,
        "layer_identifier": args.layer,
        "benchmark_identifier": args.benchmark
    }
    bundle = bundle_from_source_and_target(
        source,
        target,
        benchmark.metric,
        tokenizer=tokenizer,
        metadata=metadata,
        ceiling_value=_scalar_ceiling(benchmark)
    )
    _write_bundle(bundle, args.output)
    print("PARITY BUNDLE EXPORT COMPLETE")


if __name__ == "__main__":
    main()
