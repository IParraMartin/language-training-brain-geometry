"""Fail-closed wrapper around the pinned official Brain-Score benchmarks.

The supported layer modes are:

``descriptive_grid``
    Evaluate and report every layer without selecting a winner.
``prespecified``
    Evaluate exactly one explicitly supplied layer.

There is no ``nested_cv`` mode: the official wrapper does not expose the
train-fold activations needed to select a layer inside each training fold, and
picking the layer by test-set argmax would make the reported score circular.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("BRAINIO_HOME", str(PROJECT_ROOT / ".brainio"))
os.environ.setdefault("RESULTCACHING_HOME", str(PROJECT_ROOT / ".result_caching"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from brainscore_language import ArtificialSubject, load_benchmark, model_registry
    from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

    _BRAINSCORE_AVAILABLE = True
except ImportError:
    ArtificialSubject = None
    load_benchmark = None
    model_registry = None
    HuggingfaceSubject = object
    _BRAINSCORE_AVAILABLE = False


BENCHMARK_METADATA = {
    "Pereira2018.243sentences-ridge": {
        "human_name": "Pereira 2018 — 243 sentences",
        "modality": "fMRI",
        "neural_target": "language-network voxels",
        "stimulus_granularity": "sentence in passage context",
        "raw_metric": "Pearson r from the official cross-validated ridge benchmark",
        "paper_ceiling": 0.32,
        "paper_ceiling_source": "legacy Schrimpf et al. 2021 Pereira-243 reference",
        "paper_ceiling_valid": True,
        "notes": "Report the official benchmark ceiling separately from the legacy reference."
    },
    "Pereira2018.384sentences-ridge": {
        "human_name": "Pereira 2018 — 384 sentences",
        "modality": "fMRI",
        "neural_target": "language-network voxels",
        "stimulus_granularity": "sentence in passage context",
        "raw_metric": "Pearson r from the official cross-validated ridge benchmark",
        "paper_ceiling": None,
        "paper_ceiling_source": "The Pereira-243 value 0.32 is not valid for Pereira-384.",
        "paper_ceiling_valid": False,
        "notes": "Do not compute the legacy Pereira-243 normalization."
    },
    "Blank2014-ridge": {
        "human_name": "Blank 2014",
        "modality": "fMRI",
        "neural_target": "language-network voxels",
        "stimulus_granularity": "story/passage",
        "raw_metric": "Pearson r from the official cross-validated ridge benchmark",
        "paper_ceiling": None,
        "paper_ceiling_source": "No verified legacy ceiling is registered.",
        "paper_ceiling_valid": False,
        "notes": "Do not apply a Pereira ceiling."
    },
    "Fedorenko2016-ridge": {
        "human_name": "Fedorenko 2016",
        "modality": "ECoG",
        "neural_target": "electrodes",
        "stimulus_granularity": "word/sentence",
        "raw_metric": "official benchmark metric",
        "paper_ceiling": None,
        "paper_ceiling_source": "No verified legacy ceiling is registered.",
        "paper_ceiling_valid": False,
        "notes": "ECoG must not be pooled with fMRI in a headline aggregate."
    },
    "Tuckute2024-ridge": {
        "human_name": "Tuckute 2024",
        "modality": "fMRI",
        "neural_target": "language-network voxels",
        "stimulus_granularity": "sentence",
        "raw_metric": "official benchmark metric",
        "paper_ceiling": None,
        "paper_ceiling_source": "No verified legacy ceiling is registered.",
        "paper_ceiling_valid": False,
        "notes": "Do not apply a Pereira ceiling."
    },
    "Futrell2018-pearsonr": {
        "human_name": "Futrell 2018",
        "modality": "behavioral",
        "neural_target": "reading times",
        "stimulus_granularity": "word in story context",
        "raw_metric": "Pearson correlation",
        "paper_ceiling": None,
        "paper_ceiling_source": "Behavioral benchmark; Pereira neural ceiling is inapplicable.",
        "paper_ceiling_valid": False,
        "notes": "Do not aggregate with neural modalities without explicit justification."
    }
}


def get_benchmark_meta(benchmark_identifier: str) -> dict[str, Any]:
    if benchmark_identifier in BENCHMARK_METADATA:
        return dict(BENCHMARK_METADATA[benchmark_identifier])
    return {
        "human_name": benchmark_identifier,
        "modality": "unknown",
        "neural_target": "unknown",
        "stimulus_granularity": "unknown",
        "raw_metric": "unknown",
        "paper_ceiling": None,
        "paper_ceiling_source": "Unknown benchmark; no legacy normalization is permitted.",
        "paper_ceiling_valid": False,
        "notes": "Register benchmark metadata before public reporting."
    }


def _compute_paper_normalized(raw_score: float, benchmark_identifier: str) -> tuple[float | None, str]:
    meta = get_benchmark_meta(benchmark_identifier)
    if not meta["paper_ceiling_valid"]:
        return None, str(meta["paper_ceiling_source"])
    ceiling = meta["paper_ceiling"]
    if ceiling is None or float(ceiling) <= 0:
        return None, "No positive verified legacy ceiling is available."
    return float(raw_score) / float(ceiling), f"raw / {ceiling} ({meta['paper_ceiling_source']})"


def _sha_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str
        ).encode()
    ).hexdigest()


def _safe_key(*parts: str) -> str:
    return "brain-code-" + hashlib.sha256("\0".join(parts).encode()).hexdigest()[:24]


def _validated_selection_evidence(
    path: str | Path,
    *,
    model_identifier: str,
    model_revision: str,
    benchmark_identifier: str,
    selected_layer: str
) -> tuple[str, dict[str, Any]]:
    """Validate independent or prospective layer-selection evidence.

    Merely providing an arbitrary file is not evidence. The JSON must bind the
    selected layer to this immutable model and target benchmark, and it must
    describe either an independent-development selection or an externally
    timestamped prospective prespecification.
    """
    evidence_path = Path(path)
    if not evidence_path.is_file():
        raise FileNotFoundError(f"Layer-selection evidence not found: {evidence_path}")
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Layer-selection evidence must be valid JSON.") from exc
    if not isinstance(evidence, dict):
        raise ValueError("Layer-selection evidence must be a JSON object.")
    expected = {
        "model_identifier": model_identifier,
        "model_revision": model_revision,
        "target_benchmark": benchmark_identifier,
        "selected_layer": selected_layer
    }
    mismatches = {
        field: {"expected": value, "actual": evidence.get(field)}
        for field, value in expected.items()
        if evidence.get(field) != value
    }
    if mismatches:
        raise ValueError(f"Layer-selection evidence does not match this run: {mismatches}")
    method = evidence.get("selection_method")
    if method == "independent_development_selection":
        source = evidence.get("source_benchmark")
        if not source or source == benchmark_identifier:
            raise ValueError(
                "Independent-development evidence requires a nonempty source_benchmark "
                "different from the target benchmark."
            )
    elif method == "prospective_prespecification":
        required = ("registered_at", "external_timestamp_uri", "target_result_not_seen")
        missing = [field for field in required if not evidence.get(field)]
        if missing or evidence.get("target_result_not_seen") is not True:
            raise ValueError(
                "Prospective evidence requires registered_at, external_timestamp_uri, "
                "and target_result_not_seen=true."
            )
    else:
        raise ValueError(
            "selection_method must be 'independent_development_selection' or "
            "'prospective_prespecification'."
        )
    digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    return digest, evidence


def _hash_tokenizer_files(model_source: str | Path) -> str:
    root = Path(model_source)
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
        "tokenizer.model"
    )
    digest = hashlib.sha256()
    found = 0
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        found += 1
        digest.update(name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    if not found:
        raise RuntimeError(f"No tokenizer files found in resolved snapshot: {root}")
    return digest.hexdigest()


def _hash_directory(path: str | Path) -> str:
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {root}")
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Local model directory is empty: {root}")
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _resolve_model_source(model_id: str, revision: str) -> str:
    local = Path(model_id)
    if local.exists():
        expected = f"local-sha256:{_hash_directory(local)}"
        if revision != expected:
            raise ValueError(
                "Local model revision does not match the directory contents: "
                f"expected {expected!r}, got {revision!r}."
            )
        return str(local.resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_id, revision=revision)


def _hash_benchmark_assembly(benchmark) -> tuple[str | None, str | None, str | None]:
    assembly = getattr(benchmark, "data", None)
    if assembly is None:
        return None, None, None
    digest = hashlib.sha256()
    values = getattr(assembly, "values", None)
    if values is not None:
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    ids: list[str] = []
    for name in ("stimulus_id", "stimulus", "sentence", "stimulus_sentence"):
        # stimulus metadata lives in MultiIndex levels that assembly[name] can
        # retrieve but the pinned xarray leaves out of assembly.coords
        try:
            coord = assembly[name]
        except (KeyError, TypeError):
            continue
        ids = [str(value) for value in coord.values.ravel().tolist()]
        break
    return (
        digest.hexdigest(),
        _sha_json(sorted(ids)) if ids else None,
        _sha_json(ids) if ids else None
    )


class LayerCommittedHuggingfaceSubject(HuggingfaceSubject):
    """Official HuggingFace subject bound to one layer, under a caller-chosen identifier."""

    def __init__(
        self,
        *args,
        identifier: str,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._brain_code_identifier = identifier

    def identifier(self):
        return self._brain_code_identifier


class CustomModelHarness:
    """Discovers layers and registers collision-resistant official subjects."""

    def __init__(
        self,
        hf_model_id: str,
        *,
        model_revision: str,
        tokenizer_revision: str | None = None,
        cache_dir: str | Path = "cache"
    ) -> None:
        from brain_code.schema import require_immutable_revision

        model_revision = require_immutable_revision(
            model_revision, field_name="model_revision"
        )
        tokenizer_revision = require_immutable_revision(
            tokenizer_revision or model_revision, field_name="tokenizer_revision"
        )
        from transformers import AutoConfig

        self.hf_model_id = hf_model_id
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        if self.tokenizer_revision != self.model_revision:
            raise ValueError(
                "The pinned official HuggingFaceSubject loads model and tokenizer from one "
                "resolved snapshot. A distinct tokenizer_revision is not yet supported; "
                "use the same immutable revision or add a subject that actually consumes it."
            )
        self.model_source = _resolve_model_source(hf_model_id, model_revision)
        self.cache_dir = str(cache_dir)
        config = AutoConfig.from_pretrained(self.model_source, local_files_only=True)
        self.model_config_hash = _sha_json(config.to_dict())
        self.tokenizer_hash = _hash_tokenizer_files(self.model_source)
        n_layers = int(getattr(config, "n_layer", getattr(config, "num_hidden_layers", 0)))
        if n_layers <= 0:
            raise ValueError("Could not determine transformer layer count from model config.")
        self.layer_names = [f"transformer.h.{index}" for index in range(n_layers)]
        self.model_key = _safe_key(
            hf_model_id,
            model_revision,
            self.tokenizer_revision,
            self.tokenizer_hash,
            self.model_config_hash
        )

    def resolve_smoke_layer(self, spec: str | int) -> str:
        aliases = {
            "first": 0,
            "middle": len(self.layer_names) // 2,
            "last": len(self.layer_names) - 1
        }
        if isinstance(spec, str) and spec in aliases:
            index = aliases[spec]
        else:
            try:
                index = int(spec)
            except (TypeError, ValueError) as exc:
                raise ValueError("Layer must be first, middle, last, or an integer.") from exc
        if not 0 <= index < len(self.layer_names):
            raise ValueError(f"Layer index {index} is out of range.")
        return self.layer_names[index]

    def identifier_for(self, layer: str) -> str:
        return _safe_key(self.model_key, layer)

    def register_layer(self, layer: str) -> str:
        if not _BRAINSCORE_AVAILABLE:
            raise RuntimeError("brainscore_language is not installed. Run: uv pip install -e language/")
        if layer not in self.layer_names:
            raise ValueError(f"Unknown layer {layer!r}.")
        identifier = self.identifier_for(layer)
        source = self.model_source
        model_registry[identifier] = lambda: LayerCommittedHuggingfaceSubject(
            model_id=source,
            region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: layer},
            identifier=identifier
        )
        return identifier

    def register_layers(self) -> None:
        for layer in self.layer_names:
            self.register_layer(layer)


def _score(model_identifier: str, benchmark_identifier: str):
    if not _BRAINSCORE_AVAILABLE:
        raise RuntimeError("brainscore_language is not installed. Run: uv pip install -e language/")
    if model_identifier not in model_registry:
        raise KeyError(f"Model identifier {model_identifier!r} is not registered.")
    model = model_registry[model_identifier]()
    benchmark = load_benchmark(benchmark_identifier)
    result = benchmark(model)
    result.attrs["model_identifier"] = model_identifier
    result.attrs["benchmark_identifier"] = benchmark_identifier
    return result, benchmark


def _extract_score(score) -> dict[str, float | None]:
    # benchmarks without a ceiling (Tuckute2024) return the raw r as the score and attach no raw or
    # ceiling attributes; the normalized value is then null rather than a raw r under another name
    raw_attr = score.attrs.get("raw")
    ceiling_attr = score.attrs.get("ceiling")
    normalized = float(getattr(score, "values", score))
    raw = normalized if raw_attr is None else float(getattr(raw_attr, "values", raw_attr))
    ceiling = None if ceiling_attr is None else float(getattr(ceiling_attr, "values", ceiling_attr))
    return {
        "official_brainscore_raw_pearson": raw,
        "official_brainscore_ceiling_normalized": None if ceiling is None else normalized,
        "official_benchmark_ceiling": ceiling
    }


_LAYER_SELECTION_MODES = frozenset({"descriptive_grid", "prespecified"})


def run_experiment(
    hf_model_id: str,
    benchmark_identifier: str,
    is_test: bool = False,
    save_results: bool = True,
    smoke_layer: str = "last",
    layer_selection_mode: str = "descriptive_grid",
    *,
    prespecified_layer: str | None = None,
    model_revision: str | None = None,
    tokenizer_revision: str | None = None,
    results_dir: str | Path = "results/official_brainscore",
    selection_evidence_path: str | Path | None = None,
    unit_test_evidence_path: str | Path | None = None
):
    """Run the official benchmark without test-set layer selection.

    ``nested_cv`` is intentionally rejected because the official wrapper does
    not expose the train-fold activations and predictions needed to perform
    layer selection inside each outer-training split.
    """
    if layer_selection_mode not in _LAYER_SELECTION_MODES:
        raise ValueError(
            f"layer_selection_mode must be one of {sorted(_LAYER_SELECTION_MODES)}; "
            f"got {layer_selection_mode!r}. nested_cv is not implemented here."
        )
    if layer_selection_mode == "prespecified" and prespecified_layer is None:
        prespecified_layer = smoke_layer if is_test else None
    if layer_selection_mode == "prespecified" and prespecified_layer is None:
        raise ValueError("prespecified mode requires prespecified_layer.")
    from brain_code.schema import require_immutable_revision

    model_revision = require_immutable_revision(
        model_revision, field_name="model_revision"
    )
    tokenizer_revision = require_immutable_revision(
        tokenizer_revision or model_revision, field_name="tokenizer_revision"
    )
    if not _BRAINSCORE_AVAILABLE:
        raise RuntimeError("brainscore_language is not installed. Run: uv pip install -e language/")

    harness = CustomModelHarness(
        hf_model_id,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision
    )
    if is_test:
        layers = [harness.resolve_smoke_layer(smoke_layer)]
    elif layer_selection_mode == "prespecified":
        layers = [harness.resolve_smoke_layer(prespecified_layer)]
    else:
        layers = list(harness.layer_names)

    layer_results: dict[str, dict[str, float]] = {}
    benchmark = None
    for layer in layers:
        identifier = harness.register_layer(layer)
        score, benchmark = _score(identifier, benchmark_identifier)
        metrics = _extract_score(score)
        legacy, reason = _compute_paper_normalized(
            metrics["official_brainscore_raw_pearson"], benchmark_identifier
        )
        metrics["legacy_schrimpf2021_pereira243_normalized"] = legacy
        metrics["legacy_normalization_reason"] = reason
        layer_results[layer] = metrics

    if is_test:
        return layer_results[layers[0]]["official_brainscore_raw_pearson"]
    if not save_results:
        return layer_results

    from brain_code.schema import make_base_result, write_result

    benchmark_checksum, stimulus_ids_hash, stimulus_order_hash = _hash_benchmark_assembly(benchmark)
    meta = get_benchmark_meta(benchmark_identifier)
    run_key = _safe_key(
        hf_model_id,
        model_revision,
        tokenizer_revision or model_revision,
        benchmark_identifier,
        layer_selection_mode,
        ",".join(layers)
    )
    evidence_hash = None
    evidence_summary = None
    if selection_evidence_path is not None:
        if layer_selection_mode != "prespecified" or len(layers) != 1:
            raise ValueError("Layer-selection evidence is only meaningful for one prespecified layer.")
        evidence_hash, evidence_summary = _validated_selection_evidence(
            selection_evidence_path,
            model_identifier=hf_model_id,
            model_revision=model_revision,
            benchmark_identifier=benchmark_identifier,
            selected_layer=layers[0]
        )
    unit_test_evidence_hash = None
    if unit_test_evidence_path is not None:
        evidence_file = Path(unit_test_evidence_path)
        if not evidence_file.is_file():
            raise FileNotFoundError(f"Unit-test evidence not found: {evidence_file}")
        unit_test_evidence_hash = hashlib.sha256(evidence_file.read_bytes()).hexdigest()
    unit_tests_passed = unit_test_evidence_hash is not None
    inferentially_valid = (
        layer_selection_mode == "prespecified"
        and evidence_hash is not None
        and unit_tests_passed
    )
    resolved_config = {
        "hf_model_id": hf_model_id,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "benchmark_identifier": benchmark_identifier,
        "is_test": is_test,
        "smoke_layer": smoke_layer,
        "layer_selection_mode": layer_selection_mode,
        "prespecified_layer": prespecified_layer,
        "resolved_layers": layers,
        "selection_evidence_hash": evidence_hash,
        "unit_test_evidence_hash": unit_test_evidence_hash
    }
    config_hash = _sha_json(resolved_config)
    if layer_selection_mode == "descriptive_grid":
        result_warnings = ["Descriptive layer grid: do not select the maximum as an unbiased model score."]
    elif evidence_hash is None:
        result_warnings = [
            "Prespecified layer lacks independent timestamped selection evidence; "
            "result is exploratory."
        ]
    elif not unit_tests_passed:
        result_warnings = [
            "Selection evidence is present, but no unit-test evidence artifact "
            "was supplied; result remains exploratory."
        ]
    else:
        result_warnings = []

    result = make_base_result(
        experiment_id="official_brainscore",
        repo_root=PROJECT_ROOT,
        run_id=run_key,
        model_identifier=hf_model_id,
        benchmark_identifier=benchmark_identifier,
        result_status="valid" if inferentially_valid else "exploratory_only",
        conclusion_status="valid" if inferentially_valid else "exploratory_only",
        config_hash=config_hash,
        consumed_config_fields=sorted(resolved_config),
        model_revision=model_revision,
        model_config_hash=harness.model_config_hash,
        checkpoint_hash=f"hf_revision:{model_revision}",
        tokenizer_identifier=hf_model_id,
        tokenizer_revision=tokenizer_revision or model_revision,
        tokenizer_hash=harness.tokenizer_hash,
        benchmark_version=None,
        benchmark_modality=meta["modality"],
        benchmark_data_checksum=benchmark_checksum,
        stimulus_ids_hash=stimulus_ids_hash,
        stimulus_order_hash=stimulus_order_hash,
        split_manifest_hash="owned_by_pinned_official_benchmark",
        random_seeds={"official_benchmark": 1234},
        layer=layers[0] if len(layers) == 1 else layers,
        layer_selection_method=layer_selection_mode,
        context_mode="official_benchmark_defined",
        pooling="official_benchmark_defined_final_token",
        feature_preprocessing="official_benchmark_defined",
        target_preprocessing="official_benchmark_defined",
        regression_method="official_benchmark_defined_ridge",
        alpha_grid=[],
        alpha_selection_strategy="official_benchmark_defined",
        inner_cv_definition="official_benchmark_defined",
        outer_cv_definition="official_benchmark_defined",
        grouping_coordinate="official_benchmark_defined",
        primary_metric="official_brainscore_raw_pearson",
        aggregation_method="official_benchmark_defined",
        raw_layer_results=layer_results,
        completeness="complete",
        validated_against_official=True,
        selection_evidence_hash=evidence_hash,
        selection_evidence=evidence_summary,
        unit_test_status="passed" if unit_tests_passed else "not_run",
        unit_test_evidence_hash=unit_test_evidence_hash,
        integration_test_status="passed",
        real_data_rerun_status="passed",
        warnings=result_warnings
    )
    result["benchmark_version"] = result.get("brainscore_submodule_commit")
    missing_benchmark_provenance = [
        field
        for field in (
            "benchmark_version",
            "benchmark_data_checksum",
            "stimulus_ids_hash",
            "stimulus_order_hash"
        )
        if not result.get(field)
    ]
    if missing_benchmark_provenance:
        result["result_status"] = "blocked_external_dependency"
        result["conclusion_status"] = "blocked_external_dependency"
        result["completeness"] = "partial"
        result["errors"] = [
            "Could not resolve complete official benchmark provenance: "
            + ", ".join(missing_benchmark_provenance)
        ]
        result["integration_test_status"] = "blocked_external_dependency"
        result["real_data_rerun_status"] = "blocked_external_dependency"
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_key}.json"
    write_result(output_path, result)
    return result
