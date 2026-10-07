"""Export strict Pereira parity artifacts from an activation/target pair.

The official and custom exporters both feed the *same pinned benchmark metric*.
This module records the metric's held-out predictions while preserving enough
metadata to prove that the two paths evaluated identical stimuli, context/CV
groups, tokenization, activations, and neural targets.

Nothing in this module marks parity as passed.  Only
:func:`brain_code.experiments.pereira_parity.compare_pereira_parity` can do so,
after independently exported bundles are compared.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from brain_code.experiments.pereira_parity import MANDATORY_FIELDS  # noqa: F401

_STIMULUS_ID_COORD = "stimulus_id"
_STIMULUS_TEXT_COORD = "stimulus"
_CV_GROUP_COORD = "story"
_CONTEXT_GROUP_COORD = "passage_label"
_CONTEXT_COORD = "context"
_NEUROID_ID_COORD = "neuroid_id"


@dataclass
class RecordedFolds:
    """Per-split intermediates captured from the official metric."""

    test_stimulus_ids: list[NDArray] = field(default_factory=list)
    predictions: list[NDArray] = field(default_factory=list)
    prediction_neuroid_ids: list[NDArray] = field(default_factory=list)
    per_split_correlations: list[NDArray] = field(default_factory=list)


def _has_coord(assembly, name: str) -> bool:
    """True if ``name`` is retrievable from ``assembly``.

    Brain-Score assemblies pack their metadata (``stimulus_id``, ``story``,
    ``neuroid_id`` …) into pandas MultiIndex levels on the ``presentation`` and
    ``neuroid`` dims. In the pinned xarray, those levels are accessible via
    ``assembly[name]`` but do NOT appear in ``assembly.coords``. Testing
    membership against ``.coords`` therefore wrongly rejects present metadata,
    so we probe retrievability directly.
    """
    try:
        assembly[name]
    except KeyError:
        return False
    return True


def _coord_values(assembly, name: str) -> NDArray:
    if not _has_coord(assembly, name):
        raise ValueError(f"Assembly lacks required coordinate {name!r}.")
    return np.asarray(assembly[name].values)


def _unique_id_index(assembly, *, label: str) -> tuple[NDArray, dict[Any, int]]:
    ids = _coord_values(assembly, _STIMULUS_ID_COORD)
    if ids.ndim != 1 or len(ids) != int(assembly.sizes["presentation"]):
        raise ValueError(f"{label} stimulus_id must be one-dimensional over presentation.")
    mapping: dict[Any, int] = {}
    duplicates: list[Any] = []
    for index, value in enumerate(ids.tolist()):
        if value in mapping:
            duplicates.append(value)
        mapping[value] = index
    if duplicates:
        raise ValueError(f"{label} stimulus_id is not unique; examples={duplicates[:10]}.")
    return ids, mapping


def _canonical_alignment(source, target) -> tuple[NDArray, NDArray, NDArray]:
    source_ids, source_map = _unique_id_index(source, label="source")
    target_ids, target_map = _unique_id_index(target, label="target")
    source_set = set(source_ids.tolist())
    target_set = set(target_ids.tolist())
    if source_set != target_set:
        missing_source = sorted(map(str, target_set - source_set))[:10]
        missing_target = sorted(map(str, source_set - target_set))[:10]
        raise ValueError(
            "Source/target stimulus IDs differ; "
            f"missing_from_source={missing_source}, missing_from_target={missing_target}."
        )
    canonical_ids = np.asarray(sorted(target_ids.tolist(), key=str))
    source_order = np.asarray([source_map[value] for value in canonical_ids.tolist()])
    target_order = np.asarray([target_map[value] for value in canonical_ids.tolist()])
    return canonical_ids, source_order, target_order


def build_recording_metric(benchmark_metric):
    """Wrap the pinned metric and record each held-out prediction.

    The official regression, cross-validation, and correlation objects are
    reused unchanged.  ``apply`` only tees off artifacts after the official fit,
    prediction, and correlation operations have occurred.
    """
    records = RecordedFolds()
    metric_type = type(benchmark_metric)

    class _Recording(metric_type):  # type: ignore[valid-type,misc]
        def apply(
            self,
            source_train,
            target_train,
            source_test,
            target_test
        ):
            self.regression.fit(source_train, target_train)
            prediction = self.regression.predict(source_test)
            score = self.correlation(prediction, target_test)

            test_ids = _coord_values(target_test, _STIMULUS_ID_COORD)
            prediction_ids = _coord_values(prediction, _STIMULUS_ID_COORD)
            if set(test_ids.tolist()) != set(prediction_ids.tolist()):
                raise ValueError("Prediction and target test stimulus IDs differ.")
            order = np.argsort(prediction_ids.astype(str), kind="stable")
            sorted_ids = prediction_ids[order]
            prediction_values = np.asarray(prediction.values, dtype=np.float64)[order]
            neuroid_ids = (
                _coord_values(prediction, _NEUROID_ID_COORD)
                if _has_coord(prediction, _NEUROID_ID_COORD)
                else np.arange(prediction_values.shape[1])
            )
            records.test_stimulus_ids.append(sorted_ids)
            records.predictions.append(prediction_values)
            records.prediction_neuroid_ids.append(np.asarray(neuroid_ids))
            records.per_split_correlations.append(np.asarray(score.values, dtype=np.float64).reshape(-1))
            return score

    recording = _Recording.__new__(_Recording)
    recording.__dict__.update(benchmark_metric.__dict__)
    return recording, records


def _assemble_oof(
    canonical_ids: NDArray,
    records: RecordedFolds,
    expected_neuroid_ids: NDArray | None = None
) -> tuple[NDArray, NDArray]:
    """Assemble OOF predictions and fold IDs in canonical stimulus order."""
    if not records.predictions:
        raise ValueError("The recording metric produced no held-out predictions.")
    id_to_row = {sid: index for index, sid in enumerate(canonical_ids.tolist())}
    first_neuroid_ids = np.asarray(records.prediction_neuroid_ids[0])
    if expected_neuroid_ids is not None and not np.array_equal(
        first_neuroid_ids.astype(str),
        np.asarray(expected_neuroid_ids).astype(str)
    ):
        raise ValueError("Prediction neuroid order does not match target neuroid order.")
    n = len(canonical_ids)
    n_neuroid = records.predictions[0].shape[1]
    oof = np.full((n, n_neuroid), np.nan, dtype=np.float64)
    fold_ids = np.full(n, -1, dtype=np.int64)

    for fold_index, (test_ids, predictions, neuroid_ids) in enumerate(zip(
        records.test_stimulus_ids,
        records.predictions,
        records.prediction_neuroid_ids
    )):
        if predictions.ndim != 2 or len(predictions) != len(test_ids):
            raise ValueError(f"Fold {fold_index} prediction shape is inconsistent with IDs.")
        if predictions.shape[1] != n_neuroid:
            raise ValueError("Prediction target width changes across folds.")
        if not np.array_equal(np.asarray(neuroid_ids).astype(str), first_neuroid_ids.astype(str)):
            raise ValueError("Prediction neuroid order changes across folds.")
        for local_row, stimulus_id in enumerate(test_ids.tolist()):
            if stimulus_id not in id_to_row:
                raise ValueError(f"Fold {fold_index} test stimulus {stimulus_id!r} absent from targets.")
            global_row = id_to_row[stimulus_id]
            if fold_ids[global_row] != -1:
                raise ValueError(f"Stimulus {stimulus_id!r} assigned to more than one test fold.")
            fold_ids[global_row] = fold_index
            oof[global_row] = predictions[local_row]

    missing = [canonical_ids[index] for index in range(n) if fold_ids[index] == -1]
    if missing:
        raise ValueError(f"Outer folds leave {len(missing)} stimuli unscored; missing={missing[:10]}")
    if not np.isfinite(oof).all():
        raise ValueError("OOF predictions contain NaN or infinity.")
    return oof, fold_ids


def _json_rows(rows: list[list[int]]) -> NDArray:
    return np.asarray([json.dumps(row, separators=(",", ":")) for row in rows], dtype="U")


def tokenization_artifacts(tokenizer, contexts: NDArray) -> dict[str, NDArray]:
    """Tokenize each context exactly once, matching official no-padding calls."""
    original_truncation_side = getattr(tokenizer, "truncation_side", None)
    tokenizer.truncation_side = "left"
    token_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    final_positions: list[int] = []
    try:
        for context in np.asarray(contexts).astype(str).tolist():
            encoded = tokenizer(
                context,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True
            )
            token_ids = np.asarray(encoded["input_ids"], dtype=np.int64).reshape(-1).tolist()
            attention = np.asarray(
                encoded.get("attention_mask", np.ones(len(token_ids), dtype=np.int64)),
                dtype=np.int64
            ).reshape(-1).tolist()
            if not token_ids or len(token_ids) != len(attention):
                raise ValueError("Tokenizer produced an empty or malformed context encoding.")
            valid = np.flatnonzero(np.asarray(attention, dtype=bool))
            if not len(valid):
                raise ValueError("Tokenizer produced a context with no attended tokens.")
            token_rows.append([int(value) for value in token_ids])
            mask_rows.append([int(value) for value in attention])
            final_positions.append(int(valid[-1]))
    finally:
        if original_truncation_side is not None:
            tokenizer.truncation_side = original_truncation_side
    return {
        "token_ids_json": _json_rows(token_rows),
        "attention_masks_json": _json_rows(mask_rows),
        "final_token_positions": np.asarray(final_positions, dtype=np.int64)
    }


def _aggregate_per_neuroid(records: RecordedFolds, n_neuroid: int) -> NDArray:
    values = [np.asarray(item, dtype=np.float64).reshape(-1) for item in records.per_split_correlations]
    if not values or any(len(item) != n_neuroid for item in values):
        raise ValueError("Per-split correlation widths do not match the neural targets.")
    return np.mean(np.stack(values, axis=0), axis=0)


def _raw_scalar(score, per_neuroid: NDArray) -> float:
    values = np.asarray(score.values, dtype=np.float64)
    if values.size == 1 and np.isfinite(values).all():
        return float(values.reshape(-1)[0])
    # This fallback is diagnostic only.  Most pinned metric versions return a
    # scalar aggregate; refusing nonfinite/ambiguous values is safer than
    # silently flattening an arbitrary Score.
    finite = per_neuroid[np.isfinite(per_neuroid)]
    if not len(finite):
        raise ValueError("The metric produced neither a scalar nor finite neuroid scores.")
    return float(np.median(finite))


def bundle_from_source_and_target(
    source_assembly,
    target_assembly,
    benchmark_metric,
    *,
    tokenizer,
    metadata: Mapping[str, str] | None = None,
    ceiling_value: float | None = None
) -> dict[str, Any]:
    """Run the pinned metric and emit a strict parity bundle."""
    canonical_ids, source_order, target_order = _canonical_alignment(source_assembly, target_assembly)
    recording, records = build_recording_metric(benchmark_metric)
    score = recording(source_assembly, target_assembly)

    target_neuroid_ids = (
        _coord_values(target_assembly, _NEUROID_ID_COORD)
        if _has_coord(target_assembly, _NEUROID_ID_COORD)
        else np.arange(int(target_assembly.sizes["neuroid"]))
    )
    oof_predictions, fold_ids = _assemble_oof(canonical_ids, records, expected_neuroid_ids=target_neuroid_ids)
    per_neuroid = _aggregate_per_neuroid(records, len(target_neuroid_ids))
    raw_score = _raw_scalar(score, per_neuroid)

    contexts = _coord_values(source_assembly, _CONTEXT_COORD)[source_order].astype("U")
    bundle: dict[str, Any] = {
        "stimulus_ids": canonical_ids.astype("U"),
        "stimulus_texts": _coord_values(target_assembly, _STIMULUS_TEXT_COORD)[target_order].astype("U"),
        "cv_group_ids": _coord_values(target_assembly, _CV_GROUP_COORD)[target_order].astype("U"),
        "context_group_ids": _coord_values(target_assembly, _CONTEXT_GROUP_COORD)[target_order].astype("U"),
        "contexts": contexts,
        "activations": np.asarray(source_assembly.values, dtype=np.float64)[source_order],
        "neural_neuroid_ids": np.asarray(target_neuroid_ids).astype("U"),
        "neural_targets": np.asarray(target_assembly.values, dtype=np.float64)[target_order],
        "fold_ids": fold_ids,
        "predictions": oof_predictions,
        "per_neuroid_correlations": per_neuroid.astype(np.float64),
        "raw_score": np.asarray([raw_score], dtype=np.float64),
        **tokenization_artifacts(tokenizer, contexts)
    }
    if metadata:
        for field, value in metadata.items():
            bundle[field] = np.asarray([str(value)], dtype="U")
    if ceiling_value is not None:
        ceiling = float(ceiling_value)
        if not np.isfinite(ceiling) or ceiling <= 0:
            raise ValueError("ceiling_value must be finite and positive.")
        bundle["ceiling_value"] = np.asarray([ceiling], dtype=np.float64)
        bundle["ceiling_normalized_score"] = np.asarray([raw_score / ceiling], dtype=np.float64)
    missing = [field for field in MANDATORY_FIELDS if field not in bundle]
    if missing:
        raise AssertionError(f"Internal error: parity bundle is incomplete: {missing}")
    return bundle
