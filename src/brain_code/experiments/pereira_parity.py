"""Strict intermediate-artifact parity checks for Pereira evaluation.

A parity claim is only meaningful when the official and custom paths agree on
all inputs and intermediate quantities that determine the final score.  This
module therefore compares stimulus identity/order, the *separate* context and
cross-validation groupings, tokenization, block activations, neural targets,
fold membership, predictions, and score outputs.

The comparator never creates an official reference.  Both bundles must be
exported independently on a machine with the pinned ``brainscore_language``
submodule and benchmark data available.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

# These fields are required on both sides.  Keep them explicit so an old,
# under-specified bundle cannot accidentally certify a new pipeline.
MANDATORY_FIELDS = (
    "stimulus_ids",
    "stimulus_texts",
    "cv_group_ids",
    "context_group_ids",
    "contexts",
    "token_ids_json",
    "attention_masks_json",
    "final_token_positions",
    "activations",
    "neural_neuroid_ids",
    "neural_targets",
    "fold_ids",
    "predictions",
    "per_neuroid_correlations",
    "raw_score"
)

_EXACT_FIELDS = (
    "stimulus_ids",
    "stimulus_texts",
    "cv_group_ids",
    "context_group_ids",
    "contexts",
    "token_ids_json",
    "attention_masks_json",
    "final_token_positions",
    "neural_neuroid_ids",
    "fold_ids"
)

_NUMERIC_FIELDS = (
    "activations",
    "neural_targets",
    "predictions",
    "per_neuroid_correlations",
    "raw_score"
)

_OPTIONAL_EXACT_FIELDS = (
    "model_identifier",
    "model_revision",
    "tokenizer_revision",
    "layer_identifier",
    "benchmark_identifier"
)

_OPTIONAL_NUMERIC_FIELDS = ("ceiling_value", "ceiling_normalized_score")


@dataclass(frozen=True)
class ParityTolerance:
    activation_atol: float = 1e-6
    activation_rtol: float = 1e-5
    target_atol: float = 0.0
    target_rtol: float = 0.0
    prediction_atol: float = 1e-7
    prediction_rtol: float = 1e-6
    score_atol: float = 1e-8
    score_rtol: float = 1e-7


def _array(value: Any, field: str) -> NDArray:
    result = np.asarray(value)
    if result.size == 0:
        raise ValueError(f"Parity field {field!r} is empty.")
    return result


def _hash_array(value: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    if array.dtype.kind in {"O", "U", "S"}:
        digest.update(json.dumps(array.tolist(), ensure_ascii=False, separators=(",", ":")).encode())
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def load_parity_bundle(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Parity bundle not found: {source}")
    if source.suffix.casefold() == ".npz":
        with np.load(source, allow_pickle=False) as loaded:
            bundle = {name: loaded[name] for name in loaded.files}
    elif source.suffix.casefold() == ".json":
        bundle = json.loads(source.read_text(encoding="utf-8"))
    else:
        raise ValueError("Parity bundles must be NPZ or JSON.")
    _require_fields(bundle, label=str(source))
    return bundle


def _require_fields(bundle: Mapping[str, Any], *, label: str) -> None:
    missing = [field for field in MANDATORY_FIELDS if field not in bundle]
    if missing:
        raise ValueError(f"{label} parity bundle lacks mandatory fields: {missing}")


def _exact_check(reference: Any, custom: Any, field: str) -> dict[str, Any]:
    left = _array(reference, field)
    right = _array(custom, field)
    same_shape = left.shape == right.shape
    equal = bool(same_shape and np.array_equal(left, right))
    first_mismatch: dict[str, Any] | None = None
    if same_shape and not equal:
        mismatch = np.argwhere(left != right)
        if len(mismatch):
            index = tuple(int(item) for item in mismatch[0])
            first_mismatch = {
                "index": list(index),
                "reference": str(left[index]),
                "custom": str(right[index])
            }
    return {
        "passed": equal,
        "reference_shape": list(left.shape),
        "custom_shape": list(right.shape),
        "reference_hash": _hash_array(left),
        "custom_hash": _hash_array(right),
        "first_mismatch": first_mismatch
    }


def _numeric_check(
    reference: Any,
    custom: Any,
    field: str,
    *,
    atol: float,
    rtol: float
) -> dict[str, Any]:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(custom, dtype=np.float64)
    same_shape = left.shape == right.shape
    finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    first_mismatch: dict[str, Any] | None = None
    if same_shape and finite:
        absolute = np.abs(left - right)
        max_abs = float(absolute.max(initial=0.0))
        denominator = np.maximum(np.abs(left), np.finfo(np.float64).eps)
        max_rel = float((absolute / denominator).max(initial=0.0))
        close = np.isclose(
            left,
            right,
            atol=atol,
            rtol=rtol,
            equal_nan=False
        )
        passed = bool(close.all())
        if not passed:
            index = tuple(int(item) for item in np.argwhere(~close)[0])
            first_mismatch = {
                "index": list(index),
                "reference": float(left[index]),
                "custom": float(right[index]),
                "absolute_difference": float(absolute[index])
            }
    else:
        max_abs = float("inf")
        max_rel = float("inf")
        passed = False
    return {
        "passed": passed,
        "finite": finite,
        "reference_shape": list(left.shape),
        "custom_shape": list(right.shape),
        "max_absolute_difference": max_abs,
        "max_relative_difference": max_rel,
        "atol": float(atol),
        "rtol": float(rtol),
        "reference_hash": _hash_array(left),
        "custom_hash": _hash_array(right),
        "first_mismatch": first_mismatch
    }


def _field_tolerance(field: str, tolerance: ParityTolerance) -> tuple[float, float]:
    if field == "activations":
        return tolerance.activation_atol, tolerance.activation_rtol
    if field == "neural_targets":
        return tolerance.target_atol, tolerance.target_rtol
    if field == "predictions":
        return tolerance.prediction_atol, tolerance.prediction_rtol
    return tolerance.score_atol, tolerance.score_rtol


def compare_pereira_parity(
    reference: Mapping[str, Any],
    custom: Mapping[str, Any],
    *,
    tolerance: ParityTolerance = ParityTolerance()
) -> dict[str, Any]:
    """Compare every artifact required for a real Pereira parity claim.

    Optional metadata/ceiling fields are compared whenever either side supplies
    them.  Supplying a field on only one side is itself a failed check.
    """
    _require_fields(reference, label="reference")
    _require_fields(custom, label="custom")

    checks: dict[str, dict[str, Any]] = {
        field: _exact_check(reference[field], custom[field], field)
        for field in _EXACT_FIELDS
    }
    for field in _NUMERIC_FIELDS:
        atol, rtol = _field_tolerance(field, tolerance)
        checks[field] = _numeric_check(
            reference[field],
            custom[field],
            field,
            atol=atol,
            rtol=rtol
        )

    for field in _OPTIONAL_EXACT_FIELDS:
        present_reference = field in reference
        present_custom = field in custom
        if present_reference and present_custom:
            checks[field] = _exact_check(reference[field], custom[field], field)
        elif present_reference or present_custom:
            checks[field] = {
                "passed": False,
                "reason": "field_present_on_only_one_side",
                "reference_present": present_reference,
                "custom_present": present_custom
            }

    for field in _OPTIONAL_NUMERIC_FIELDS:
        present_reference = field in reference
        present_custom = field in custom
        if present_reference and present_custom:
            checks[field] = _numeric_check(
                reference[field],
                custom[field],
                field,
                atol=tolerance.score_atol,
                rtol=tolerance.score_rtol
            )
        elif present_reference or present_custom:
            checks[field] = {
                "passed": False,
                "reason": "field_present_on_only_one_side",
                "reference_present": present_reference,
                "custom_present": present_custom
            }

    ordered_fields = [*MANDATORY_FIELDS, *_OPTIONAL_EXACT_FIELDS, *_OPTIONAL_NUMERIC_FIELDS]
    first_failed = next(
        (field for field in ordered_fields if field in checks and not checks[field]["passed"]),
        None
    )
    passed = first_failed is None
    payload = {
        "validated_against_official": bool(passed),
        "first_failed_field": first_failed,
        "checks": checks,
        "tolerance": {
            "activation_atol": tolerance.activation_atol,
            "activation_rtol": tolerance.activation_rtol,
            "target_atol": tolerance.target_atol,
            "target_rtol": tolerance.target_rtol,
            "prediction_atol": tolerance.prediction_atol,
            "prediction_rtol": tolerance.prediction_rtol,
            "score_atol": tolerance.score_atol,
            "score_rtol": tolerance.score_rtol
        }
    }
    payload["parity_artifact_hash"] = hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str
    ).encode()).hexdigest()
    return payload
