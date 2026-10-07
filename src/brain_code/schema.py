"""Versioned, fail-closed schema for scientific result artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "2.0"

_RESULT_STATUSES = {
    "valid",
    "exploratory_only",
    "retrospective_not_confirmatory",
    "invalid",
    "stale",
    "partial",
    "failed",
    "rerun_required",
    "blocked_external_dependency"
}
_BLOCKED_STATUSES = {
    "invalid",
    "stale",
    "partial",
    "failed",
    "rerun_required",
    "blocked_external_dependency"
}
_TEST_STATUSES = {
    "passed",
    "failed",
    "not_run",
    "not_applicable",
    "blocked_external_dependency"
}
_IMPLEMENTATION_STATUSES = {
    "implemented",
    "partial",
    "not_implemented",
    "blocked_external_dependency"
}
_CONCLUSION_STATUSES = {
    "valid",
    "exploratory_only",
    "retrospective_not_confirmatory",
    "rerun_required",
    "invalid",
    "withdrawn",
    "blocked_external_dependency"
}

_REQUIRED = {
    "schema_version",
    "experiment_id",
    "run_id",
    "result_status",
    "implementation_status",
    "unit_test_status",
    "integration_test_status",
    "real_data_rerun_status",
    "conclusion_status",
    "created_at",
    "git_commit",
    "git_dirty",
    "brainscore_submodule_commit",
    "environment_lock_hash",
    "python_version",
    "command",
    "config_path",
    "config_hash",
    "consumed_config_fields",
    "ignored_config_fields",
    "model_identifier",
    "model_revision",
    "model_config_hash",
    "checkpoint_hash",
    "tokenizer_identifier",
    "tokenizer_revision",
    "tokenizer_hash",
    "benchmark_identifier",
    "benchmark_version",
    "benchmark_modality",
    "benchmark_data_checksum",
    "stimulus_ids_hash",
    "stimulus_order_hash",
    "split_manifest_hash",
    "random_seeds",
    "layer",
    "layer_selection_method",
    "context_mode",
    "pooling",
    "feature_preprocessing",
    "target_preprocessing",
    "regression_method",
    "alpha_grid",
    "alpha_selection_strategy",
    "inner_cv_definition",
    "outer_cv_definition",
    "grouping_coordinate",
    "primary_metric",
    "aggregation_method",
    "warnings",
    "errors",
    "completeness",
    "validated_against_official"
}


REQUIRED_FIELDS: frozenset[str] = frozenset(_REQUIRED)


class ResultSchemaError(ValueError):
    """Raised when a result record violates the schema."""


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(value: Any) -> str:
    """Hashes a JSON-serializable value with sorted keys and compact separators."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str
    )
    return sha256_bytes(payload.encode())


def _git(args: list[str], cwd: str | Path = ".") -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _submodule_commit(root: Path, path: str) -> str | None:
    """Returns the initialized gitlink commit, never the parent repository HEAD."""
    status = _git(
        [
            "submodule",
            "status",
            "--",
            path
        ],
        root
    )
    if not status:
        return None
    # `git submodule status` marks uninitialized entries with `-`, modified ones with `+`
    # and clean ones with a space; an uninitialized submodule is not valid provenance, while
    # a modified one keeps its checked-out commit and the parent's dirty flag catches it
    marker = status[0]
    if marker == "-":
        return None
    fields = status[1:].strip().split() if marker in {" ", "+", "U"} else status.split()
    if not fields:
        return None
    commit = fields[0]
    if len(commit) != 40 or any(character not in "0123456789abcdefABCDEF" for character in commit):
        return None
    return commit.lower()


def collect_repository_provenance(repo_root: str | Path = ".") -> dict[str, Any]:
    """Collects the git, submodule and environment-lock provenance fields."""
    root = Path(repo_root)
    lock = root / "uv.lock"
    status = _git(["status", "--porcelain"], root)
    return {
        "git_commit": _git(["rev-parse", "HEAD"], root),
        "git_dirty": bool(status) if status is not None else None,
        "brainscore_submodule_commit": _submodule_commit(root, "language"),
        "environment_lock_hash": sha256_file(lock),
        "python_version": platform.python_version()
    }


def _nonempty(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def is_immutable_revision(value: Any) -> bool:
    """Returns whether a model or tokenizer revision is content-addressed.

    Hugging Face commit revisions are full 40-character git SHAs. Local model
    directories are represented by the explicit ``local-sha256:<digest>`` form.
    Arbitrary branch names, tags, and short SHAs are rejected because they do
    not bind a result to one immutable set of files.
    """
    revision = str(value or "").strip()
    return bool(
        re.fullmatch(r"[0-9a-fA-F]{40}", revision)
        or re.fullmatch(r"local-sha256:[0-9a-fA-F]{64}", revision)
    )


def require_immutable_revision(value: Any, *, field_name: str = "revision") -> str:
    """Returns a normalized immutable revision, raising before any expensive work.

    Raises:
        ValueError: If the revision is not a full commit SHA or a local-sha256 digest.
    """
    revision = str(value or "").strip()
    if not is_immutable_revision(revision):
        raise ValueError(
            f"{field_name} must be a full 40-character Hugging Face commit SHA "
            "or local-sha256:<64-hex-digest>; mutable branches, tags, and short "
            f"SHAs are forbidden (got {value!r})."
        )
    if revision.startswith("local-sha256:"):
        return "local-sha256:" + revision.split(":", 1)[1].lower()
    return revision.lower()


def validate_result(data: dict[str, Any], *, require_complete_provenance: bool = True) -> None:
    """Checks a result record against the schema and its status rules.

    Raises:
        ResultSchemaError: Listing every violation found.
    """
    errors: list[str] = []
    missing = sorted(_REQUIRED - data.keys())
    if missing:
        errors.append(f"missing required fields: {missing}")
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}; got {data.get('schema_version')!r}")
    if data.get("result_status") not in _RESULT_STATUSES:
        errors.append(f"invalid result_status: {data.get('result_status')!r}")
    if data.get("implementation_status") not in _IMPLEMENTATION_STATUSES:
        errors.append(f"invalid implementation_status: {data.get('implementation_status')!r}")
    for field in ("unit_test_status", "integration_test_status", "real_data_rerun_status"):
        if data.get(field) not in _TEST_STATUSES:
            errors.append(f"invalid {field}: {data.get(field)!r}")
    if data.get("conclusion_status") not in _CONCLUSION_STATUSES:
        errors.append(f"invalid conclusion_status: {data.get('conclusion_status')!r}")

    status = data.get("result_status")
    if status == "valid":
        required_test_states = {
            "unit_test_status": "passed",
            "integration_test_status": "passed",
            "real_data_rerun_status": "passed"
        }
        for field, expected in required_test_states.items():
            if data.get(field) != expected:
                errors.append(f"valid result requires {field}={expected!r}")
    if status in {"valid", "exploratory_only", "retrospective_not_confirmatory"}:
        if data.get("completeness") != "complete":
            errors.append("renderable-status result must have completeness='complete'")
        if data.get("errors"):
            errors.append("renderable-status result may not contain errors")
        if require_complete_provenance:
            provenance_fields = (
                "git_commit",
                "brainscore_submodule_commit",
                "environment_lock_hash",
                "config_hash",
                "model_revision",
                "model_config_hash",
                "checkpoint_hash",
                "tokenizer_revision",
                "tokenizer_hash",
                "benchmark_version",
                "benchmark_data_checksum",
                "stimulus_ids_hash",
                "stimulus_order_hash",
                "split_manifest_hash"
            )
            absent = [field for field in provenance_fields if not _nonempty(data.get(field))]
            if absent:
                errors.append(f"renderable-status result lacks provenance: {absent}")
        if data.get("git_dirty") is not False:
            errors.append("renderable-status result requires git_dirty=false")
        for revision_field in ("model_revision", "tokenizer_revision"):
            if not is_immutable_revision(data.get(revision_field)):
                errors.append(
                    "renderable-status result requires a full 40-character git SHA "
                    f"or local-sha256 digest for {revision_field}"
                )

    if (
        status == "valid"
        and str(data.get("primary_metric", "")).startswith("custom_encoding_")
        and data.get("validated_against_official") is not True
    ):
        errors.append("a valid custom-encoding result requires validated_against_official=true")

    selection = data.get("layer_selection_method")
    if selection == "test_set_argmax":
        errors.append("test_set_argmax is forbidden as an inferential layer selection method")
    if selection == "prespecified" and data.get("layer") in {None, "all", "sweep"}:
        errors.append("prespecified layer selection requires one explicit layer")
    if (
        data.get("result_status") == "valid"
        and selection == "prespecified"
        and not data.get("selection_evidence_hash")
    ):
        errors.append(
            "a valid prespecified-layer result requires selection_evidence_hash; "
            "otherwise mark it exploratory_only"
        )
    if selection == "nested_cv" and not data.get("nested_selection_artifact_hash"):
        errors.append("nested_cv requires nested_selection_artifact_hash")

    benchmark = str(data.get("benchmark_identifier", ""))
    if "384sentences" in benchmark and data.get("legacy_schrimpf2021_pereira243_normalized") is not None:
        errors.append("Pereira-384 may not contain the legacy Pereira-243 normalization")
    if (
        benchmark != "Pereira2018.243sentences-ridge"
        and data.get("legacy_schrimpf2021_pereira243_normalized") is not None
    ):
        errors.append("legacy Pereira-243 normalization is benchmark-specific")

    if data.get("ignored_config_fields"):
        errors.append("ignored_config_fields must be empty")
    if errors:
        raise ResultSchemaError(
            f"Result schema validation failed ({len(errors)} issue(s)):\n"
            + "\n".join(f"  - {error}" for error in errors)
        )


def is_renderable(data: dict[str, Any], *, allow_exploratory: bool = False) -> bool:
    """Returns whether a result is valid and has a status that may be plotted or tabulated."""
    try:
        validate_result(data)
    except ResultSchemaError:
        return False
    status = data["result_status"]
    if status == "valid":
        return True
    return allow_exploratory and status in {"exploratory_only", "retrospective_not_confirmatory"}


def load_and_validate(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_result(data)
    return data


def load_validated_results(result_dir: str | Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(Path(result_dir).glob("*.json")):
        results.append(load_and_validate(path))
    return results


def load_renderable_results(result_dir: str | Path, *, allow_exploratory: bool = False) -> list[dict[str, Any]]:
    return [
        result
        for result in load_validated_results(result_dir)
        if is_renderable(result, allow_exploratory=allow_exploratory)
    ]


def make_base_result(
    *,
    experiment_id: str,
    run_id: str,
    model_identifier: str,
    benchmark_identifier: str,
    result_status: str,
    conclusion_status: str,
    repo_root: str | Path = ".",
    **overrides: Any
) -> dict[str, Any]:
    """Builds a schema-complete result record with provenance filled in and `overrides` applied."""
    provenance = collect_repository_provenance(repo_root)
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "result_status": result_status,
        "implementation_status": "implemented",
        "unit_test_status": "not_run",
        "integration_test_status": "not_run",
        "real_data_rerun_status": "not_run",
        "conclusion_status": conclusion_status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **provenance,
        "command": " ".join(sys.argv),
        "config_path": None,
        "config_hash": None,
        "consumed_config_fields": [],
        "ignored_config_fields": [],
        "model_identifier": model_identifier,
        "model_revision": None,
        "model_config_hash": None,
        "checkpoint_hash": None,
        "tokenizer_identifier": model_identifier,
        "tokenizer_revision": None,
        "tokenizer_hash": None,
        "benchmark_identifier": benchmark_identifier,
        "benchmark_version": None,
        "benchmark_modality": None,
        "benchmark_data_checksum": None,
        "stimulus_ids_hash": None,
        "stimulus_order_hash": None,
        "split_manifest_hash": None,
        "random_seeds": {},
        "layer": None,
        "layer_selection_method": "prespecified",
        "context_mode": None,
        "pooling": None,
        "feature_preprocessing": None,
        "target_preprocessing": None,
        "regression_method": None,
        "alpha_grid": [],
        "alpha_selection_strategy": None,
        "inner_cv_definition": None,
        "outer_cv_definition": None,
        "grouping_coordinate": None,
        "primary_metric": None,
        "aggregation_method": None,
        "warnings": [],
        "errors": [],
        "completeness": "partial",
        "validated_against_official": False
    }
    base.update(overrides)
    return base


def write_result(path: str | Path, data: dict[str, Any]) -> None:
    """Validates a result and writes it atomically as sorted, indented JSON."""
    validate_result(data)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
            default=str
        ) + "\n",
        encoding="utf-8"
    )
    temporary.replace(output)


def require_consumed_config(config: dict[str, Any], consumed_fields: Iterable[str]) -> None:
    """Raises ValueError if the config has fields the caller did not consume."""
    consumed = set(consumed_fields)
    ignored = sorted(set(config) - consumed)
    if ignored:
        raise ValueError(f"Configuration fields were not consumed: {ignored}")
