"""Tests for result schema validation in brain_code.schema."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from brain_code.schema import (
    SCHEMA_VERSION,
    REQUIRED_FIELDS,
    ResultSchemaError,
    is_renderable,
    load_and_validate,
    make_base_result,
    require_immutable_revision,
    validate_result
)


def _stub_result(**overrides) -> dict:
    """Minimal valid rerun_required stub built from make_base_result defaults."""
    base = make_base_result(
        experiment_id="test_exp",
        run_id="test_run",
        model_identifier="gpt2",
        benchmark_identifier="Pereira2018.243sentences-ridge",
        result_status="rerun_required",
        conclusion_status="invalid"
    )
    # a prespecified layer_selection_method would need a layer, so use the grid
    base.update({
        "implementation_status": "blocked_external_dependency",
        "unit_test_status": "not_run",
        "integration_test_status": "not_run",
        "real_data_rerun_status": "not_run",
        "layer_selection_method": "descriptive_grid"
    })
    base.update(overrides)
    return base


def _full_valid_result(**overrides) -> dict:
    """Valid renderable result with all required provenance fields."""
    base = _stub_result(
        result_status="valid",
        implementation_status="implemented",
        conclusion_status="valid",
        unit_test_status="passed",
        integration_test_status="passed",
        real_data_rerun_status="passed",
        completeness="complete",
        errors=[],
        # provenance fields required for renderable results
        git_commit="abc1234deadbeefabc1234deadbeefabc1234de",
        git_dirty=False,
        brainscore_submodule_commit="a89732599a5a458baaf1695f3eb6c89056944bd2",
        environment_lock_hash="sha256:deadbeef" * 4,
        config_hash="sha256:01234567",
        model_revision="a" * 40,
        model_config_hash="sha256:cafecafe",
        checkpoint_hash="hf_revision:" + "a" * 40,
        tokenizer_revision="b" * 40,
        tokenizer_hash="sha256:beefdead",
        benchmark_version="1.0.0",
        benchmark_data_checksum="sha256:aabbccdd",
        stimulus_ids_hash="sha256:11223344",
        stimulus_order_hash="sha256:55667788",
        split_manifest_hash="sha256:99aabbcc",
        layer="last",
        layer_selection_method="descriptive_grid"
    )
    base.update(overrides)
    return base


def test_validate_result_accepts_valid_stub():
    validate_result(_stub_result())  # must not raise


def test_validate_result_accepts_full_valid():
    validate_result(_full_valid_result())  # must not raise


def test_validate_result_rejects_wrong_schema_version():
    r = _stub_result(schema_version="0.9")
    with pytest.raises(ResultSchemaError, match="schema_version"):
        validate_result(r)


def test_validate_result_rejects_missing_required_field():
    """Deleting any single required field must raise ResultSchemaError."""
    for field in REQUIRED_FIELDS:
        r = _stub_result()
        del r[field]
        with pytest.raises(ResultSchemaError):
            validate_result(r)


def test_validate_result_rejects_invalid_result_status():
    r = _stub_result(result_status="unknown_status")
    with pytest.raises(ResultSchemaError, match="result_status"):
        validate_result(r)


def test_validate_result_rejects_invalid_implementation_status():
    r = _stub_result(implementation_status="excellent")
    with pytest.raises(ResultSchemaError, match="implementation_status"):
        validate_result(r)


def test_validate_result_accepts_all_blocked_statuses():
    blocked = (
        "invalid",
        "stale",
        "partial",
        "failed",
        "rerun_required",
        "blocked_external_dependency"
    )
    for status in blocked:
        r = _stub_result(result_status=status)
        validate_result(r)  # must not raise


def test_validate_result_renderable_requires_completeness_and_provenance():
    """A 'valid' result missing provenance must fail."""
    r = _stub_result(
        result_status="valid",
        implementation_status="implemented",
        conclusion_status="valid"
    )
    with pytest.raises(ResultSchemaError):
        validate_result(r)


def test_validate_result_exploratory_only_requires_completeness():
    """An 'exploratory_only' result missing completeness must fail."""
    r = _stub_result(
        result_status="exploratory_only",
        implementation_status="implemented",
        conclusion_status="exploratory_only"
    )
    with pytest.raises(ResultSchemaError, match="completeness"):
        validate_result(r)


def test_validate_result_rejects_test_set_argmax_layer_selection():
    """test_set_argmax is forbidden as a layer selection method."""
    r = _stub_result(layer_selection_method="test_set_argmax")
    with pytest.raises(ResultSchemaError, match="test_set_argmax"):
        validate_result(r)


def test_validate_result_rejects_384_with_legacy_pereira243_ceiling():
    """Pereira-384 results must not carry the legacy Pereira-243 normalization."""
    r = _stub_result(
        benchmark_identifier="Pereira2018.384sentences-ridge",
        legacy_schrimpf2021_pereira243_normalized=0.2355
    )
    with pytest.raises(ResultSchemaError, match="384"):
        validate_result(r)


def test_is_renderable_true_for_valid():
    assert is_renderable(_full_valid_result()) is True


def test_is_renderable_true_for_exploratory_with_flag():
    """exploratory_only is renderable only when allow_exploratory=True."""
    r = _full_valid_result(
        result_status="exploratory_only",
        conclusion_status="exploratory_only"
    )
    assert is_renderable(r, allow_exploratory=True) is True
    assert is_renderable(r, allow_exploratory=False) is False


def test_is_renderable_true_for_retrospective_with_flag():
    """retrospective_not_confirmatory is renderable only when allow_exploratory=True."""
    r = _full_valid_result(
        result_status="retrospective_not_confirmatory",
        conclusion_status="retrospective_not_confirmatory"
    )
    assert is_renderable(r, allow_exploratory=True) is True
    assert is_renderable(r, allow_exploratory=False) is False


def test_is_renderable_false_for_invalid():
    assert is_renderable(_stub_result(result_status="invalid")) is False


def test_is_renderable_false_for_rerun_required():
    assert is_renderable(_stub_result(result_status="rerun_required")) is False


def test_is_renderable_false_for_stale():
    assert is_renderable(_stub_result(result_status="stale")) is False


def test_is_renderable_false_for_blocked():
    assert is_renderable(_stub_result(result_status="blocked_external_dependency")) is False


def test_load_and_validate_reads_json_file():
    r = _stub_result()
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(r, f)
        path = Path(f.name)
    try:
        loaded = load_and_validate(path)
        assert loaded["benchmark_identifier"] == r["benchmark_identifier"]
    finally:
        path.unlink()


def test_load_and_validate_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        load_and_validate(Path("/nonexistent/path/result.json"))


def test_load_and_validate_raises_on_invalid_schema():
    bad = {"schema_version": "999.0"}
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(bad, f)
        path = Path(f.name)
    try:
        with pytest.raises(ResultSchemaError):
            load_and_validate(path)
    finally:
        path.unlink()


def test_make_base_result_sets_schema_version():
    r = make_base_result(
        experiment_id="test",
        run_id="test_run",
        result_status="rerun_required",
        conclusion_status="invalid",
        model_identifier="gpt2",
        benchmark_identifier="Pereira2018.243sentences-ridge"
    )
    assert r["schema_version"] == SCHEMA_VERSION


def test_make_base_result_is_validatable():
    r = make_base_result(
        experiment_id="test",
        run_id="test_run",
        result_status="rerun_required",
        conclusion_status="invalid",
        model_identifier="gpt2",
        benchmark_identifier="Pereira2018.243sentences-ridge",
        layer_selection_method="descriptive_grid"
    )
    validate_result(r)  # must not raise


def test_make_base_result_blocked_is_not_renderable():
    r = make_base_result(
        experiment_id="test",
        run_id="test_run",
        result_status="rerun_required",
        conclusion_status="invalid",
        model_identifier="gpt2",
        benchmark_identifier="Pereira2018.243sentences-ridge",
        layer_selection_method="descriptive_grid"
    )
    assert is_renderable(r) is False


def test_pereira384_rejects_legacy_fixed_ceiling():
    """The schema rejects a Pereira-384 result carrying the legacy Pereira-243 ceiling.

    This is a schema error, not just a bad conclusion_status: the 0.32 ceiling
    normalization must not appear on Pereira-384.
    """
    r = _stub_result(
        benchmark_identifier="Pereira2018.384sentences-ridge",
        legacy_schrimpf2021_pereira243_normalized=0.2355
    )
    with pytest.raises(ResultSchemaError, match="384"):
        validate_result(r)


def test_official_metric_namespace_separation():
    """Official and custom encoding namespaces coexist without schema errors."""
    for ns in ("official_brainscore", "custom_encoding"):
        r = _stub_result(metric_namespace=ns)
        validate_result(r)  # must not raise
        assert r["metric_namespace"] == ns


def test_require_immutable_revision_rejects_branch_and_short_sha():
    for revision in (None, "", "main", "v1.0", "abc123"):
        with pytest.raises(ValueError, match="full 40-character"):
            require_immutable_revision(revision, field_name="model_revision")


def test_require_immutable_revision_normalizes_content_addressed_values():
    assert require_immutable_revision("A" * 40) == "a" * 40
    local = "local-sha256:" + "B" * 64
    assert require_immutable_revision(local) == "local-sha256:" + "b" * 64
