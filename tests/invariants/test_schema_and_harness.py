from __future__ import annotations

import pytest

from brain_code.experiments.brain_score_harness import run_experiment
from brain_code.schema import ResultSchemaError, make_base_result, validate_result


def test_official_harness_rejects_fake_nested_cv_before_running():
    with pytest.raises(ValueError, match="nested_cv is not implemented"):
        run_experiment("org/model", "Pereira2018.243sentences-ridge", layer_selection_mode="nested_cv")


def test_valid_prespecified_result_requires_selection_evidence():
    result = make_base_result(
        experiment_id="x",
        run_id="x",
        model_identifier="model",
        benchmark_identifier="Pereira2018.243sentences-ridge",
        result_status="valid",
        conclusion_status="valid",
        layer="transformer.h.1",
        layer_selection_method="prespecified",
        completeness="complete"
    )
    with pytest.raises(ResultSchemaError, match="selection_evidence_hash"):
        validate_result(result, require_complete_provenance=False)


def test_pereira384_rejects_legacy_ceiling():
    result = make_base_result(
        experiment_id="x",
        run_id="x",
        model_identifier="model",
        benchmark_identifier="Pereira2018.384sentences-ridge",
        result_status="rerun_required",
        conclusion_status="rerun_required",
        legacy_schrimpf2021_pereira243_normalized=0.5
    )
    with pytest.raises(ResultSchemaError, match="Pereira-384"):
        validate_result(result, require_complete_provenance=False)


def test_selection_evidence_is_bound_to_model_layer_and_target(tmp_path):
    from brain_code.experiments.brain_score_harness import _validated_selection_evidence

    evidence = {
        "selection_method": "independent_development_selection",
        "model_identifier": "org/model",
        "model_revision": "abc123",
        "source_benchmark": "Pereira2018.243sentences-ridge",
        "target_benchmark": "Pereira2018.384sentences-ridge",
        "selected_layer": "transformer.h.3"
    }
    path = tmp_path / "evidence.json"
    path.write_text(__import__("json").dumps(evidence))
    digest, loaded = _validated_selection_evidence(
        path,
        model_identifier="org/model",
        model_revision="abc123",
        benchmark_identifier="Pereira2018.384sentences-ridge",
        selected_layer="transformer.h.3"
    )
    assert len(digest) == 64
    assert loaded == evidence


def test_arbitrary_selection_evidence_file_is_rejected(tmp_path):
    from brain_code.experiments.brain_score_harness import _validated_selection_evidence

    path = tmp_path / "evidence.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="does not match"):
        _validated_selection_evidence(
            path,
            model_identifier="org/model",
            model_revision="abc123",
            benchmark_identifier="Pereira2018.384sentences-ridge",
            selected_layer="transformer.h.3"
        )
