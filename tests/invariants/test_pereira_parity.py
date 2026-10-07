from __future__ import annotations

import numpy as np

from brain_code.experiments.pereira_parity import compare_pereira_parity


def _bundle():
    return {
        "stimulus_ids": np.array(["a", "b"]),
        "stimulus_texts": np.array(["one", "two"]),
        "cv_group_ids": np.array(["story-a", "story-a"]),
        "context_group_ids": np.array(["passage-1", "passage-1"]),
        "contexts": np.array(["one", "one two"]),
        "token_ids_json": np.array(["[1]", "[1,2]"]),
        "attention_masks_json": np.array(["[1]", "[1,1]"]),
        "final_token_positions": np.array([0, 1]),
        "activations": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "neural_neuroid_ids": np.array(["n1"]),
        "neural_targets": np.array([[0.1], [0.2]]),
        "fold_ids": np.array([0, 1]),
        "predictions": np.array([[0.1], [0.2]]),
        "per_neuroid_correlations": np.array([0.5]),
        "raw_score": np.array(0.5)
    }


def test_pereira_parity_passes_identical_intermediate_artifacts():
    bundle = _bundle()
    report = compare_pereira_parity(bundle, {key: np.array(value, copy=True) for key, value in bundle.items()})
    assert report["validated_against_official"] is True
    assert len(report["parity_artifact_hash"]) == 64


def test_pereira_parity_fails_context_mismatch_even_when_scores_match():
    reference = _bundle()
    custom = {key: np.array(value, copy=True) for key, value in reference.items()}
    custom["contexts"][1] = "two"
    report = compare_pereira_parity(reference, custom)
    assert report["validated_against_official"] is False
    assert report["checks"]["contexts"]["passed"] is False
    assert report["checks"]["raw_score"]["passed"] is True
