"""Tests for grouped fold manifests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from brain_code.experiments.split_manifests import (
    build_grouped_folds,
    fold_manifest_hash,
    save_fold_manifest,
    load_fold_manifest,
    get_grouping_col
)


@pytest.fixture
def passage_df():
    return pd.DataFrame({
        "sentence": [f"s{i}" for i in range(30)],
        "passage_id": [f"p{i // 5}" for i in range(30)]
    })


def test_all_indices_covered(passage_df):
    folds = build_grouped_folds(passage_df, group_col="passage_id", n_folds=5)
    all_test = np.concatenate([t for _, t in folds])
    assert sorted(all_test.tolist()) == list(range(30))


def test_no_index_in_both_train_and_test(passage_df):
    folds = build_grouped_folds(passage_df, group_col="passage_id", n_folds=5)
    for train_idx, test_idx in folds:
        overlap = set(train_idx.tolist()) & set(test_idx.tolist())
        assert not overlap, f"Train/test overlap: {overlap}"


def test_groups_not_split_across_folds(passage_df):
    """Each passage must appear entirely in one fold's test set, never split."""
    folds = build_grouped_folds(passage_df, group_col="passage_id", n_folds=5)
    for fold_idx, (_, test_idx) in enumerate(folds):
        test_passages = set(passage_df.iloc[test_idx]["passage_id"].unique())
        for other_fold, (_, other_test) in enumerate(folds):
            if other_fold == fold_idx:
                continue
            other_passages = set(passage_df.iloc[other_test]["passage_id"].unique())
            split_passages = test_passages & other_passages
            assert not split_passages, (
                f"Passage(s) {split_passages} appear in both fold {fold_idx} and {other_fold}."
            )


def test_non_empty_train_and_test(passage_df):
    folds = build_grouped_folds(passage_df, group_col="passage_id", n_folds=5)
    for train_idx, test_idx in folds:
        assert len(train_idx) > 0
        assert len(test_idx) > 0


def test_raises_when_group_col_is_missing():
    """A missing group_col must raise rather than fall back to ungrouped folds."""
    df = pd.DataFrame({"sentence": [f"s{i}" for i in range(10)]})
    with pytest.raises((ValueError, TypeError)):
        build_grouped_folds(df, group_col="nonexistent_col", n_folds=5)


def test_hash_changes_with_different_seed(passage_df):
    h1 = fold_manifest_hash(
        passage_df,
        "passage_id",
        n_folds=5,
        seed=42
    )
    h2 = fold_manifest_hash(
        passage_df,
        "passage_id",
        n_folds=5,
        seed=99
    )
    assert h1 != h2


def test_hash_changes_with_different_n_folds(passage_df):
    h1 = fold_manifest_hash(
        passage_df,
        "passage_id",
        n_folds=5,
        seed=42
    )
    h2 = fold_manifest_hash(
        passage_df,
        "passage_id",
        n_folds=3,
        seed=42
    )
    assert h1 != h2


def test_save_and_load_manifest_roundtrip(passage_df, tmp_path):
    folds = build_grouped_folds(passage_df, group_col="passage_id", n_folds=5)
    out = tmp_path / "manifest.json"
    save_fold_manifest(folds, out, meta={"test": True})
    loaded = load_fold_manifest(out)
    for (tr, te), (lr, le) in zip(folds, loaded):
        np.testing.assert_array_equal(tr, lr)
        np.testing.assert_array_equal(te, le)


def test_get_grouping_col_registered():
    assert get_grouping_col("Pereira2018.243sentences-ridge") == "story"
    assert get_grouping_col("Blank2014-ridge") == "story"
    assert get_grouping_col("Fedorenko2016-ridge") == "sentence_id"


def test_get_grouping_col_unknown_raises():
    """An unknown benchmark must raise rather than fall back to a default grouping."""
    with pytest.raises(KeyError, match="UnknownBenchmark2099"):
        get_grouping_col("UnknownBenchmark2099")
