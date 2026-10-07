"""Strict grouped outer-fold construction and immutable split manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

_GROUPING_COLS = {
    # must match the pinned official benchmark's split_coord: Pereira builds
    # context within passage_label but cross-validates by the coarser story, so
    # grouping by passage_id/passage_label would leak story categories across folds
    "Pereira2018.243sentences-ridge": "story",
    "Pereira2018.384sentences-ridge": "story",
    "Blank2014-ridge": "story",
    "Fedorenko2016-ridge": "sentence_id",
    "Tuckute2024-ridge": "run_id",
    "Futrell2018-pearsonr": "story_id"
}


def get_grouping_col(benchmark_id: str) -> str:
    """Returns the CV grouping coordinate registered for a benchmark.

    Raises:
        KeyError: If the benchmark has no registered grouping coordinate.
    """
    try:
        return _GROUPING_COLS[benchmark_id]
    except KeyError as exc:
        raise KeyError(
            f"No grouping coordinate is registered for {benchmark_id!r}. "
            "Row-wise grouping is forbidden for inferential analyses."
        ) from exc


def _validate_source(df: pd.DataFrame, group_col: str, n_folds: int) -> pd.Series:
    """Checks the frame can be split into ``n_folds`` grouped folds and returns its groups."""
    if group_col not in df.columns:
        raise ValueError(f"Missing required grouping column {group_col!r}.")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if len(df) == 0:
        raise ValueError("Cannot split an empty dataframe.")
    groups = df[group_col]
    if groups.isna().any():
        raise ValueError(f"Grouping column {group_col!r} contains missing values.")
    if groups.nunique() < n_folds:
        raise ValueError(
            f"Cannot build {n_folds} folds from {groups.nunique()} unique groups."
        )
    return groups


def build_grouped_folds(
    df: pd.DataFrame,
    group_col: str,
    n_folds: int = 5,
    seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Builds size-balanced outer folds that never split a group.

    Groups are assigned largest first to the currently smallest fold, with a
    seeded random tie-break, so fold membership is deterministic given ``seed``.

    Returns:
        One ``(train_idx, test_idx)`` pair per fold; every row is tested exactly once.
    """
    groups = _validate_source(df, group_col, n_folds)
    unique, counts = np.unique(groups.to_numpy(), return_counts=True)
    rng = np.random.default_rng(seed)
    tie_break = {group: float(rng.random()) for group in unique}
    ordered = sorted(
        zip(unique.tolist(), counts.tolist()),
        key=lambda pair: (-pair[1], tie_break[pair[0]], str(pair[0]))
    )
    bins: list[list[object]] = [[] for _ in range(n_folds)]
    sizes = np.zeros(n_folds, dtype=np.int64)
    for group, count in ordered:
        destination = int(np.argmin(sizes))
        bins[destination].append(group)
        sizes[destination] += count

    group_array = groups.to_numpy()
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    test_counts = np.zeros(len(df), dtype=np.int64)
    for held_out in bins:
        test_mask = np.isin(group_array, held_out)
        train_idx = np.flatnonzero(~test_mask)
        test_idx = np.flatnonzero(test_mask)
        if not len(train_idx) or not len(test_idx):
            raise ValueError("Grouped fold construction produced an empty split.")
        if set(group_array[train_idx]) & set(group_array[test_idx]):
            raise AssertionError("Internal error: group leakage in outer folds.")
        test_counts[test_idx] += 1
        folds.append((train_idx, test_idx))
    if not np.all(test_counts == 1):
        raise AssertionError("Every observation must occur in exactly one outer test fold.")
    return folds


def fold_manifest_hash(
    df: pd.DataFrame,
    group_col: str,
    n_folds: int,
    seed: int,
    stimulus_id_col: str = "stimulus_id"
) -> str:
    """Hashes the inputs that determine fold membership, in row order."""
    groups = _validate_source(df, group_col, n_folds)
    if stimulus_id_col in df.columns:
        stimulus_ids = [str(value) for value in df[stimulus_id_col].tolist()]
    else:
        stimulus_ids = [str(index) for index in df.index.tolist()]
    payload = {
        "n_rows": len(df),
        "group_col": group_col,
        "n_folds": int(n_folds),
        "seed": int(seed),
        "stimulus_ids_in_order": stimulus_ids,
        "group_values_in_order": [str(value) for value in groups.tolist()]
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def save_fold_manifest(
    folds: list[tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    meta: dict | None = None
) -> None:
    """Writes the fold indices and metadata to a JSON manifest."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_folds": len(folds),
        "meta": meta or {},
        "folds": [
            {"train": np.asarray(train).tolist(), "test": np.asarray(test).tolist()}
            for train, test in folds
        ]
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_fold_manifest(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reads the ``(train_idx, test_idx)`` pairs written by :func:`save_fold_manifest`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        (np.asarray(fold["train"], dtype=np.int64), np.asarray(fold["test"], dtype=np.int64))
        for fold in payload["folds"]
    ]
