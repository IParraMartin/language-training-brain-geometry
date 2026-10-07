"""Typed access to the results bundle written by ``scripts/reproduction/build_results_bundle.py``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from brain_code.reproduction.config import CODE_CORPORA
from brain_code.reproduction.config import CORPORA
from brain_code.reproduction.config import N_BLOCKS

REQUIRED = (
    "layer_scores.csv",
    "official_seed42.csv",
    "geometry_multiseed.csv",
    "cka_pairs.csv",
    "pc_ablation_rsa.csv",
    "progressive_transplant.csv",
    "embedding_adaptation.csv",
    "single_block_transplant.csv",
    "factorial_milestones.csv",
    "corpus_diagnostics.csv",
    "spectrum_single_checkpoint.csv",
    "training_provenance.csv",
    "packed_splits.json",
    "benchmarks.json",
    "corpus_audit.json",
    "MANIFEST.json"
)


@dataclass(frozen=True)
class ScoreGrid:
    """Layer scores of one benchmark arranged as ``(n_models, n_blocks)`` arrays per condition.

    Attributes:
        benchmark: Benchmark key.
        trained: Corpus -> array with one row per training seed, ordered by seed.
        seeds: Corpus -> seeds in row order.
        random: Array with one row per random-weight initialization.
        random_seeds: Initialization seeds in row order.
    """

    benchmark: str
    trained: dict[str, np.ndarray]
    seeds: dict[str, list[int]]
    random: np.ndarray
    random_seeds: list[int]

    def depth_summary(self, corpus: str) -> np.ndarray:
        """Mean over all twelve blocks for each model of a corpus (or 'random')."""
        grid = self.random if corpus == "random" else self.trained[corpus]
        return grid.mean(axis=1)

    @property
    def code(self) -> dict[str, np.ndarray]:
        return {c: self.trained[c] for c in CODE_CORPORA}


class Bundle:
    """Loads and validates the bundle once; every table is read through here."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        missing = [name for name in REQUIRED if not (self.root / name).exists()]
        if missing:
            raise FileNotFoundError(f"results bundle at {self.root} lacks {missing}")
        self._cache: dict[str, pd.DataFrame] = {}

    def table(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            self._cache[name] = pd.read_csv(self.root / f"{name}.csv")
        return self._cache[name]

    def json(self, name: str) -> dict:
        return json.loads((self.root / f"{name}.json").read_text())

    def scores(self, benchmark: str) -> ScoreGrid:
        """Arranges one benchmark's per-fold re-extracted scores and checks the design is complete."""
        frame = self.table("layer_scores")
        frame = frame[frame["benchmark"] == benchmark]
        if frame.empty:
            raise KeyError(f"no layer scores for {benchmark}")
        trained, seeds = {}, {}
        for corpus in CORPORA:
            sub = frame[frame["corpus"] == corpus]
            pivot = sub.pivot(index="seed", columns="block", values="raw_r").sort_index()
            if pivot.shape != (5, N_BLOCKS) or pivot.isna().any().any():
                raise ValueError(f"{benchmark}/{corpus}: expected 5 seeds x {N_BLOCKS} blocks, got {pivot.shape}")
            trained[corpus] = pivot.to_numpy()
            seeds[corpus] = list(pivot.index)
        rand = frame[frame["corpus"] == "random"].pivot(index="seed", columns="block", values="raw_r").sort_index()
        if rand.shape != (3, N_BLOCKS):
            raise ValueError(f"{benchmark}/random: expected 3 initializations, got {rand.shape}")
        return ScoreGrid(
            benchmark=benchmark,
            trained=trained,
            seeds=seeds,
            random=rand.to_numpy(),
            random_seeds=list(rand.index)
        )
