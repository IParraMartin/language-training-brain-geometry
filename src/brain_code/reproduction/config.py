"""Single source of truth for the corpus order, seeds, benchmarks, statistics and paths of the analysis."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from brain_code.config import ROOT

CORPORA = ["tinystories", "python", "source", "assembly", "binary"]
CODE_CORPORA = ["python", "source", "assembly", "binary"]
RANDOM = "random"

# primary trained replicates and random-weight references
TRAINING_SEEDS = [42, 43, 44, 45, 46]
RANDOM_SEEDS = [42, 43, 44]
N_BLOCKS = 12
LAYERS = [f"transformer.h.{i}" for i in range(N_BLOCKS)]

# benchmark key -> display name, Brain-Score identifier and result-archive files holding per-fold scores
BENCHMARKS = {
    "pereira_243": {
        "label": "Pereira-243",
        "identifier": "Pereira2018.243sentences-ridge",
        "files": [
            "official_brainscore/layer_cis.json",
            "official_brainscore/layer_cis_e1_seeds.json",
            "official_brainscore/layer_cis_randominit.json"
        ]
    },
    "pereira_384": {
        "label": "Pereira-384",
        "identifier": "Pereira2018.384sentences-ridge",
        "files": [
            "official_brainscore_384/layer_cis_384.json",
            "official_brainscore_384/layer_cis_e1_seeds.json",
            "official_brainscore_384/layer_cis_randominit.json"
        ]
    },
    "tuckute": {
        "label": "Tuckute2024",
        "identifier": "Tuckute2024-ridge",
        "files": [
            "multi_bench/tuckute/layer_cis_tuckute.json",
            "multi_bench/tuckute/layer_cis_tuckute_seeds.json"
        ]
    },
    "blank": {
        "label": "Blank2014",
        "identifier": "Blank2014-ridge",
        "files": [
            "multi_bench/blank/layer_cis_blank.json",
            "multi_bench/blank/layer_cis_blank_seeds.json"
        ]
    },
    "fedorenko": {
        "label": "Fedorenko2016",
        "identifier": "Fedorenko2016-ridge",
        "files": [
            "multi_bench/fedorenko/layer_cis_fedorenko.json",
            "multi_bench/fedorenko/layer_cis_fedorenko_seeds.json"
        ]
    }
}
# the twelve-test confirmatory family: four benchmark families x three contrasts; Pereira-384 is a
# second stimulus set of the Pereira study and enters only the depth-resolved replication
PRIMARY_BENCHMARKS = ["pereira_243", "tuckute", "blank", "fedorenko"]
CONTRASTS = ["L-R", "C-R", "L-C"]
ALPHA = 0.05

# pinned Hugging Face revisions of the seed-42 models
HUB_REVISIONS = {
    "tinystories": "56a6fd99e62aae13179944e085ae6b2370ba7d99",
    "python": "2f8cc2c9e53529348fe237b3d22ad411e6185c21",
    "source": "a8fee0121a62b4782cf9af310c920392cc06bf46",
    "assembly": "2a602bb7bec98b630e8de0cf13d9912f0af434e6",
    "binary": "2bfbabf7f74f10092bca5dea42b1025395c7d116"
}
DATASET_REPO = "IParraMartin/BrainScore-Code-English"
DATASET_REVISION = "cc83566aebbaa2eeb76104cc956e883a722b9ded"
BRAINSCORE_LANGUAGE_COMMIT = "6650678f5f28f3d71ca4a0d086f622834794eaa0"

_PATH_KEYS = (
    "results_archive",
    "campaign_artifacts",
    "seed_models",
    "results_bundle",
    "recovered"
)


@dataclass(frozen=True)
class AnalysisPaths:
    """Locations of the raw archives and of the derived results bundle.

    Attributes:
        results_archive: The exported results tree of the original runs.
        campaign_artifacts: The geometry and factorial campaign outputs.
        seed_models: Local checkpoints of seeds 43-46 and the random-weight models.
        results_bundle: The derived, released bundle that every statistic is computed from.
        recovered: Outputs of the metadata-recovery scripts (packed splits, corpus audit, benchmarks).
    """

    results_archive: Path
    campaign_artifacts: Path
    seed_models: Path
    results_bundle: Path
    recovered: Path


def load_paths(config_file: str | Path | None = None) -> AnalysisPaths:
    """Resolves analysis paths from ``configs/paths.yaml``, overridden by ``BRAIN_CODE_<KEY>`` variables.

    Relative paths are resolved against the repository root.

    Args:
        config_file: Alternative yaml file with the same keys.

    Returns:
        The resolved paths; nothing is checked for existence here.
    """
    path = Path(config_file) if config_file else ROOT / "configs" / "paths.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(raw) - set(_PATH_KEYS)
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
    resolved = {}
    for key in _PATH_KEYS:
        value = os.environ.get(f"BRAIN_CODE_{key.upper()}", raw.get(key))
        if value is None:
            raise ValueError(f"{path} does not set {key!r} and BRAIN_CODE_{key.upper()} is unset")
        candidate = Path(os.path.expanduser(str(value)))
        resolved[key] = candidate if candidate.is_absolute() else ROOT / candidate
    return AnalysisPaths(**resolved)


def require(path: Path, what: str) -> Path:
    """Returns ``path`` or fails with a message naming the missing artifact.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}. Set the matching key in configs/paths.yaml.")
    return path
