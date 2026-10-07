"""Registry of config, tokenizer, dataset and benchmark keys used across scripts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_CONFIGS = ROOT / "configs"
_TOKENIZERS = ROOT / "data" / "tokenizers"

PATHS = {
    "model_configs": {
        "micro": str(_CONFIGS / "models" / "gpt2_micro.yaml"),
        "small": str(_CONFIGS / "models" / "gpt2_small.yaml")
    },
    "train_configs": {
        "default": str(_CONFIGS / "training" / "train.yaml"),
        "python": str(_CONFIGS / "training" / "train_python.yaml"),
        "tinystories": str(_CONFIGS / "training" / "train_tinystories.yaml"),
        "assembly": str(_CONFIGS / "training" / "train_assembly.yaml"),
        "source": str(_CONFIGS / "training" / "train_source.yaml"),
        "binary": str(_CONFIGS / "training" / "train_binary.yaml"),
        "test": str(_CONFIGS / "training" / "test.yaml"),
        # five steps, no checkpointing: plumbing check for the gpu smoke run
        "smoke": str(_CONFIGS / "training" / "smoke.yaml"),
        # seeds 43-46 of every corpus: the corpus recipe without the Hub push
        "seedrep": str(_CONFIGS / "training" / "train_seedrep.yaml"),
        # pseudoword and factorial runs: seedrep with enough updates to reach the 500M milestone
        "milestones": str(_CONFIGS / "training" / "train_milestones.yaml")
    },
    "tokenizer": {
        "default": str(_TOKENIZERS / "50k_50mb")
    }
}

EXPERIMENTS = {
    "pereira_243": "Pereira2018.243sentences-ridge",
    "pereira_384": "Pereira2018.384sentences-ridge",
    "blank": "Blank2014-ridge",
    "fedorenko": "Fedorenko2016-ridge",
    "tuckute": "Tuckute2024-ridge",
    "futrell": "Futrell2018-pearsonr"
}

DATASETS = {
    "python": ("IParraMartin/BrainScore-Code-English", "python"),
    "tinystories": ("IParraMartin/BrainScore-Code-English", "tinystories"),
    "assembly": ("IParraMartin/BrainScore-Code-English", "assembly"),
    "binary": ("IParraMartin/BrainScore-Code-English", "binary"),
    "source": ("IParraMartin/BrainScore-Code-English", "source")
}
