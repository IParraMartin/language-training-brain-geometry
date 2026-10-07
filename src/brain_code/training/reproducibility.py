"""Reproducibility primitives that must run before model construction."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


@dataclass(frozen=True)
class SeedState:
    """Seeding settings applied by seed_everything."""

    seed: int
    deterministic: bool
    python_hash_seed: str
    cuda_available: bool


def seed_everything(seed: int, *, deterministic: bool = False) -> SeedState:
    """Seeds Python, NumPy and PyTorch; call before constructing the model."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    cuda_available = False
    # torch is optional so the module stays importable without it
    try:
        import torch

        torch.manual_seed(seed)
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
    return SeedState(
        seed,
        deterministic,
        str(seed),
        cuda_available
    )


def parameter_hash(model: Any) -> str:
    """Hashes named model parameters in a deterministic order."""
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda pair: pair[0]):
        value = parameter.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def file_or_mapping(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Returns a mapping as a dict, or loads one from a YAML file.

    Raises:
        FileNotFoundError: If a path is given and the file does not exist.
        ValueError: If the file does not contain a mapping.
    """
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return dict(loaded)


def override_training_seed(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Returns a copy with all supported Trainer seed fields overridden."""
    result = dict(config)
    result["seed"] = int(seed)
    result["data_seed"] = int(seed)
    return result
