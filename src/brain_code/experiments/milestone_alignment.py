"""Frozen neural-alignment entrypoint that brain-checkpoint-sweep binds to every milestone checkpoint.

``scripts/run_checkpoint_sweep.py`` calls ``frozen_alignment(checkpoint_dir, checkpoint_hash)`` at each
token milestone and stores the returned dict on the milestone record. The benchmark and layers are read
from the environment and have no defaults, so every milestone of a sweep is scored under one protocol
fixed before it starts::

    export BRAIN_CODE_ALIGNMENT_BENCHMARK=Pereira2018.243sentences-ridge
    export BRAIN_CODE_ALIGNMENT_LAYERS=first,middle,last
    brain-checkpoint-sweep --dataset-path data/factorial_corpora_500m/condition_A.txt \\
        --alignment-entrypoint brain_code.experiments.milestone_alignment:frozen_alignment ...

Layers are prespecified, not swept: a grid would select a layer by test score at every milestone.
``first``, ``middle`` and ``last`` resolve per architecture (blocks 0, 6 and 11 of GPT-2 small).
"""

from __future__ import annotations

import os
from typing import Any

_BENCHMARK_ENV = "BRAIN_CODE_ALIGNMENT_BENCHMARK"
_LAYERS_ENV = "BRAIN_CODE_ALIGNMENT_LAYERS"

_ALIAS_NORMALISATION = {
    "mid": "middle",
    "centre": "middle",
    "center": "middle"
}


def _require_env(name: str, purpose: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. The checkpoint sweep needs {purpose} fixed before it starts so every "
            "milestone is scored under one protocol; there is deliberately no default."
        )
    return value


def parse_layer_specs(raw: str) -> list[str | int]:
    """Splits a comma-separated layer list into aliases and integer block indices.

    Args:
        raw: Items such as ``first``, ``middle`` (``mid`` accepted), ``last``, an integer index or a
            ``transformer.h.N`` name.

    Returns:
        One spec per item, in order.

    Raises:
        ValueError: On an unreadable item, an empty list or a repeated layer.
    """
    specs: list[str | int] = []
    for item in (part.strip() for part in raw.split(",")):
        if not item:
            continue
        lowered = _ALIAS_NORMALISATION.get(item.lower(), item.lower())
        if lowered in {"first", "middle", "last"}:
            specs.append(lowered)
            continue
        if item.startswith("transformer.h."):
            index = item.rsplit(".", 1)[-1]
            if not index.isdigit():
                raise ValueError(f"Cannot read a layer index from {item!r}.")
            specs.append(int(index))
            continue
        try:
            specs.append(int(item))
        except ValueError as exc:
            raise ValueError(
                f"Unusable layer spec {item!r}. Use first/middle/last, an integer index, or transformer.h.N."
            ) from exc
    if not specs:
        raise ValueError(f"{_LAYERS_ENV} contained no layer specs.")
    if len(set(map(str, specs))) != len(specs):
        raise ValueError(f"{_LAYERS_ENV} repeats a layer: {raw!r}.")
    return specs


def frozen_alignment(checkpoint_dir: str, checkpoint_hash: str) -> dict[str, Any]:
    """Scores one training checkpoint at each prespecified layer through the official harness.

    Args:
        checkpoint_dir: Directory of the saved checkpoint.
        checkpoint_hash: sha256 over the checkpoint directory; it doubles as the immutable model revision,
            which the harness re-derives and checks before scoring.

    Returns:
        One entry per scored layer under ``layers``, plus the protocol fields that show two milestones
        were measured the same way.
    """
    benchmark = _require_env(_BENCHMARK_ENV, "the benchmark")
    raw_layers = _require_env(_LAYERS_ENV, "the layers")
    specs = parse_layer_specs(raw_layers)

    from brain_code.experiments.brain_score_harness import run_experiment

    revision = f"local-sha256:{checkpoint_hash}"
    scored: dict[str, dict[str, Any]] = {}
    for spec in specs:
        # the milestone record is the result, so no per-layer result file is written
        layer_results = run_experiment(
            hf_model_id=checkpoint_dir,
            benchmark_identifier=benchmark,
            save_results=False,
            layer_selection_mode="prespecified",
            prespecified_layer=spec,
            model_revision=revision,
            tokenizer_revision=revision
        )
        if len(layer_results) != 1:
            raise RuntimeError(f"Expected exactly one scored layer for {spec!r}, got {sorted(layer_results)}.")
        layer_name, metrics = next(iter(layer_results.items()))
        if layer_name in scored:
            raise ValueError(
                f"Layer specs {raw_layers!r} resolved {spec!r} onto {layer_name!r}, which was already scored."
            )
        scored[layer_name] = dict(metrics, requested_as=str(spec))

    return {
        "checkpoint_hash": checkpoint_hash,
        "benchmark_identifier": benchmark,
        "layer_selection_mode": "prespecified",
        "requested_layers": [str(spec) for spec in specs],
        "n_layers_scored": len(scored),
        "layers": scored
    }
