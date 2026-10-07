"""Exact non-padding token accounting for the token-milestone checkpoint sweep.

A nominal ``step * batch_size * seq_len`` count is not precise enough to match milestones across corpora, so
the training loop calls :meth:`ExactTokenCounter.observe_batch` for every processed microbatch and binds saved
checkpoints to the counter value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Iterable

import numpy as np


def count_nonpadding_tokens(batch: Any) -> int:
    """Counts the ones in a batch's binary attention mask.

    Raises:
        ValueError: If the batch is not a mapping, lacks attention_mask, or the mask is not binary.
    """
    # collators return BatchEncoding, a UserDict rather than a dict subclass, so match on Mapping
    if not isinstance(batch, Mapping):
        raise ValueError(f"Every checkpoint-sweep training batch must be a mapping; got {type(batch).__name__}.")
    if "attention_mask" not in batch:
        raise ValueError("Every checkpoint-sweep training batch must contain attention_mask.")
    mask = batch["attention_mask"]
    # torch is optional here; without it the mask is counted as a numpy array
    try:
        import torch

        if isinstance(mask, torch.Tensor):
            if mask.ndim < 1:
                raise ValueError("attention_mask must have at least one dimension.")
            if not torch.all((mask == 0) | (mask == 1)):
                raise ValueError("attention_mask must be binary.")
            return int(mask.to(dtype=torch.int64).sum().item())
    except ImportError:
        pass
    array = np.asarray(mask)
    if array.ndim < 1 or not np.isin(array, [0, 1]).all():
        raise ValueError("attention_mask must be a binary array.")
    return int(array.astype(np.int64).sum())


@dataclass
class ExactTokenCounter:
    """Thread-safe cumulative non-padding token count with milestone crossing."""

    milestones: Iterable[int]
    world_size: int = 1
    _tokens: int = 0
    _triggered: set[int] = field(default_factory=set)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def __post_init__(self) -> None:
        ordered = sorted({int(value) for value in self.milestones})
        if not ordered or ordered[0] <= 0:
            raise ValueError("milestones must contain positive token counts.")
        if int(self.world_size) < 1:
            raise ValueError("world_size must be positive.")
        self.milestones = tuple(ordered)
        self.world_size = int(self.world_size)

    @property
    def tokens_seen(self) -> int:
        return self._tokens

    def observe_batch(self, batch: dict, *, globally_reduced_tokens: int | None = None) -> tuple[int, ...]:
        """Adds one microbatch to the count and returns the milestones it crosses.

        Args:
            batch: The microbatch; its attention_mask gives the local count.
            globally_reduced_tokens: The all-reduced count across ranks, required when world_size > 1.

        Returns:
            Milestones crossed for the first time by this batch.
        """
        local = count_nonpadding_tokens(batch)
        increment = int(globally_reduced_tokens) if globally_reduced_tokens is not None else local
        if self.world_size > 1 and globally_reduced_tokens is None:
            raise ValueError("Distributed token accounting requires an all-reduced global token count.")
        if increment < 0:
            raise ValueError("Token increment cannot be negative.")
        with self._lock:
            before = self._tokens
            self._tokens += increment
            crossed = tuple(
                milestone
                for milestone in self.milestones
                if before < milestone <= self._tokens and milestone not in self._triggered
            )
            self._triggered.update(crossed)
        return crossed
