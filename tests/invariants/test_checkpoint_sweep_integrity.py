from __future__ import annotations

import numpy as np
import pytest
import torch

from brain_code.training.exact_token_accounting import ExactTokenCounter, count_nonpadding_tokens
from brain_code.training.reproducibility import parameter_hash, seed_everything


def test_actual_nonpadding_tokens_are_counted():
    mask = np.array([[1, 1, 0], [1, 0, 0]])
    assert count_nonpadding_tokens({"attention_mask": mask}) == 3
    counter = ExactTokenCounter([2, 5])
    assert counter.observe_batch({"attention_mask": mask}) == (2,)
    assert counter.observe_batch({"attention_mask": np.array([[1, 1]])}) == (5,)
    assert counter.tokens_seen == 5


def test_distributed_count_requires_all_reduce():
    counter = ExactTokenCounter([2], world_size=2)
    with pytest.raises(ValueError, match="all-reduced"):
        counter.observe_batch({"attention_mask": np.ones((1, 2))})


def test_seed_before_initialization_controls_parameter_hash():
    seed_everything(11)
    first = torch.nn.Linear(5, 3)
    seed_everything(11)
    second = torch.nn.Linear(5, 3)
    seed_everything(12)
    third = torch.nn.Linear(5, 3)
    assert parameter_hash(first) == parameter_hash(second)
    assert parameter_hash(first) != parameter_hash(third)
