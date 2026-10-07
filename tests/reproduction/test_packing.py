"""The packed-split replay counts exactly the sequences the trainer's packing produces."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("datasets")

from brain_code.training.train_model import LMTrainer
from scripts.reproduction.recover_packed_splits import MAP_BATCH, packed_blocks


class _Packer:
    """Just enough of LMTrainer to call its packing functions with a small block size."""

    max_len = 8

    class tokenizer:
        eos_token_id = 0


@pytest.mark.parametrize("eos", [False, True])
def test_replay_matches_trainer_packing(eos):
    rng = np.random.default_rng(3)
    lengths = rng.integers(0, 30, size=2 * MAP_BATCH + 37)
    docs = [list(range(1, n + 1)) for n in lengths]
    packer = _Packer()
    pack = LMTrainer.group_texts_with_eos if eos else LMTrainer.group_texts
    produced = 0
    for start in range(0, len(docs), MAP_BATCH):
        batch = docs[start:start + MAP_BATCH]
        out = pack(packer, {"input_ids": batch, "attention_mask": [[1] * len(d) for d in batch]})
        assert all(len(seq) == packer.max_len for seq in out["input_ids"])
        produced += len(out["input_ids"])
    import scripts.reproduction.recover_packed_splits as module
    original = module.CONTEXT
    module.CONTEXT = packer.max_len
    try:
        assert packed_blocks(np.asarray(lengths), eos=eos) == produced
    finally:
        module.CONTEXT = original


def test_remainders_are_dropped_per_map_batch():
    import scripts.reproduction.recover_packed_splits as module
    lengths = np.full(2 * MAP_BATCH, 1)
    whole_stream = (lengths.sum()) // module.CONTEXT
    assert packed_blocks(lengths, eos=False) == 2 * (MAP_BATCH // module.CONTEXT)
    assert packed_blocks(lengths, eos=False) <= whole_stream
