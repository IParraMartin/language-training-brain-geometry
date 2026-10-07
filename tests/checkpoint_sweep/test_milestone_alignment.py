"""Layer specs and the no-default protocol of the milestone alignment entrypoint."""

from __future__ import annotations

import pytest

from brain_code.experiments.milestone_alignment import frozen_alignment, parse_layer_specs


def test_aliases_indices_and_block_names():
    assert parse_layer_specs("first, mid ,last") == ["first", "middle", "last"]
    assert parse_layer_specs("0,transformer.h.6,11") == [0, 6, 11]


@pytest.mark.parametrize("raw", ["", " , ", "first,first", "top", "transformer.h.x"])
def test_unusable_specs_raise(raw):
    with pytest.raises(ValueError):
        parse_layer_specs(raw)


def test_missing_protocol_raises_instead_of_defaulting(monkeypatch):
    monkeypatch.delenv("BRAIN_CODE_ALIGNMENT_BENCHMARK", raising=False)
    monkeypatch.delenv("BRAIN_CODE_ALIGNMENT_LAYERS", raising=False)
    with pytest.raises(RuntimeError, match="BRAIN_CODE_ALIGNMENT_BENCHMARK"):
        frozen_alignment("unused", "0" * 64)
