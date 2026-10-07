"""The activation cache must not serve one model's or one input set's activations to another.

A cache keyed on the environment string alone gives a trained and an untrained model of the same
architecture the same file, and two probe sets of equal length the same file; both collisions pass a
row-count check and return wrong activations silently.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from brain_code.experiments.data_loading import (  # noqa: E402
    _activation_cache_path,
    _model_fingerprint,
    extract_hidden_states
)

SETTINGS = {
    "layer": 0,
    "pooling": "last",
    "batch_size": 2,
    "device": "cpu"
}


def _tiny_model(config, seed: int):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(config)


def test_fingerprint_separates_two_seeds_of_one_architecture(micro_model_config):
    a = _tiny_model(micro_model_config, seed=0)
    b = _tiny_model(micro_model_config, seed=1)
    assert _model_fingerprint(a) != _model_fingerprint(b)


def test_fingerprint_is_stable_for_the_same_weights(micro_model_config):
    a = _tiny_model(micro_model_config, seed=0)
    assert _model_fingerprint(a) == _model_fingerprint(a)


def test_cache_path_varies_with_model_and_inputs(tmp_path):
    base = {
        "cache_dir": tmp_path,
        "cache_key": "k",
        "context_mode": "independent_sentence",
        "pooling": "last",
        "layer_index": 0
    }
    reference = _activation_cache_path(**base, model_fingerprint="mA", inputs_hash="iA")
    other_model = _activation_cache_path(**base, model_fingerprint="mB", inputs_hash="iA")
    other_inputs = _activation_cache_path(**base, model_fingerprint="mA", inputs_hash="iB")
    assert reference != other_model
    assert reference != other_inputs


def test_untrained_features_are_not_served_from_the_trained_cache(
    tmp_path,
    monkeypatch,
    tiny_tokenizer,
    micro_model_config
):
    monkeypatch.setenv("BRAIN_CODE_ACTIVATION_CACHE", str(tmp_path))
    monkeypatch.setenv("BRAIN_CODE_CACHE_KEY", "shared-key")
    sentences = ["the cat sat", "a dog ran", "birds fly high"]
    trained = _tiny_model(micro_model_config, seed=0)
    untrained = _tiny_model(micro_model_config, seed=1)
    x_trained = extract_hidden_states(
        trained,
        tiny_tokenizer,
        sentences,
        **SETTINGS
    )
    x_untrained = extract_hidden_states(
        untrained,
        tiny_tokenizer,
        sentences,
        **SETTINGS
    )
    assert x_trained.shape == x_untrained.shape
    assert not np.allclose(x_trained, x_untrained)


def test_two_probe_sets_of_equal_length_do_not_collide(
    tmp_path,
    monkeypatch,
    tiny_tokenizer,
    micro_model_config
):
    monkeypatch.setenv("BRAIN_CODE_ACTIVATION_CACHE", str(tmp_path))
    monkeypatch.setenv("BRAIN_CODE_CACHE_KEY", "shared-key")
    model = _tiny_model(micro_model_config, seed=0)
    probe_a = ["the cat sat", "a dog ran", "birds fly high"]
    probe_b = ["xyz qrs tuv", "lmn opq rst", "abc def ghi"]
    x_a = extract_hidden_states(
        model,
        tiny_tokenizer,
        probe_a,
        **SETTINGS
    )
    x_b = extract_hidden_states(
        model,
        tiny_tokenizer,
        probe_b,
        **SETTINGS
    )
    assert not np.allclose(x_a, x_b)
