"""Correctness tests for the geometry analysis's last-token activation extraction.

The whole geometry analysis rests on one indexing decision, and it is the kind
that fails silently: with right padding the final real token of a short sentence
is NOT at position -1, it is at ``attention_mask.sum() - 1``. Reading -1 would
return a pad state for every sentence shorter than its batch maximum -- most of
them -- and the resulting "geometry" would partly describe padding. Nothing
downstream would raise; the numbers would simply be wrong.

The decisive test is therefore batch-invariance: extracting with padding must
equal extracting each sentence alone, where no padding exists at all.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.run_latent_geometry import discover_models, extract_last_token_states


@pytest.fixture(scope="module")
def micro_checkpoint(tmp_path_factory):
    """A saved 2-layer GPT-2 with a whitespace tokenizer over single letters."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

    vocab = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
    vocab["[UNK]"], vocab["[EOS]"] = 26, 27
    backend = Tokenizer(BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="[EOS]",
        pad_token="[EOS]"
    )

    config = AutoConfig.for_model(
        "gpt2",
        vocab_size=len(vocab),
        n_layer=2,
        n_head=2,
        n_embd=64,
        n_positions=128,
        n_ctx=128,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).eval()

    path = tmp_path_factory.mktemp("micro_ckpt")
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return path


# Deliberately ragged: extraction is only trustworthy if unequal lengths agree
# with the unpadded reference.
SENTENCES = [
    "a b c d e f g h",
    "b c",
    "c d e",
    "d",
    "e f g h i j k l m n"
]


def _extract(path, sentences, batch_size):
    return extract_last_token_states(
        str(path),
        sentences,
        device="cpu",
        batch_size=batch_size,
        max_length=128
    )


class TestLastTokenExtraction:
    def test_shape_and_block_names(self, micro_checkpoint):
        out = _extract(micro_checkpoint, SENTENCES, 8)
        assert sorted(out) == ["transformer.h.0", "transformer.h.1"]
        for value in out.values():
            assert value.shape == (len(SENTENCES), 64)

    def test_padding_does_not_change_the_result(self, micro_checkpoint):
        """The load-bearing test: batched-with-padding == one-at-a-time.

        Batch size 1 never pads, so it is ground truth. If the last-token index
        were taken as -1, the ragged sentences would pick up pad states here and
        this would fail.
        """
        batched = _extract(micro_checkpoint, SENTENCES, len(SENTENCES))
        unpadded = _extract(micro_checkpoint, SENTENCES, 1)
        for block in batched:
            np.testing.assert_allclose(
                batched[block],
                unpadded[block],
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"{block} differs under padding"
            )

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 64])
    def test_invariant_across_batch_sizes(self, micro_checkpoint, batch_size):
        reference = _extract(micro_checkpoint, SENTENCES, 1)
        got = _extract(micro_checkpoint, SENTENCES, batch_size)
        for block in reference:
            np.testing.assert_allclose(
                got[block],
                reference[block],
                rtol=1e-4,
                atol=1e-5
            )

    def test_naive_last_position_would_differ(self, micro_checkpoint):
        """Proves the test above has teeth.

        Reproduces the wrong indexing (position -1 under right padding) and
        asserts it disagrees with the correct extraction. If both agreed, the
        padding test would be vacuous.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(micro_checkpoint))
        tok.padding_side = "right"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(str(micro_checkpoint)).eval()
        enc = tok(SENTENCES, return_tensors="pt", padding=True)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states[1]
        naive = hs[:, -1].numpy()

        correct = _extract(micro_checkpoint, SENTENCES, len(SENTENCES))["transformer.h.0"]
        assert not np.allclose(
            naive,
            correct,
            rtol=1e-3,
            atol=1e-4
        )

    def test_single_sentence_batch_is_exact(self, micro_checkpoint):
        out = _extract(micro_checkpoint, ["a b c"], 1)
        assert out["transformer.h.0"].shape == (1, 64)
        assert np.isfinite(out["transformer.h.0"]).all()


class TestDiscoverModels:
    def test_finds_checkpoint_directories(self, micro_checkpoint, tmp_path):
        root = tmp_path / "models"
        (root / "m1").mkdir(parents=True)
        (root / "m1" / "config.json").write_text("{}")
        (root / "not_a_model").mkdir()
        assert [n for n, _ in discover_models(root, [], None)] == ["m1"]

    def test_hub_models_are_appended(self, tmp_path):
        root = tmp_path / "models"
        (root / "m1").mkdir(parents=True)
        (root / "m1" / "config.json").write_text("{}")
        names = [n for n, _ in discover_models(root, ["ts=Org/gpt2-ts"], None)]
        assert names == ["m1", "ts"]

    def test_malformed_hub_spec_raises(self, tmp_path):
        root = tmp_path / "models"
        (root / "m1").mkdir(parents=True)
        (root / "m1" / "config.json").write_text("{}")
        with pytest.raises(SystemExit, match="NAME=HUB_ID"):
            discover_models(root, ["no-equals-sign"], None)

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(SystemExit, match="no checkpoints found"):
            discover_models(tmp_path, [], None)

    def test_limit_truncates(self, tmp_path):
        root = tmp_path / "models"
        for name in ("a", "b", "c"):
            (root / name).mkdir(parents=True)
            (root / name / "config.json").write_text("{}")
        assert len(discover_models(root, [], 2)) == 2


class TestPassageContexts:
    """The benchmark represents stimulus k by the passage up to k, not by k alone."""

    @staticmethod
    def _frame(rows):
        import pandas as pd
        return pd.DataFrame(rows, columns=["stimulus", "passage_label"])

    def test_context_accumulates_within_a_passage(self):
        from scripts.run_latent_geometry import build_passage_contexts

        frame = self._frame([
            ("Bees fly.", "bees"),
            ("They make honey.", "bees"),
            ("Honey is sweet.", "bees")
        ])
        assert build_passage_contexts(frame) == [
            "Bees fly.",
            "Bees fly. They make honey.",
            "Bees fly. They make honey. Honey is sweet."
        ]

    def test_context_does_not_leak_across_passages(self):
        """Each passage restarts. Leakage would put another topic in the context."""
        from scripts.run_latent_geometry import build_passage_contexts

        frame = self._frame([
            ("Bees fly.", "bees"),
            ("They make honey.", "bees"),
            ("Rome fell.", "rome"),
            ("It was large.", "rome")
        ])
        contexts = build_passage_contexts(frame)
        assert contexts[2] == "Rome fell."
        assert "Bees" not in contexts[3]

    def test_interleaved_passages_keep_assembly_order(self):
        from scripts.run_latent_geometry import build_passage_contexts

        frame = self._frame([
            ("A one.", "a"),
            ("B one.", "b"),
            ("A two.", "a"),
            ("B two.", "b")
        ])
        assert build_passage_contexts(frame) == [
            "A one.",
            "B one.",
            "A one. A two.",
            "B one. B two."
        ]

    def test_matches_brainscore_prepare_context_exactly(self):
        """Reuse, not reimplementation: identical to the benchmark's own joiner."""
        from brainscore_language.model_helpers.preprocessing import prepare_context
        from scripts.run_latent_geometry import build_passage_contexts

        parts = ["First bit .", "Second bit ?", "Third bit ,"]
        frame = self._frame([(p, "only") for p in parts])
        contexts = build_passage_contexts(frame)
        for k in range(len(parts)):
            assert contexts[k] == prepare_context(parts[:k + 1])

    def test_punctuation_spacing_artifact_is_removed(self):
        """prepare_context strips the space the join introduces before punctuation."""
        from scripts.run_latent_geometry import build_passage_contexts

        frame = self._frame([("Hello", "p"), (", world.", "p")])
        assert build_passage_contexts(frame)[1] == "Hello, world."

    def test_length_and_order_preserved(self):
        from scripts.run_latent_geometry import build_passage_contexts

        frame = self._frame([(f"S{i}.", f"p{i % 3}") for i in range(30)])
        contexts = build_passage_contexts(frame)
        assert len(contexts) == 30
        assert all(isinstance(c, str) and c for c in contexts)
        # Stimulus i must be the final part of its own context.
        for i, context in enumerate(contexts):
            assert context.endswith(f"S{i}.")
