from __future__ import annotations

from brain_code.training.corpus_transforms import CONDITION_NAMES, PseudowordSubstitutor


def test_pseudoword_mapping_is_injective_and_avoids_source_vocabulary(tmp_path):
    texts = ["Brave dragons collect crystals.", "Quiet wizards collect lanterns."]
    substitutor = PseudowordSubstitutor(seed=9).fit(texts)
    mapping = substitutor.mapping
    assert len(mapping) == len(set(mapping.values()))
    source = {word.lower().strip(".") for text in texts for word in text.split()}
    assert not (set(mapping.values()) & source)
    assert substitutor.audit.is_injective
    path = tmp_path / "mapping.json"
    substitutor.save_mapping(path)
    assert path.is_file()


def test_condition_names_do_not_claim_syntax_or_semantics_removed():
    joined = " ".join(CONDITION_NAMES.values()).lower()
    assert "syntax_removed" not in joined
    assert "semantics_removed" not in joined
