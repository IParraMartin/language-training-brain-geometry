from __future__ import annotations

from transformers import AutoModelForCausalLM


def null_positions(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """Zeroes and freezes the absolute positional embedding matrix (wpe).

    This removes the learned absolute position signal encoded in ``wpe``.
    It does NOT make the model order-invariant: the causal attention mask,
    token sequence order and prefix structure remain order-sensitive, so the
    resulting model is neither "positionally invariant" nor a bag-of-words
    model. The condition is named ``no_absolute_position_embeddings``; the
    ``--null_pos`` CLI flag selects it.

    Args:
        model: A GPT-2-style ``AutoModelForCausalLM`` with a ``transformer.wpe``
            positional embedding attribute.

    Returns:
        The model with ``wpe.weight`` zeroed and frozen.

    Raises:
        ValueError: If the model architecture does not expose ``transformer.wpe``.
    """
    try:
        model.transformer.wpe.weight.data.zero_()
        model.transformer.wpe.weight.requires_grad = False
        print(
            "\nWARNING: Absolute position embeddings (wpe) zeroed and frozen "
            "(condition: no_absolute_position_embeddings).  "
            "Token order information persists via the causal attention mask — "
            "this model is NOT positionally invariant.\n"
        )
        return model
    except AttributeError:
        raise ValueError(
            "The used model is not compatible with absolute positional embedding removal.  "
            "Expected 'model.transformer.wpe'."
        )
