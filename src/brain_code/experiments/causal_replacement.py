from __future__ import annotations

import os
from pathlib import Path

_cache = str(Path(__file__).resolve().parents[3] / "cache")
os.makedirs(_cache, exist_ok=True)
os.environ.setdefault("HF_HOME", _cache)

from transformers import AutoTokenizer, AutoModelForCausalLM


_RESULTS_DIR = Path("results") / "models"


def _log_weight_tie_status(model, label: str) -> bool:
    """Logs and returns whether wte and lm_head share the same storage."""
    wte = model.transformer.wte.weight
    try:
        head = model.lm_head.weight
        tied = wte.data_ptr() == head.data_ptr()
    except AttributeError:
        tied = False
        head = None
    tie_cfg = getattr(model.config, "tie_word_embeddings", None)
    print(
        f"[{label}] tie_word_embeddings config={tie_cfg}  "
        f"wte/lm_head storage identity={tied}  "
        f"wte id={id(wte)}  "
        f"{'lm_head id=' + str(id(head)) if head is not None else 'lm_head not found'}"
    )
    return tied


def patch_embeddings(
    donor_model: str = "IParraMartin/gpt2-tinystories",
    receiver_model: str = "IParraMartin/gpt2-assembly",
    include_position_embeddings: bool = True,
    sync_lm_head: bool = False
) -> AutoModelForCausalLM:
    """Replaces the token (and optionally position) embeddings of a receiver model.

    Copies the input embedding matrix ``wte`` from the donor model into the
    receiver model in-place.  By default (``sync_lm_head=False``) the output
    language-model head is NOT updated, isolating the input representation.
    Set ``sync_lm_head=True`` to also sync the head via ``tie_weights()``; in
    that case the experiment measures input embeddings AND the output head and
    must be labelled accordingly in results — NOT as "embedding-only".

    GPT-2 ties ``wte`` and ``lm_head`` by default; this function logs the
    tied/untied status before and after patching so the condition is explicit.

    Args:
        donor_model: HuggingFace Hub model ID or local path whose embeddings
            will be transplanted.
        receiver_model: HuggingFace Hub model ID or local path that receives
            the transplanted embeddings.
        include_position_embeddings: If ``True``, also copy ``wpe`` from donor.
        sync_lm_head: If ``True``, call ``tie_weights()`` after patching.
            Label this condition ``'input-embedding-plus-output-head'``.
            Defaults to ``False``.

    Returns:
        The receiver model with donor embeddings in-place.
    """
    donor = AutoModelForCausalLM.from_pretrained(donor_model)
    receiver = AutoModelForCausalLM.from_pretrained(receiver_model)

    print("Pre-patch weight tie status:")
    _log_weight_tie_status(donor, "donor")
    receiver_tied = _log_weight_tie_status(receiver, "receiver")

    if receiver_tied and not sync_lm_head:
        print(
            "WARNING: receiver wte and lm_head share storage (tied weights).  "
            "Patching wte.weight.data will inevitably update lm_head.weight.data too.  "
            "This condition modifies BOTH input embeddings and the output head.  "
            "Set sync_lm_head=True to make this explicit in your labelling."
        )

    donor_wte_shape = donor.transformer.wte.weight.shape
    recv_wte_shape = receiver.transformer.wte.weight.shape
    if donor_wte_shape != recv_wte_shape:
        raise ValueError(
            f"wte shape mismatch: donor {donor_wte_shape} != receiver {recv_wte_shape}. "
            "Both models must share the same tokenizer and hidden size."
        )
    receiver.transformer.wte.weight.data = donor.transformer.wte.weight.data.clone()
    if include_position_embeddings:
        donor_wpe_shape = donor.transformer.wpe.weight.shape
        recv_wpe_shape = receiver.transformer.wpe.weight.shape
        if donor_wpe_shape != recv_wpe_shape:
            raise ValueError(
                f"wpe shape mismatch: donor {donor_wpe_shape} != receiver {recv_wpe_shape}. "
                "Both models must have the same max_position_embeddings."
            )
        receiver.transformer.wpe.weight.data = donor.transformer.wpe.weight.data.clone()

    if sync_lm_head:
        receiver.tie_weights()
        print(
            "sync_lm_head=True: tie_weights() called.  "
            "This condition modifies BOTH input embeddings AND the output head.  "
            "Label results as 'input-embedding-plus-output-head', not 'embedding-only'."
        )
    elif receiver_tied:
        print(
            "sync_lm_head=False with tied weights: lm_head.weight was also updated "
            "because it shares storage with wte.  "
            "Label results as 'input-embedding-plus-output-head' (tied model)."
        )
    else:
        print(
            "sync_lm_head=False (untied model): lm_head left at receiver values.  "
            "This condition modifies input embeddings only."
        )

    print("Post-patch weight tie status:")
    _log_weight_tie_status(receiver, "receiver")
    return receiver


def patch_block(
    donor_model: str = "IParraMartin/gpt2-tinystories",
    receiver_model: str = "IParraMartin/gpt2-assembly",
    layer: int = 6
) -> AutoModelForCausalLM:
    """Replaces a single transformer block of a receiver model with a donor's block.

    Copies all parameters of ``transformer.h[layer]`` from the donor model
    into the corresponding block of the receiver model using
    ``load_state_dict``. This surgical swap enables causal replacement
    analysis: by substituting one block at a time we can localize which
    layer is responsible for differences in brain predictivity between two
    models trained on different corpora.

    Args:
        donor_model (str): HuggingFace Hub model ID or local path whose
            transformer block will be transplanted. Defaults to
            ``'IParraMartin/gpt2-tinystories'``.
        receiver_model (str): HuggingFace Hub model ID or local path that
            receives the transplanted block. Defaults to
            ``'IParraMartin/gpt2-assembly'``.
        layer (int): Zero-based index of the transformer block to replace.
            Defaults to 6.

    Returns:
        AutoModelForCausalLM: The receiver model with the specified donor
            block in-place. The model is not saved; call :func:`save_patched`
            to persist it.
    """
    donor = AutoModelForCausalLM.from_pretrained(donor_model)
    receiver = AutoModelForCausalLM.from_pretrained(receiver_model)

    n_donor = len(donor.transformer.h)
    n_recv = len(receiver.transformer.h)
    if layer < 0 or layer >= n_donor or layer >= n_recv:
        raise ValueError(
            f"layer={layer} is out of range: donor has {n_donor} blocks, "
            f"receiver has {n_recv} blocks (valid: 0–{min(n_donor, n_recv) - 1})."
        )
    receiver.transformer.h[layer].load_state_dict(donor.transformer.h[layer].state_dict())
    return receiver


def save_patched(model, tokenizer_source: str, out_dir: Path = None) -> None:
    """Saves a patched model and its tokenizer to disk.

    Writes both the model weights and tokenizer files to ``out_dir`` using
    the HuggingFace ``save_pretrained`` convention. The tokenizer is always
    sourced from the donor or receiver model, not inferred from ``model``
    itself, since patched models do not carry a tokenizer reference.

    Args:
        model (AutoModelForCausalLM): A patched model produced by
            :func:`patch_embeddings` or :func:`patch_block`.
        tokenizer_source (str): HuggingFace Hub model ID or local path from
            which to load the tokenizer. Typically the donor or receiver
            model ID used during patching.
        out_dir (Path, optional): Destination directory. Defaults to
            ``results/models/patched_model/``.
    """
    out_dir = Path(out_dir or (_RESULTS_DIR / "patched_model"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)

    print(f"Saved patched model to {out_dir}")
