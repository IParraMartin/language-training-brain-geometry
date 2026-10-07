"""Progressive transplants of a donor GPT-2's leading blocks, and optionally its embeddings, into a receiver.

Arm ``k`` copies the donor's blocks ``0..k-1`` into a copy of the receiver. The embedding arm also
copies the token and position embeddings and re-ties the output head to the new token embedding. The
final layer norm is copied only when every block is transplanted, so the full arm with embeddings is
exactly the donor and ``k = 0`` without embeddings is exactly the receiver.
"""

from __future__ import annotations

import copy
import hashlib

import torch
from torch import nn


def _check_compatible(receiver: nn.Module, donor: nn.Module) -> None:
    """Requires identical GPT-2 shapes, since copied tensors must line up parameter by parameter."""
    for (name_r, p_r), (name_d, p_d) in zip(receiver.named_parameters(), donor.named_parameters()):
        if name_r != name_d or p_r.shape != p_d.shape:
            raise ValueError(f"incompatible models at {name_r} {tuple(p_r.shape)} vs {name_d} {tuple(p_d.shape)}")


@torch.no_grad()
def transplant(
    receiver: nn.Module,
    donor: nn.Module,
    k: int,
    with_embeddings: bool
) -> nn.Module:
    """Returns a new model: the receiver with the donor's first ``k`` blocks (and embeddings if asked).

    Args:
        receiver: GPT-2 causal LM whose parts are kept unless replaced.
        donor: GPT-2 causal LM with identical shapes supplying the transplanted parts.
        k: Number of leading transformer blocks to copy, between 0 and the number of blocks.
        with_embeddings: Also copy the token and position embeddings and re-tie the output head.

    Returns:
        A deep copy of the receiver with the transplanted tensors; the inputs are not modified.

    Raises:
        ValueError: If the models differ in shape or ``k`` is out of range.
    """
    n_layer = len(receiver.transformer.h)
    if not 0 <= k <= n_layer:
        raise ValueError(f"k must lie in [0, {n_layer}], got {k}")
    _check_compatible(receiver, donor)
    model = copy.deepcopy(receiver)
    for i in range(k):
        model.transformer.h[i].load_state_dict(donor.transformer.h[i].state_dict())
    if k == n_layer:
        model.transformer.ln_f.load_state_dict(donor.transformer.ln_f.state_dict())
    if with_embeddings:
        model.transformer.wte.weight.copy_(donor.transformer.wte.weight)
        model.transformer.wpe.weight.copy_(donor.transformer.wpe.weight)
        model.tie_weights()
    return model.eval()


def state_dict_hash(model: nn.Module) -> str:
    """The parameter hash recorded with the transplant results: sha256 over sorted state-dict names and bytes.

    It includes buffers and the tied output head and omits dtype and shape, unlike
    ``brain_code.training.reproducibility.parameter_hash``; the two are not interchangeable.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
