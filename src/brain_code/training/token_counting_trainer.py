"""HuggingFace ``Trainer`` subclass with exact non-padding token accounting.

A nominal ``step * batch_size * seq_len`` count over-counts padded and truncated microbatches and cannot align
milestones across corpora with different token densities, so this trainer observes the true non-padding token
count of every processed microbatch:

* ``training_step`` runs once per microbatch, including gradient-accumulation substeps, so counting there
  captures every batch the optimizer sees.
* In distributed training the local count is all-reduced (SUM) across ranks before it updates the shared
  :class:`ExactTokenCounter`.

The counter is the single source of truth for milestone crossing; the paired ``ExactTokenMilestoneCallback``
reads ``counter.tokens_seen`` rather than the global step.
"""

from __future__ import annotations

from transformers import Trainer

from brain_code.training.exact_token_accounting import ExactTokenCounter, count_nonpadding_tokens


class TokenCountingTrainer(Trainer):
    """Trainer that feeds exact non-padding token counts to an ExactTokenCounter.

    Pass ``token_counter=`` or set ``token_counter`` after construction; every microbatch is observed in
    ``training_step``.
    """

    def __init__(
        self,
        *args,
        token_counter: ExactTokenCounter | None = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.token_counter = token_counter

    def _globally_reduced_tokens(self, inputs) -> int:
        """Local non-padding token count, all-reduced across ranks when distributed."""
        local = count_nonpadding_tokens(inputs)
        world_size = int(getattr(self.args, "world_size", 1) or 1)
        if world_size <= 1:
            return local
        import torch
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            # world_size > 1 without a process group means the global count cannot be known
            raise RuntimeError(
                "Distributed world_size>1 but torch.distributed is not initialized; "
                "cannot all-reduce the non-padding token count."
            )
        device = getattr(self.args, "device", None) or "cpu"
        tensor = torch.tensor([local], dtype=torch.int64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return int(tensor.item())

    def training_step(
        self,
        model,
        inputs,
        num_items_in_batch=None
    ):
        if self.token_counter is not None:
            reduced = self._globally_reduced_tokens(inputs)
            self.token_counter.observe_batch(inputs, globally_reduced_tokens=reduced)
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
