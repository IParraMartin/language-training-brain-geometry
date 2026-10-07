"""Packed fixed-length token dataset for short adaptation runs."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm


class _TokenChunkDataset(TorchDataset):
    """A PyTorch dataset that packs tokenized text into fixed-length chunks.

    Tokenizes a list of text strings in batches, concatenates all token IDs
    into a single flat sequence, and slices that sequence into non-overlapping
    chunks of exactly ``context_length`` tokens. Any remainder shorter than
    one full chunk is discarded. Each item returned by ``__getitem__`` contains
    ``input_ids`` and ``labels`` (identical) suitable for causal language model
    training.

    Attributes:
        _data (torch.Tensor): 2-D tensor of shape
            ``(n_chunks, context_length)`` containing the packed token IDs.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer,
        context_length: int,
        max_tokens: int | None = None,
        tokenize_batch: int = 512
    ) -> None:
        """Tokenizes texts and builds the fixed-length chunk tensor.

        Args:
            texts (list[str]): Raw text examples to tokenize and pack.
            tokenizer: A HuggingFace-compatible fast tokenizer. Must accept a
                list of strings and return ``'input_ids'``.
            context_length (int): Number of tokens per chunk. Sequences are
                sliced into non-overlapping blocks of this length; any
                remainder is dropped.
            max_tokens (int | None): If set, tokenization stops early once
                this many token IDs have been collected. Useful for bounding
                memory usage during short ablation runs. Defaults to ``None``
                (use all tokens).
            tokenize_batch (int): Number of text examples passed to the
                tokenizer per call, allowing the fast tokenizer to parallelize
                internally. Defaults to 512.
        """
        # stop early once enough tokens are collected to cover all training steps
        ids: list[int] = []
        for i in tqdm(range(0, len(texts), tokenize_batch), desc="tokenizing", ncols=100):
            batch = texts[i : i + tokenize_batch]
            encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
            for seq in encoded:
                ids.extend(seq)
            if max_tokens is not None and len(ids) >= max_tokens:
                break
        if max_tokens is not None:
            ids = ids[:max_tokens]
        total = (len(ids) // context_length) * context_length
        self._data = torch.tensor(ids[:total], dtype=torch.long).reshape(-1, context_length)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self._data[idx]
        return {"input_ids": chunk, "labels": chunk}
