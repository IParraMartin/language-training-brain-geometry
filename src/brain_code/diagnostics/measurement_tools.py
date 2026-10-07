"""Tokenizer fertility and corpus token counts over the registered datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path

from tqdm import tqdm

# hf_home has to be set before datasets and transformers are imported
_cache = str(Path(__file__).resolve().parents[3] / "cache")
os.makedirs(_cache, exist_ok=True)
os.environ.setdefault("HF_HOME", _cache)

from datasets import load_dataset
from transformers import AutoTokenizer

from brain_code.config import DATASETS, PATHS


SAMPLE_BYTES = 10_000_000


def encode_without_specials(tokenizer, texts: list[str]) -> list[list[int]]:
    """Returns the token IDs of each text, without special tokens, masks or type IDs."""
    return tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False
    )["input_ids"]


def collect_sample(ds_iter, target_bytes: int) -> list[str]:
    """Collects ``code`` texts from a streaming dataset until ``target_bytes`` UTF-8 bytes are reached.

    Args:
        ds_iter: Iterable of examples with a ``code`` field.
        target_bytes: Byte budget; iteration stops once it is met.

    Returns:
        Texts whose combined UTF-8 size is at least ``target_bytes``, or the
        whole dataset if it is smaller.
    """
    buf = []
    total = 0
    for ex in ds_iter:
        t = str(ex["code"])
        buf.append(t)
        total += len(t.encode("utf-8"))
        if total >= target_bytes:
            break
    return buf


def measure_corpus(
    tokenizer,
    texts: list[str],
    batch_size: int
) -> dict:
    """Computes fertility and compression statistics of a tokenizer on a corpus.

    A lower tokens-per-byte ratio means better compression.

    Args:
        tokenizer: HuggingFace-compatible tokenizer.
        texts: The corpus as a flat list of strings.
        batch_size: Texts tokenized per call.

    Returns:
        A dict with ``n_tokens``, ``n_bytes``, ``n_chars``, ``tokens_per_byte``,
        ``tokens_per_char`` and ``bytes_per_token``.
    """
    n_tokens = 0
    n_bytes = 0
    n_chars = 0
    for i in tqdm(range(0, len(texts), batch_size), ncols=100, leave=False):
        chunk = texts[i : i + batch_size]
        ids = encode_without_specials(tokenizer, chunk)
        for text, id_list in zip(chunk, ids):
            n_tokens += len(id_list)
            n_bytes += len(text.encode("utf-8"))
            n_chars += len(text)

    return {
        "n_tokens": n_tokens,
        "n_bytes": n_bytes,
        "n_chars": n_chars,
        "tokens_per_byte": n_tokens / n_bytes if n_bytes else 0.0,
        "tokens_per_char": n_tokens / n_chars if n_chars else 0.0,
        "bytes_per_token": n_bytes / n_tokens if n_tokens else 0.0
    }


def run_tokenizer_metrics(
    tokenizer_name: str = PATHS["tokenizer"]["default"],
    batch_size: int = 4096,
    out_path: str = "fertility_results.json"
) -> dict:
    """Measures tokenizer fertility on a fixed-byte sample of every registered dataset.

    Args:
        tokenizer_name: Path or HuggingFace Hub ID of the tokenizer.
        batch_size: Texts tokenized per call.
        out_path: Output filename, written to ``results/<out_path>``.

    Returns:
        Dataset name mapped to its :func:`measure_corpus` output.
    """
    os.makedirs("results", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    results = {}
    for name, (path, config) in DATASETS.items():
        ds = load_dataset(
            path,
            config,
            split="train",
            streaming=True
        )
        sample = collect_sample(ds, SAMPLE_BYTES)

        print(f"\n--- {name} ({len(sample)} examples, ~{SAMPLE_BYTES // 1e6}mb) ---")
        stats = measure_corpus(tokenizer, sample, batch_size)
        results[name] = stats

        print(
            f"{name}: "
            f"tokens/byte={stats['tokens_per_byte']:.4f}  "
            f"tokens/char={stats['tokens_per_char']:.4f}  "
            f"bytes/token={stats['bytes_per_token']:.2f}"
        )

    with open(f"results/{out_path}", "w") as f:
        json.dump(results, f, indent=2)

    return results


def count_corpus_tokens(
    tokenizer_name: str = PATHS["tokenizer"]["default"],
    text_field: str = "code",
    batch_size: int = 4096,
    out_path: str = "token_counts.json"
) -> dict[str, int]:
    """Counts the tokens in the full training split of every registered dataset.

    Unlike :func:`run_tokenizer_metrics` this streams each dataset in full, so
    the counts are exact and usable for training-step budgets.

    Args:
        tokenizer_name: Path or HuggingFace Hub ID of the tokenizer.
        text_field: Column holding the text to tokenize.
        batch_size: Texts tokenized per call.
        out_path: Output filename, written to ``results/<out_path>``.

    Returns:
        Dataset name mapped to its exact token count.
    """
    os.makedirs("results", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    counts = {}
    for name, (path, config) in DATASETS.items():
        ds = load_dataset(
            path,
            config,
            split="train",
            streaming=True
        )
        n_tokens = 0
        batch: list[str] = []

        for ex in tqdm(
            ds,
            desc=name,
            ncols=100,
            leave=False
        ):
            batch.append(str(ex[text_field]))
            if len(batch) == batch_size:
                n_tokens += sum(len(seq) for seq in encode_without_specials(tokenizer, batch))
                batch = []

        if batch:
            n_tokens += sum(len(seq) for seq in encode_without_specials(tokenizer, batch))

        counts[name] = n_tokens
        print(f"{name}: {n_tokens:,} tokens")

    with open(f"results/{out_path}", "w") as f:
        json.dump(counts, f, indent=2)

    return counts
