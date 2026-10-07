"""Content-addressed cache keys.

A cache hit must be invalid when any input dependency changes. Human-readable
filenames like ``python.pkl`` are insufficient because they do not encode the
tokenizer, dataset revision, text field or preprocessing configuration, so cache
files are named by a digest of all of them.

HuggingFace model weights already have their own content-addressed cache
(``cache_dir``); these keys are for project-local caches such as corpus
frequency tables, tokenized stimuli and parser outputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def make_cache_key(*args: Any, **kwargs: Any) -> str:
    """Returns a short hex digest encoding all provided fields.

    The digest changes whenever any field value changes, so a cache file named
    by it is invalidated when any dependency changes.

    Example::

        key = make_cache_key(
            corpus_name="python",
            tokenizer_path="tokenizers/50k_50mb",
            dataset_revision="abc123",
            text_field="code"
        )
        cache_path = cache_dir / f"{key}.pkl"

    Returns:
        A 16-character lowercase hex string (64-bit prefix of SHA-256).
    """
    payload = json.dumps(
        {"args": args, "kwargs": kwargs},
        sort_keys=True,
        ensure_ascii=True,
        default=str
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:16]


def keyed_cache_path(
    cache_dir: Path,
    prefix: str,
    suffix: str = ".pkl",
    **fields: Any
) -> Path:
    """Returns a content-addressed cache path ``<prefix>_<digest><suffix>`` under ``cache_dir``.

    Args:
        cache_dir: Directory that holds the cache file.
        prefix: Human-readable filename prefix (e.g. ``"freq_table"``).
        suffix: File extension.
        **fields: Every dependency of the cached value (tokenizer path, dataset
            revision, text field, ...); the digest is derived from them.

    Returns:
        A path inside ``cache_dir``.
    """
    key = make_cache_key(**fields)
    return Path(cache_dir) / f"{prefix}_{key}{suffix}"
