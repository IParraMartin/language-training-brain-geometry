"""BPE tokenizer training over a byte-budgeted sample of several corpora."""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

_cache = str(Path(__file__).resolve().parents[3] / "cache")
warnings.filterwarnings("ignore")


class TrainTokenizer:
    """Trains a BPE tokenizer from scratch on a multi-corpus dataset.

    Uses GPT-2 as a base architecture and trains a new vocabulary via
    HuggingFace's ``train_new_from_iterator`` API. Each corpus is streamed
    up to a configurable byte budget to keep memory usage bounded, and all
    corpora are shuffled before sampling.

    Attributes:
        datasets (list): A list of ``(path, config)`` tuples identifying each
            HuggingFace dataset to include in training.
        vocab_size (int): Target vocabulary size for the trained tokenizer.
        bytes_per_dataset (int): Maximum number of UTF-8 bytes to draw from
            each individual corpus.
        seed (int): Random seed used for corpus shuffling.
    """

    def __init__(
        self,
        datasets: list,
        vocab_size: int = 50000,
        bytes_per_dataset: int = 50_000_000,
        seed: int = 42
    ):
        """Initializes TrainTokenizer with corpus specs and training hyperparameters.

        Args:
            datasets (list): A list of ``(path, config)`` tuples, where ``path``
                is a HuggingFace dataset identifier and ``config`` is the split
                name or configuration key (e.g., ``'python'``, ``'assembly'``).
            vocab_size (int): Number of tokens in the trained vocabulary.
                Defaults to 50000.
            bytes_per_dataset (int): Byte budget per corpus. Streaming stops
                once this many UTF-8 bytes have been consumed from a single
                dataset. Defaults to 50 MB (50_000_000).
            seed (int): Random seed for dataset shuffling. Defaults to 42.
        """
        self.datasets = datasets
        self.vocab_size = vocab_size
        self.bytes_per_dataset = bytes_per_dataset
        self.seed = seed

    def train(self) -> None:
        """Trains a new BPE tokenizer and saves it to disk.

        Initializes from the GPT-2 tokenizer architecture, then trains a
        fresh vocabulary over all corpora via :meth:`_batch_iterator`. The
        resulting tokenizer is saved to a ``tokenizer/`` directory in the
        current working directory.
        """
        tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=_cache)
        new_tokenizer = tokenizer.train_new_from_iterator(self._batch_iterator(), self.vocab_size)
        new_tokenizer.save_pretrained("tokenizer")

    def _detect_text_column(self, example: dict) -> str:
        """Infers the text column name from a single dataset example.

        Checks for the column names ``'text'``, ``'code'``, and ``'assembly'``
        in order of priority. This allows the tokenizer to handle heterogeneous
        datasets that store text under different field names.

        Args:
            example (dict): A single dataset example dictionary.

        Returns:
            str: The name of the first matching text column.

        Raises:
            ValueError: If none of the expected column names are present in
                the example.
        """
        for candidate in ("text", "code", "assembly"):
            if candidate in example:
                return candidate
        raise ValueError(f"no text column found; available keys: {list(example.keys())}")

    def _stream_corpus(
        self,
        path: str,
        config: str,
        batch_size: int = 1000
    ):
        """Streams batches of text from a single dataset up to the byte budget.

        Loads the dataset in streaming mode, shuffles it with a fixed-size
        buffer, then yields lists of text strings until ``bytes_per_dataset``
        UTF-8 bytes have been consumed or the dataset is exhausted. Any
        remaining examples in a partial batch are yielded at the end.

        Args:
            path (str): HuggingFace dataset identifier (e.g.,
                ``'IParraMartin/BrainScore-Code-English'``).
            config (str): Dataset configuration or split name (e.g.,
                ``'python'``, ``'assembly'``).
            batch_size (int): Number of text strings per yielded batch.
                Defaults to 1000.

        Yields:
            list[str]: Batches of text strings from the corpus.
        """
        ds = load_dataset(
            path,
            config,
            streaming=True,
            cache_dir=_cache
        )["train"].shuffle(seed=self.seed, buffer_size=10_000)

        it = iter(ds)
        first = next(it, None)
        if first is None:
            return
        text_col = self._detect_text_column(first)

        batch = []
        total_bytes = 0
        for example in itertools.chain([first], it):
            text = example[text_col]
            batch.append(text)
            total_bytes += len(text.encode("utf-8"))

            if len(batch) == batch_size:
                yield batch
                batch = []

            if total_bytes >= self.bytes_per_dataset:
                break

        if batch:
            yield batch

    def _batch_iterator(self, batch_size: int = 1000):
        """Iterates over all corpora in sequence, yielding text batches.

        Calls :meth:`_stream_corpus` for each ``(path, config)`` tuple in
        ``self.datasets`` and chains their output into a single iterator.
        Intended to be passed directly to a HuggingFace tokenizer's
        ``train_new_from_iterator`` method.

        Args:
            batch_size (int): Number of text strings per yielded batch.
                Defaults to 1000.

        Yields:
            list[str]: Batches of text strings drawn sequentially from each
                corpus up to its byte budget.
        """
        for path, config in self.datasets:
            yield from self._stream_corpus(path, config, batch_size)
