"""Builds the four-condition factorial corpus set for the text manipulations.

Writes ``condition_{A,B,C,D}.txt`` (one text per line), ``meta.json`` and ``pseudoword_mapping.json`` to
``--out_dir``, then runs the validation battery.

Example: brain-build-corpora --dataset tinystories --out_dir data/factorial_corpora --seed 42
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

from brain_code.config import DATASETS
from brain_code.training.corpus_transforms import (
    CONDITION_LABELS,
    CONDITION_NAMES,
    PseudowordSubstitutor,
    SentenceShuffler
)
from brain_code.training.corpus_validation import run_validation_battery, all_passed


def _load_hub_texts(dataset_key: str, n_texts: int, text_field: str | None) -> list[str]:
    """Loads natural-language texts from a HuggingFace corpus.

    The transforms (within-sentence shuffle, pseudoword substitution) are
    defined over natural English prose, so a prose split such as ``tinystories``
    is the appropriate source — the code splits have no sentence structure for
    the syntax manipulation to disrupt.

    Args:
        dataset_key: Key from ``DATASETS`` in ``src/brain_code/config.py``.
        n_texts: Maximum number of non-empty texts to take.
        text_field: Column holding the text. Auto-detected when ``None``.

    Returns:
        list[str]: Up to ``n_texts`` non-empty stripped documents.

    Raises:
        SystemExit: If the key is unknown or no usable text column is found.
    """
    from datasets import load_dataset

    if dataset_key not in DATASETS:
        raise SystemExit(
            f"Unknown --dataset {dataset_key!r}. One of: {sorted(DATASETS)}."
        )
    repo, config = DATASETS[dataset_key]

    # not streaming: an abandoned streaming iterator leaves a reader thread that aborts
    # at interpreter shutdown with exit 134 after the corpora are written, which the
    # launcher's `set -e` reads as a failed build; a bounded split slice has no reader thread
    dataset = load_dataset(repo, config, split=f"train[:{n_texts}]")

    if text_field is None:
        for candidate in ("text", "code", "content", "sentence"):
            if candidate in dataset.column_names:
                text_field = candidate
                break
        else:
            raise SystemExit(
                f"Could not find a text column in {repo}/{config}; "
                f"available: {dataset.column_names}. Pass --text_field explicitly."
            )

    # corpora are written one document per line, so a document with newlines would be
    # read back as several, changing both the validation statistics and the training data
    texts = [" ".join(t.split()) for t in dataset[text_field] if t and t.strip()]
    del dataset
    gc.collect()
    return texts


_TOY_TEXTS = [
    "The cat sat on the mat and looked around.",
    "Scientists discovered a new species of fish in the ocean.",
    "She quickly ran through the forest to find shelter.",
    "The ancient ruins revealed many secrets about the civilization.",
    "He carefully placed the fragile vase on the shelf.",
    "Birds migrate south every winter to escape the cold.",
    "The researcher published important findings about climate change.",
    "Children learn language through exposure and practice.",
    "Mountains form over millions of years through geological processes.",
    "The programmer wrote a function to parse the input data.",
    "Rain fell heavily on the empty streets during the night.",
    "They discovered that neurons communicate through chemical signals.",
    "The novel explores themes of memory and identity.",
    "Engineers designed a bridge that could withstand earthquakes.",
    "Flowers bloom in spring when temperatures rise and days lengthen.",
    "The professor explained the theory of relativity to her students.",
    "Waves crash against the rocky shore every morning.",
    "He solved the difficult equation using algebraic methods.",
    "The market opened at nine and closed at five.",
    "Stars are born from clouds of gas and dust in space."
]


def _text_corpus_hash(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build(
    texts: list[str],
    out_dir: Path,
    seed: int,
    validate: bool,
    *,
    forbidden_tokens: tuple[str, ...] = ()
) -> None:
    """Builds, writes and validates all four factorial conditions.

    Args:
        texts: Source documents (one per element, no embedded newlines).
        out_dir: Directory to write ``condition_{A,B,C,D}.txt`` and ``meta.json``.
        seed: Seed for the stochastic transforms.
        validate: Whether to run the validation battery after writing.

    Raises:
        SystemExit: If the validation battery fails.
    """
    # validation and `brain-train --dataset_path` both read these files linewise
    texts = [" ".join(t.split()) for t in texts if t and t.strip()]
    out_dir.mkdir(parents=True, exist_ok=True)
    substitutor = PseudowordSubstitutor(
        seed=seed, forbidden_tokens=forbidden_tokens
    ).fit(texts)
    condition_c = substitutor.transform_corpus(texts)
    corpora: dict[str, list[str]] = {
        "A": list(texts),
        "B": SentenceShuffler(seed=seed).transform_corpus(list(texts)),
        "C": condition_c,
        "D": SentenceShuffler(seed=seed).transform_corpus(condition_c)
    }
    substitutor.save_mapping(out_dir / "pseudoword_mapping.json")
    for cond in CONDITION_LABELS:
        transformed = corpora[cond]
        out_path = out_dir / f"condition_{cond}.txt"
        out_path.write_text("\n".join(transformed) + "\n", encoding="utf-8")
        print(f"  [{cond}] {len(transformed)} texts → {out_path}")

    meta = {
        "n_texts": len(texts),
        "seed": seed,
        "conditions": {
            condition: {
                "operation": CONDITION_NAMES[condition],
                "sha256": _text_corpus_hash(corpora[condition]),
                "n_texts": len(corpora[condition])
            }
            for condition in CONDITION_LABELS
        },
        "source_sha256": _text_corpus_hash(texts),
        "pseudoword_mapping_hash": substitutor.audit.mapping_hash,
        "pseudoword_mapping_injective": substitutor.audit.is_injective,
        "forbidden_tokens": sorted(set(forbidden_tokens))
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if validate:
        print("\nRunning validation battery...")
        results = run_validation_battery(
            texts_a=corpora["A"],
            texts_b=corpora["B"],
            texts_c=corpora["C"],
            texts_d=corpora["D"],
            verbose=True
        )
        if not all_passed(results):
            raise SystemExit(
                "Validation FAILED. Redesign the corpus transforms before training."
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Build the factorial corpus set.")
    p.add_argument(
        "--dataset-path",
        "--dataset_path",
        dest="dataset_path",
        default=None,
        help="Path to plain-text file (one text per line)."
    )
    p.add_argument(
        "--dataset",
        default=None,
        help=f"Build from a HuggingFace corpus instead of a local file. "
             f"One of: {sorted(DATASETS)}. Use a prose split (e.g. "
             f"'tinystories') — the transforms need real sentences."
    )
    p.add_argument(
        "--n-texts",
        "--n_texts",
        dest="n_texts",
        type=int,
        default=50_000,
        help="Maximum documents to take when using --dataset."
    )
    p.add_argument(
        "--text-field",
        "--text_field",
        dest="text_field",
        default=None,
        help="Text column for --dataset (auto-detected when omitted)."
    )
    p.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        default="data/factorial_corpora"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--forbidden-token",
        action="append",
        default=[],
        help="Token string that may not be emitted as a pseudoword. Repeatable."
    )
    p.add_argument(
        "--no-validate",
        "--no_validate",
        dest="no_validate",
        action="store_true",
        help="Skip validation battery (not recommended)."
    )
    p.add_argument(
        "--toy-mode",
        "--toy_mode",
        dest="toy_mode",
        action="store_true",
        help="Run on built-in toy corpus (20 sentences)."
    )
    args = p.parse_args()

    if args.toy_mode:
        texts = _TOY_TEXTS
        out_dir = Path("results/factorial/toy")
        print(f"=== TOY MODE: {len(texts)} texts ===")
    elif args.dataset_path:
        path = Path(args.dataset_path)
        texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        out_dir = Path(args.out_dir)
        print(f"=== {len(texts)} texts from {path} ===")
    elif args.dataset:
        texts = _load_hub_texts(args.dataset, args.n_texts, args.text_field)
        out_dir = Path(args.out_dir)
        print(f"=== {len(texts)} texts from HuggingFace corpus {args.dataset!r} ===")
    else:
        p.error("Provide --dataset-path, --dataset, or use --toy-mode.")

    build(
        texts,
        out_dir,
        seed=args.seed,
        validate=not args.no_validate,
        forbidden_tokens=tuple(args.forbidden_token)
    )

    if args.toy_mode:
        print("\n=== TOY MODE COMPLETE ===")


if __name__ == "__main__":
    main()
