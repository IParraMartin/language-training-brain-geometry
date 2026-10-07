"""Audits the five training corpora as stored in the public dataset revision the models were trained on.

Counts documents, exact duplicates, empty documents and duplicates shared between the training and
validation documents of the seed-43-46 split; characterizes the stored representation (character set,
comment markers, Python AST round trips); and keeps short verified excerpts. Writes one JSON.

Example: python scripts/reproduction/audit_corpora.py --out results/recovered/corpus_audit.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset

from brain_code.reproduction.config import CORPORA, DATASET_REPO, DATASET_REVISION

SAMPLE = 5000
HEX = set(string.hexdigits) | set(" \n\t")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--corpora",
        nargs="+",
        default=CORPORA,
        choices=CORPORA
    )
    return p.parse_args()


def python_checks(texts: list[str]) -> dict:
    """AST round-trip and docstring/comment indicators on a sample of Python files."""
    parsed = fixed_point = docstrings = 0
    string_chars = total_chars = 0
    for text in texts:
        total_chars += len(text)
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            continue
        parsed += 1
        fixed_point += ast.unparse(tree).strip() == text.strip()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                string_chars += len(node.value)
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(node, (
                ast.Module,
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef
            )) and body:
                first = body[0]
                value = getattr(first, "value", None)
                if isinstance(first, ast.Expr) and isinstance(value, ast.Constant) and isinstance(value.value, str):
                    docstrings += 1
                    break
    return {
        "parse_rate": parsed / len(texts),
        "ast_unparse_fixed_point_rate": fixed_point / len(texts),
        "files_with_leading_docstring_rate": docstrings / len(texts),
        "string_literal_character_share": string_chars / max(total_chars, 1)
    }


def representation_checks(corpus: str, texts: list[str]) -> dict:
    """Character-level description of what a document contains."""
    chars = Counter("".join(texts))
    total = sum(chars.values())
    record = {
        "distinct_characters": len(chars),
        "top_characters": [[c, n / total] for c, n in chars.most_common(20)],
        "hex_or_whitespace_character_share": sum(n for c, n in chars.items() if c in HEX) / total,
        "ascii_letter_share": sum(n for c, n in chars.items() if c in string.ascii_letters) / total,
        "mean_characters": float(np.mean([len(t) for t in texts]))
    }
    patterns = {
        "c_block_comment": r"/\*",
        "c_line_comment": r"(^|[^:])//",
        "hash_comment_line": r"(?m)^\s*#",
        "semicolon_comment": r";\s*[A-Za-z]",
        "english_word_run": r"\b[a-z]{4,}\s+[a-z]{3,}\s+[a-z]{3,}\b"
    }
    record["documents_matching"] = {
        name: float(np.mean([bool(re.search(p, t)) for t in texts])) for name, p in patterns.items()
    }
    if corpus == "python":
        record.update(python_checks(texts))
    return record


def audit(corpus: str) -> dict:
    ds = load_dataset(
        DATASET_REPO,
        corpus,
        split="train",
        revision=DATASET_REVISION
    )
    texts = ds["code"]
    digests = np.array([hashlib.md5(t.encode("utf-8")).hexdigest() for t in texts])
    unique, counts = np.unique(digests, return_counts=True)
    languages = Counter(ds.select(range(0, len(ds), max(1, len(ds) // 20000)))["language"])
    idx = np.asarray(ds["idx"])
    split = Dataset.from_dict({"i": np.arange(len(ds))}).train_test_split(test_size=0.1, seed=43)
    train_digests = set(digests[np.asarray(split["train"]["i"])])
    validation = digests[np.asarray(split["test"]["i"])]
    rng = np.random.default_rng(0)
    sample = [texts[int(i)] for i in rng.choice(len(texts), min(SAMPLE, len(texts)), replace=False)]
    nonempty = [t for t in sample if t.strip()]
    return {
        "n_documents": len(ds),
        "unique_documents": int(len(unique)),
        "exact_duplicate_rows": int(len(ds) - len(unique)),
        "max_copies_of_one_document": int(counts.max()),
        "empty_documents": int(sum(1 for t in texts if not t.strip())),
        "idx_unique_values": int(len(np.unique(idx))),
        "language_field": dict(languages),
        "validation_documents_seed43": int(len(validation)),
        "validation_documents_also_in_train_seed43": int(sum(d in train_digests for d in validation)),
        "representation": representation_checks(corpus, nonempty),
        "excerpts": [t[:240] for t in nonempty[:2]]
    }


def main() -> None:
    args = parse_args()
    result = json.loads(args.out.read_text()) if args.out.exists() else {}
    result.update({"dataset": DATASET_REPO, "dataset_revision": DATASET_REVISION})
    result.setdefault("corpora", {})
    for corpus in args.corpora:
        if corpus in result["corpora"]:
            continue
        result["corpora"][corpus] = audit(corpus)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        r = result["corpora"][corpus]
        print(
            corpus,
            r["n_documents"],
            r["unique_documents"],
            r["validation_documents_also_in_train_seed43"],
            flush=True
        )


if __name__ == "__main__":
    main()
