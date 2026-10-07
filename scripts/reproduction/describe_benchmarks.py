"""Describes the neural benchmarks exactly as the pinned Brain-Score loader presents them to the models.

Loads each benchmark through the `language/` submodule and records the data identifier and sha1, the
number of response units, stimuli, groups and subjects, the cross-validation scheme and the ceiling.
For benchmarks whose cumulative contexts can exceed the 1,024-token window, it also counts truncated
contexts and identical model inputs under right and left truncation. Writes one JSON; no model is scored.

Example: python scripts/reproduction/describe_benchmarks.py --out results/recovered/benchmarks.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

# benchmark identifier, data identifier, Pereira experiment, context group and the cross-validation the
# pinned benchmark factory passes to its metric (language/brainscore_language/benchmarks/*/benchmark.py)
BENCHMARKS = {
    "pereira_243": {
        "identifier": "Pereira2018.243sentences-ridge",
        "data": "Pereira2018.language",
        "experiment": "243sentences",
        "context_group": "passage_label",
        "cross_validation": "GroupKFold, 10 splits over story, shuffle, random_state=1234"
    },
    "pereira_384": {
        "identifier": "Pereira2018.384sentences-ridge",
        "data": "Pereira2018.language",
        "experiment": "384sentences",
        "context_group": "passage_label",
        "cross_validation": "GroupKFold, 10 splits over story, shuffle, random_state=1234"
    },
    "tuckute": {
        "identifier": "Tuckute2024-ridge",
        "data": "Tuckute2024.language",
        "experiment": None,
        "context_group": "stimulus_id",
        "cross_validation": "metric default: ShuffleSplit, 10 splits, train_size 0.9, random_state=1"
    },
    "blank": {
        "identifier": "Blank2014-ridge",
        "data": "Blank2014.fROI",
        "experiment": None,
        "context_group": "story",
        "cross_validation": "GroupKFold, 8 splits over story"
    },
    "fedorenko": {
        "identifier": "Fedorenko2016-ridge",
        "data": "Fedorenko2016.language",
        "experiment": None,
        "context_group": "sentence_id",
        "cross_validation": "GroupKFold over sentence_id"
    }
}
CONTEXT = 1024


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokenizer", default="data/tokenizers/50k_50mb")
    p.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(BENCHMARKS),
        choices=list(BENCHMARKS)
    )
    return p.parse_args()


def coord_values(assembly, name: str) -> np.ndarray | None:
    """Returns a coordinate's values, or None when the assembly lacks it."""
    try:
        return np.asarray(assembly[name].values)
    except KeyError:
        return None


def context_diagnostic(texts: np.ndarray, groups: np.ndarray, tokenizer_dir: str) -> dict:
    """Counts long cumulative contexts and the inputs they collapse to under each truncation side.

    The benchmark feeds stimulus k of a group as the space-joined text of stimuli 0..k, so contexts grow
    within a group. Under right truncation a context longer than the window keeps its first 1,024 tokens
    and stops changing; under left truncation it keeps the most recent 1,024 tokens.
    """
    from transformers import AutoTokenizer

    from brainscore_language.model_helpers.preprocessing import prepare_context

    order = sorted(set(groups))
    contexts = []
    for group in order:
        parts = list(texts[groups == group])
        contexts.extend(prepare_context(parts[:k + 1]) for k in range(len(parts)))
    lengths = []
    record = {"n_contexts": len(contexts)}
    for side in ("right", "left"):
        tok = AutoTokenizer.from_pretrained(tokenizer_dir, truncation_side=side)
        full = [tok(c)["input_ids"] for c in contexts] if side == "right" else None
        if full is not None:
            lengths = [len(ids) for ids in full]
        ids = [tuple(tok(c, truncation=True, max_length=CONTEXT)["input_ids"]) for c in contexts]
        seen: set = set()
        repeats = 0
        for x in ids:
            repeats += x in seen
            seen.add(x)
        record[f"identical_to_earlier_input_{side}_truncation"] = repeats
        record[f"distinct_inputs_{side}_truncation"] = len(seen)
    record["contexts_over_window"] = int(np.sum(np.asarray(lengths) > CONTEXT))
    record["max_context_tokens"] = int(max(lengths))
    return record


def describe(key: str, tokenizer_dir: str) -> dict:
    """Loads one benchmark's data assembly, selected as the benchmark selects it, and summarizes it."""
    from brainscore_language import load_dataset

    spec = BENCHMARKS[key]
    data = load_dataset(spec["data"])
    if spec["experiment"] is not None:
        data = data.sel(experiment=spec["experiment"]).dropna("neuroid")
    record = {
        "benchmark_identifier": spec["identifier"],
        "data_identifier": spec["data"],
        "cross_validation": spec["cross_validation"],
        "dims": {d: int(s) for d, s in zip(data.dims, data.shape)},
        "n_presentations": int(data.sizes.get("presentation", 0)),
        "n_neuroids": int(data.sizes.get("neuroid", 0)),
        "n_unique_stimulus_ids": int(len(np.unique(coord_values(data, "stimulus_id")))),
        "context_group": spec["context_group"]
    }
    groups = coord_values(data, spec["context_group"])
    if groups is not None:
        record["n_context_groups"] = int(len(np.unique(groups)))
    stories = coord_values(data, "story")
    if stories is not None:
        record["n_stories"] = int(len(np.unique(stories)))
    for subject_coord in ("subject", "subject_UID", "subject_id"):
        subjects = coord_values(data, subject_coord)
        if subjects is not None:
            record["subject_coord"] = subject_coord
            record["n_subjects"] = int(len(np.unique(subjects)))
            break
    texts = coord_values(data, "stimulus")
    if texts is not None and groups is not None:
        record["context"] = context_diagnostic(texts, groups, tokenizer_dir)
    return record


def main() -> None:
    args = parse_args()
    commit = subprocess.run(
        ["git", "-C", "language", "rev-parse", "HEAD"],
        capture_output=True,
        text=True
    ).stdout.strip()
    result = json.loads(args.out.read_text()) if args.out.exists() else {}
    result["brainscore_language_commit"] = commit
    result.setdefault("benchmarks", {})
    for key in args.benchmarks:
        if key in result["benchmarks"]:
            print(f"{key}: already described, skipping")
            continue
        result["benchmarks"][key] = describe(key, args.tokenizer)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(key, json.dumps(result["benchmarks"][key])[:400], flush=True)


if __name__ == "__main__":
    main()
