"""Embedding-trainedness and cross-model frequency diagnostics for the shared-tokenizer confound."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

# hf_home has to be set before transformers is imported
_cache = str(Path(__file__).resolve().parents[3] / "cache")
os.makedirs(_cache, exist_ok=True)
os.environ.setdefault("HF_HOME", _cache)

import torch
from transformers import AutoModelForCausalLM


_RESULTS_DIR = Path("results") / "confounds"
_N_FREQ_BINS = 20


def compute_embedding_trainedness(
    corpus_name: str,
    model_id: str,
    freq_table: Counter,
    stimulus_token_ids: list[list[int]],
    stimulus_sentences: list[str],
    init_embeddings: Optional[str] = None,
    n_freq_bins: int = _N_FREQ_BINS,
    out_dir: Optional[Path] = None,
    hf_token: Optional[str] = None
) -> dict:
    """Computes embedding trainedness diagnostics for one trained model.

    For each vocabulary token, computes the L2 norm of its embedding and, when
    initialization weights are given, its L2 distance from initialization. Both
    are binned by training frequency (log-spaced bins): rare or
    unseen tokens should stay close to their initial values while frequently
    trained tokens move far; a flat profile would undermine the confound concern.

    Per-stimulus means over each sentence's tokens are returned for
    :func:`correlational_analysis`, and stimuli whose mean norm is at or below
    the 25th-percentile vocabulary norm are flagged as potentially confounded.

    Args:
        corpus_name: Label used in the output path.
        model_id: HuggingFace Hub model ID or local checkpoint path.
        freq_table: Token frequency table of the corpus that trained this
            model, from :func:`build_corpus_freq_table`.
        stimulus_token_ids: Token-ID sequence of each stimulus, from
            :func:`tokenize_stimuli`.
        stimulus_sentences: Raw stimulus strings, printed for flagged stimuli.
        init_embeddings: Path to a ``.npy`` initialization embedding matrix of
            shape ``(vocab_size, hidden_size)``, saved right after model
            construction and before any training; ``None`` skips
            distance-from-init.
        n_freq_bins: Number of log-spaced frequency bins.
        out_dir: Directory for the JSON; defaults to
            ``results/confounds/trainedness/<corpus_name>/``.
        hf_token: HuggingFace token for private or gated repositories.

    Returns:
        A dict with ``corpus``; ``per_bin`` (``mean_norm``, ``n_tokens`` and,
        with init embeddings, ``mean_dist_from_init`` per bin label);
        ``per_stimulus_mean_norm``; ``per_stimulus_mean_dist`` (``None``
        without init embeddings); and ``flagged_stimuli`` (stimulus indices).
    """
    out_dir = Path(out_dir or (_RESULTS_DIR / "trainedness" / corpus_name))
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        token=hf_token
    )
    model.eval()

    W: np.ndarray = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    vocab_size = W.shape[0]
    norms = np.linalg.norm(W, axis=1)  # (vocab_size,)

    dists: Optional[np.ndarray] = None
    if init_embeddings is not None:
        init_embeddings = np.load(init_embeddings)
        if init_embeddings.shape != W.shape:
            raise ValueError(
                f"init_embeddings shape {init_embeddings.shape} != current shape {W.shape}"
            )
        dists = np.linalg.norm(W - init_embeddings, axis=1)

    # log-spaced frequency bins; pseudocount +1 so zero-count tokens → bin 0
    raw_counts = np.array([freq_table.get(i, 0) for i in range(vocab_size)], dtype=np.float64)
    log_counts = np.log10(raw_counts + 1)
    max_log = log_counts.max() + 1e-9
    bin_edges = np.linspace(0.0, max_log, n_freq_bins + 1)
    # clip to [0, n_freq_bins-1] to keep all tokens inside the defined range
    bin_indices = np.clip(np.digitize(log_counts, bin_edges) - 1, 0, n_freq_bins - 1)

    per_bin: dict[str, dict] = {}
    for b in range(n_freq_bins):
        mask = bin_indices == b
        if not mask.any():
            continue
        lo = int(10 ** bin_edges[b] - 1)
        hi = int(10 ** bin_edges[b + 1] - 1)
        label = f"{lo}-{hi}"
        mean_norm = float(norms[mask].mean())
        entry: dict = {"mean_norm": round(mean_norm, 6), "n_tokens": int(mask.sum())}
        if dists is not None:
            mean_dist = float(dists[mask].mean())
            entry["mean_dist_from_init"] = round(mean_dist, 6)
        per_bin[label] = entry

    low_norm_threshold = float(np.percentile(norms, 25))
    stim_mean_norms: list[float] = []
    stim_mean_dists: Optional[list[float]] = [] if dists is not None else None
    flagged: list[int] = []

    for s_idx, seq in enumerate(stimulus_token_ids):
        valid = [tok for tok in seq if tok < vocab_size]
        if not valid:
            stim_mean_norms.append(0.0)
            if stim_mean_dists is not None:
                stim_mean_dists.append(0.0)
            continue
        mean_norm = float(np.mean([norms[tok] for tok in valid]))
        stim_mean_norms.append(mean_norm)
        if stim_mean_dists is not None:
            stim_mean_dists.append(float(np.mean([dists[tok] for tok in valid])))
        if mean_norm <= low_norm_threshold:
            flagged.append(s_idx)

    if flagged:
        print(f"\nFlagged {len(flagged)} stimuli with mean norm ≤ p25:")
        for idx in flagged[:10]:
            print(f"  [{idx}] {stimulus_sentences[idx]}")
        if len(flagged) > 10:
            print(f"  ... and {len(flagged) - 10} more.")

    out = {
        "corpus": corpus_name,
        "per_bin": per_bin,
        "per_stimulus_mean_norm": stim_mean_norms,
        "per_stimulus_mean_dist": stim_mean_dists,
        "flagged_stimuli": flagged
    }

    with open(out_dir / "trainedness.json", "w") as f:
        json.dump(out, f, indent=2)

    return out


def correlational_analysis(
    brain_scores: dict[str, dict],
    freq_tables: dict[str, Counter],
    stimulus_token_ids: list[list[int]],
    out_path: Optional[Path] = None
) -> dict:
    """Regresses each model's overall brain score on the mean stimulus training frequency.

    The frequency is that of the stimulus tokens in the model's own corpus. A
    non-significant slope means frequency does not explain cross-model score
    variance, which supports the shared-tokenizer choice.

    Args:
        brain_scores: Corpus/model name mapped to a dict with an ``overall``
            brain score.
        freq_tables: Per-corpus frequency tables from
            :func:`build_corpus_freq_table`.
        stimulus_token_ids: Token-ID sequence of each stimulus, from
            :func:`tokenize_stimuli`.
        out_path: JSON output path; defaults to
            ``results/confounds/correlational_analysis.json``.

    Returns:
        A dict whose ``across_models`` entry holds the regression summary
        (slope, intercept, r, r_squared, p_value), empty with fewer than two
        models.
    """
    out_path = Path(out_path or (_RESULTS_DIR / "correlational_analysis.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict = {"across_models": {}}

    all_toks = [tok for seq in stimulus_token_ids for tok in seq]
    model_names, mean_freqs, overall_scores = [], [], []
    for name, score_dict in brain_scores.items():
        if name not in freq_tables:
            continue
        freq = freq_tables[name]
        mean_freqs.append(float(np.mean([freq.get(tok, 0) for tok in all_toks])))
        overall_scores.append(float(score_dict["overall"]))
        model_names.append(name)

    if len(model_names) >= 2:
        log_freqs = np.log10(np.array(mean_freqs) + 1)
        centered = log_freqs - np.mean(log_freqs)
        std = np.std(log_freqs)
        scaled_freqs = centered / std if std > 0 else centered
        slope, intercept, r, p, _ = stats.linregress(scaled_freqs, overall_scores)
        interp = (
            "confound detected: frequency drives brain score variance"
            if p < 0.05
            else "non-significant: frequency does not explain brain score variance"
        )
        results["across_models"] = {
            "models": model_names,
            "mean_stimulus_freq": mean_freqs,
            "overall_brain_scores": overall_scores,
            "slope": round(float(slope), 6),
            "intercept": round(float(intercept), 6),
            "r": round(float(r), 4),
            "r_squared": round(float(r ** 2), 4),
            "p_value": round(float(p), 6)
        }
        print(
            f"\nAcross-model regression (n={len(model_names)}):\n"
            f"  slope={slope:.4f}  r2={r**2:.4f}  p={p:.4f}\n"
            f"  -> {interp}"
        )
    else:
        print("Too few models for across-model regression (need ≥2).")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved correlational analysis -> {out_path}")
    return results
