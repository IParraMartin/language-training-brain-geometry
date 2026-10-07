"""Exports per-fold alignment and leave-one-story-out jackknifed geometry for the spectrum audit.

Per-fold scores let corpora be compared paired within fold, which removes fold-to-fold difficulty; the
resampling unit is the story because sentences within a story are correlated. Writes
results/spectrum/uncertainty_<corpus>.npz.

Example: uv run python scripts/compute_spectrum_uncertainty.py
"""

from __future__ import annotations

import numpy as np

from brain_code.config import ROOT
from brain_code.reproduction.config import HUB_REVISIONS

REVISIONS = HUB_REVISIONS
BENCHMARK = "Pereira2018.243sentences-ridge"
OUT = ROOT / "results" / "spectrum"
SEED = 42
ALPHAS = np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from brain_code.experiments.data_loading import (
        extract_hidden_states_multi,
        load_neural_benchmark
    )
    from brain_code.experiments.manifold_pressure import (
        effective_rank,
        participation_ratio,
        spectral_decay
    )
    from brain_code.experiments.subspace_erasure import _folds_from_groups
    from brain_code.experiments.variance_partitioning import eval_encoding_model_oof

    data = load_neural_benchmark(BENCHMARK)
    folds = _folds_from_groups(data.groups, 5, SEED)
    stories = np.asarray([str(g) for g in data.groups])
    unique_stories = np.unique(stories)
    story_rows = {s: np.where(stories == s)[0] for s in unique_stories}
    print(f"{len(data.sentences)} stimuli, {len(unique_stories)} stories, {len(folds)} folds")

    # leave-one-story-out index sets, identical for every corpus and layer so that
    # differences between corpora are not differences in resampling. a bootstrap over
    # stories would duplicate rows, collapse the covariance rank and push the tail
    # eigenvalues to ~1e-38, so the power-law fit would measure that cliff instead
    plans = [
        np.concatenate([story_rows[s] for s in unique_stories if s != dropped])
        for dropped in unique_stories
    ]
    n_jack = len(plans)
    print(f"jackknife: {n_jack} leave-one-story-out replicates")

    OUT.mkdir(parents=True, exist_ok=True)
    for corpus, revision in REVISIONS.items():
        model_id = f"IParraMartin/gpt2-{corpus}"
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
        multi = extract_hidden_states_multi(
            model,
            tokenizer,
            data.sentences,
            layers=("all",),
            poolings=("last",),
            batch_size=16,
            device="cuda",
            groups=data.context_groups,
            context_mode="independent_sentence"
        )
        reps = {layer: np.asarray(a, dtype=np.float64) for (layer, _), a in multi.items()}
        layers = sorted(reps)

        metrics = ("spectral_decay", "participation_ratio", "effective_rank")
        fold_r = np.zeros((len(layers), len(folds)))
        jack = {k: np.zeros((len(layers), n_jack)) for k in metrics}
        full = {k: np.zeros(len(layers)) for k in metrics}

        for li, layer in enumerate(layers):
            X = reps[layer]
            result = eval_encoding_model_oof(
                X,
                data.responses,
                folds,
                alpha_grid=ALPHAS,
                n_inner_folds=4,
                groups=data.groups,
                alpha_strategy="per_target"
            )
            fold_r[li] = np.asarray(result["fold_r_means"], dtype=np.float64)

            full["spectral_decay"][li] = spectral_decay(X)
            full["participation_ratio"][li] = participation_ratio(X)
            full["effective_rank"][li] = effective_rank(X - X.mean(0))
            for ji, rows in enumerate(plans):
                Xj = X[rows]
                jack["spectral_decay"][li, ji] = spectral_decay(Xj)
                jack["participation_ratio"][li, ji] = participation_ratio(Xj)
                jack["effective_rank"][li, ji] = effective_rank(Xj - Xj.mean(0))

        # Jackknife standard error: sqrt((n-1)/n * sum (theta_i - theta_bar)^2).
        se = {
            k: np.sqrt((n_jack - 1) / n_jack * ((v - v.mean(axis=1, keepdims=True)) ** 2).sum(axis=1))
            for k, v in jack.items()
        }
        np.savez_compressed(
            OUT / f"uncertainty_{corpus}.npz",
            layers=np.asarray(layers),
            fold_r=fold_r,
            n_jack=n_jack,
            seed=SEED,
            **{f"jack_{k}": v for k, v in jack.items()},
            **{f"full_{k}": v for k, v in full.items()},
            **{f"se_{k}": v for k, v in se.items()}
        )
        mean_r = fold_r.mean(axis=1)
        sem_r = fold_r.std(axis=1, ddof=1) / np.sqrt(fold_r.shape[1])
        peak = int(np.argmax(mean_r))
        print(
            f"{corpus:<12} peak r={mean_r[peak]:.4f} +/- {sem_r[peak]:.4f} (SEM, 5 folds) "
            f"@L{peak}   alpha(L0)={full['spectral_decay'][0]:.3f}"
            f" +/- {se['spectral_decay'][0]:.3f} (jackknife SE)",
            flush=True
        )


if __name__ == "__main__":
    main()
