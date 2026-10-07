"""Exports the full covariance eigenvalue spectrum of every layer for the single-checkpoint spectrum audit.

The brain-manifold result JSONs keep only scalar summaries, so this reruns one forward pass per seed-42
model on the benchmark's own sentences and writes results/spectrum/spectra_<corpus>.npz.

Example: uv run python scripts/compute_spectra.py
"""

from __future__ import annotations

import numpy as np

from brain_code.config import ROOT
from brain_code.reproduction.config import HUB_REVISIONS

REVISIONS = HUB_REVISIONS
BENCHMARK = "Pereira2018.243sentences-ridge"
OUT_DIR = ROOT / "results" / "spectrum"


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from brain_code.experiments.data_loading import (
        extract_hidden_states_multi,
        load_neural_benchmark
    )
    from brain_code.experiments.manifold_pressure import covariance_spectrum

    data = load_neural_benchmark(BENCHMARK)
    print(f"{len(data.sentences)} stimuli from {BENCHMARK}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
        spectra = np.vstack([covariance_spectrum(reps[layer]) for layer in layers])
        np.savez_compressed(
            OUT_DIR / f"spectra_{corpus}.npz",
            spectra=spectra,
            layers=np.asarray(layers),
            revision=revision,
            benchmark=BENCHMARK,
            n_stimuli=len(data.sentences)
        )
        nonzero = int((spectra[0] > spectra[0].max() * 1e-12).sum())
        print(f"{corpus:<12} spectra {spectra.shape}  rank~{nonzero}  "
              f"lambda1={spectra[0, 0]:.3e}  lambda_last={spectra[0, -1]:.3e}")


if __name__ == "__main__":
    main()
