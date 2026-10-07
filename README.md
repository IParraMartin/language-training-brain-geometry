# Language training shapes brain-aligned geometry

Code and data to reproduce a controlled comparison of GPT-2 small models trained on English (TinyStories)
and on four code corpora (Python, C, x86-64 assembly, and machine code written as hexadecimal text), all
with one shared tokenizer and training recipe. The models are scored against Brain-Score language
benchmarks (Pereira2018, Tuckute2024, Blank2014 and Fedorenko2016) relative to matched random-weight
models. Their representational geometry is compared with linear CKA, covariance spectra and RSA, and
block transplants and embedding-only adaptation test which learned components carry the alignment.

## Contents

```
src/brain_code/
  config.py            dataset, tokenizer, model-config and benchmark keys
  schema.py            provenance envelope required on every result file
  training/            tokenizer and GPT-2 training, packing, token-milestone checkpoints, factorial corpora
  experiments/         Brain-Score harness, per-fold scores, geometry, RSA/CKA, transplants, spectra
  diagnostics/         corpus token counts, stimulus coverage, frequency correlation, fertility
  reproduction/        seed-level statistics computed from the results bundle
scripts/               one entry point per analysis (also installed as brain-* commands)
  reproduction/        build the results bundle, recover corpus and benchmark metadata, compute results
configs/
  models/, training/   GPT-2 architecture and training recipes
  paths.yaml           locations of the results bundle and the raw archives
data/
  tokenizers/50k_50mb  the shared tokenizer every model uses
  factorial/pos_map.json  part-of-speech classes for the lexical-permutation cells
  results_bundle.zip   the per-model scores and measurements every statistic is computed from
language/              git submodule: Brain-Score language at commit 6650678
patches/               the fix applied to that submodule
docs/                  project page and documentation site (reproduction, corpora, benchmarks, code map)
tests/                 pytest suite (CPU only, no downloads)
```

## Setup

The project needs [uv](https://docs.astral.sh/uv/) and Python 3.11. The Brain-Score stack pins
`xarray<2022.6`, which fails on Python 3.12, and `numpy<2`.

```bash
git clone --recurse-submodules <repository-url>
cd language-training-brain-geometry
bash setup_env.sh            # initializes language/ if needed and runs uv sync
uv sync --extra dev          # adds pytest, ruff and concept-erasure
```

`setup_env.sh` also applies `patches/ridgecv-single-neuroid.patch` to the submodule, a fix without which
Tuckute2024 cannot be scored (see `patches/README.md`). `brainscore-language` is installed from the
`language/` submodule by `uv sync`. Do not install it with
`uv pip install -e language/`, because the next sync removes it. `requirements.txt` holds the same
pinned versions for pip users.

## Data

- **Results bundle.** `data/results_bundle.zip` contains per-model, per-block scores on every benchmark,
  multi-seed geometry and CKA, PC-ablation RSA, transplant and adaptation scores, factorial and pseudoword
  milestones, corpus diagnostics, the one-checkpoint spectrum audit, training provenance, recovered packed
  split sizes, the corpus audit and the loaded-benchmark descriptions. `MANIFEST.json` lists the sha256 of
  every file and of every raw input it was built from. Unpack it with `make data`.
- **Training corpora.** The corpora come from the Hugging Face dataset
  `IParraMartin/BrainScore-Code-English` at revision `cc83566aebbaa2eeb76104cc956e883a722b9ded`, and they
  are downloaded on first use. See `docs/corpora.md`.
- **Models.** The seed-42 models are public on the Hugging Face Hub (`IParraMartin/gpt2-<corpus>`), with
  pinned revisions in `src/brain_code/reproduction/config.py`.
- **Neural data.** The neural data are fetched by the Brain-Score loader on first use and are not
  redistributed. See `docs/benchmarks.md`.

## Running

```bash
make data          # unpack data/results_bundle.zip into data/results_bundle/
make results       # every statistic from the bundle, in seconds on a CPU
make test          # unit tests
make smoke         # CPU smoke tests of the training, scoring and analysis paths
```

The project page and documentation build with `uv run --only-group docs sphinx-build -b html docs site`
and `.github/workflows/docs.yml` deploys them to GitHub Pages when the repository variable `DEPLOY_DOCS`
is `true`.

`make results` runs `scripts/reproduction/compute_results.py`. Recomputing the bundle from checkpoints
and retraining the models are described in `docs/reproduction.md`. The main commands are:

| command | purpose |
|---|---|
| `brain-train-tokenizer` | train the shared BPE tokenizer |
| `brain-train` | train a GPT-2 model on one corpus |
| `brain-score` | score a model on a Brain-Score benchmark at every block |
| `python -m scripts.extract_layer_cis` | per-fold encoding scores of Hub checkpoints |
| `python -m scripts.score_local_checkpoints` | per-fold encoding scores of local checkpoints (seeds 43-46, random weights, adapted models) |
| `python -m scripts.make_random_init_models` | the random-weight reference models |
| `python scripts/run_latent_geometry.py` | multi-seed CKA, participation ratio and RSA |
| `python scripts/run_pc_ablation_rsa.py` | RSA after removing leading principal components |
| `python -m scripts.run_progressive_transplant` | progressive block transplants, built in memory and scored |
| `brain-patch`, `python -m scripts.single_block_paired_tests` | single-block and embedding transplants and their fold-paired tests |
| `brain-embed-ft` | embedding-only adaptation |
| `python -m scripts.dump_corpus`, `brain-build-corpora`, `python -m scripts.build_lexical_permutation` | pseudoword and factorial corpora |
| `brain-checkpoint-sweep` | training with token-milestone checkpoints |
| `brain-measure`, `brain-confounds`, `brain-correlate` | corpus token counts, coverage and frequency diagnostics |
| `brain-manifold` | one-checkpoint covariance-spectrum audit |

Every command takes `--help`. Dataset, tokenizer, model and training-recipe arguments are keys defined
in `src/brain_code/config.py`, not file paths.

## Outputs

`make results` writes three files to `results/summary/` and overwrites them on each run. The outputs are
byte-identical across runs.

- `results.json` holds every statistic. The primary contrasts are computed with the random-weight
  reference as a sampled group and as a fixed mean. It also holds block-wise tests, geometry,
  interventions, factorial and pseudoword tests, corpus diagnostics and exposure.
- `training_exposure.csv` has one row per primary training run, with the recipe, fingerprints, steps,
  evaluated checkpoint, processed tokens, pass equivalents and missing fields.
- `corpus_provenance.json` contains the corpus audit and the recovered packed-split sizes.

The analysis scripts write their outputs under `results/`, which is not tracked. Every result JSON
carries the provenance envelope from `src/brain_code/schema.py`: model revision, benchmark identifier,
`language/` commit and whether the git tree was clean.

## License

MIT, see `LICENSE`. The training corpora, checkpoints and neural data have their own terms, described
in `docs/corpora.md` and `docs/benchmarks.md`.
