# Where each analysis lives

Each analysis lists the code that computes it and the bundle file it is summarised from. Level 1 is
computed from the results bundle, level 2 from checkpoints and raw scores, and level 3 needs
retraining (see the [reproduction guide](reproduction.md)).

**Per-fold encoding scores of every model** (level 2)
: Code: `scripts/extract_layer_cis.py` (Hub), `scripts/score_local_checkpoints.py` (local), `scripts/make_random_init_models.py`.
: Bundle data: `layer_scores.csv`.

**Primary contrasts (TinyStories, code mean, random weights; Holm over twelve tests)** (level 1)
: Code: `brain_code.reproduction.inference.primary_contrasts`, `holm`; `analysis.primary_results`.
: Bundle data: `layer_scores.csv`.

**Block profiles and block-wise tests** (level 1)
: Code: `inference.layer_tests`; `analysis.layer_results`.
: Bundle data: `layer_scores.csv`.

**CKA, participation ratio, RSA** (levels 1 and 2)
: Code: `brain_code.experiments.representational_similarity`, `scripts/run_latent_geometry.py`; `analysis.geometry_results`.
: Bundle data: `geometry_multiseed.csv`, `cka_pairs.csv`.

**PC-ablation RSA** (levels 1 and 2)
: Code: `scripts/run_pc_ablation_rsa.py`; `analysis.pc_ablation_results`.
: Bundle data: `pc_ablation_rsa.csv`.

**Progressive transplants** (levels 1 and 2)
: Code: `brain_code.experiments.progressive_transplant`, `scripts/run_progressive_transplant.py`.
: Bundle data: `progressive_transplant.csv`.

**Embedding-only adaptation** (levels 1 and 2)
: Code: `brain_code.training.module_adaptors.run_embedding_ablation`; `analysis.adaptation_results`.
: Bundle data: `embedding_adaptation.csv`.

**Single-block transplants** (levels 1 and 2)
: Code: `brain_code.experiments.causal_replacement` (`brain-patch`), `scripts/single_block_paired_tests.py`.
: Bundle data: `single_block_transplant.csv`.

**Pseudoword and factorial controls** (levels 1 and 3)
: Code: `brain_code.training.corpus_transforms`, `scripts/build_lexical_permutation.py`, `scripts/run_checkpoint_sweep.py`, `brain_code.experiments.milestone_alignment`; `analysis.factorial_results`.
: Bundle data: `factorial_milestones.csv`.

**Corpus diagnostics** (level 1)
: Code: `brain_code.diagnostics`; `analysis.corpus_results`.
: Bundle data: `corpus_diagnostics.csv`.

**Corpus provenance and exposure** (levels 1 and 2)
: Code: `scripts/reproduction/audit_corpora.py`, `scripts/reproduction/recover_packed_splits.py`; `analysis.exposure_results`.
: Bundle data: `corpus_audit.json`, `packed_splits.json`, `training_provenance.csv`.

**Loaded benchmarks** (level 2)
: Code: `scripts/reproduction/describe_benchmarks.py`.
: Bundle data: `benchmarks.json`.

**One-checkpoint spectrum audit** (levels 1 and 2)
: Code: `scripts/run_manifold_pressure.py`, `scripts/compute_spectra.py`, `scripts/compute_spectrum_uncertainty.py`.
: Bundle data: `spectrum_single_checkpoint.csv`.

## Module map of `brain_code.reproduction`

- `config` holds the corpus order, benchmark registry, seeds and pinned revisions, plus `load_paths()`
  for `configs/paths.yaml`.
- `bundle` gives typed, validated access to the bundle. `Bundle.scores(benchmark)` returns a `ScoreGrid`
  and refuses incomplete designs.
- `inference` provides the Welch-Satterthwaite contrasts, Holm correction, the exact separation test and
  the block tests.
- `analysis` computes every reported statistic as JSON-serializable dicts.

## Tests

- `tests/reproduction/` covers the contrasts, the packing replay, transplant endpoints and tying, the
  embedding-adaptation gradients, and the reported statistics (the last only when the bundle is
  unpacked).
- `tests/geometry/` covers the CKA, RSA and extraction invariants.
- The remaining directories cover the module of the same name.
