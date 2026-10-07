# Reproduction

The results can be reproduced at three levels. All paths come from `configs/paths.yaml`. They are
relative to the repository root, and each can be overridden with an environment variable
`BRAIN_CODE_<KEY>`. Scripts that read raw archives stop with the missing path rather than substituting
other data.

| Key | Contents | Needed for |
|---|---|---|
| `results_bundle` | tidy per-model tables unpacked from `data/results_bundle.zip`; `MANIFEST.json` lists sha256 of every input and output | level 1 |
| `results_archive` | exported results tree of the original runs (`official_brainscore/`, `multi_bench/`, `experiments/`, `confounds/`, `blimp/`), not distributed | level 2 |
| `campaign_artifacts` | geometry, PC-ablation and factorial outputs of the original analysis runs, not distributed | level 2 |
| `seed_models` | checkpoints of seeds 43-46 and the random-weight models | level 2 |
| `recovered` | outputs of the metadata-recovery scripts | level 2 |

## Level 1: statistics from the results bundle

This level needs only the Python environment: no GPU, checkpoints or neural data. It runs in under 10
seconds on a laptop CPU.

```bash
make data       # unzip data/results_bundle.zip into data/results_bundle/
make results    # python scripts/reproduction/compute_results.py --out results/summary
```

The command writes `results.json`, `training_exposure.csv` and `corpus_provenance.json` (described in
the README) and overwrites them on each run. `tests/reproduction/test_bundle_reproduction.py` checks key
statistics against the bundle whenever the bundle is unpacked.

## Level 2: the bundle from checkpoints, corpora and raw scores

The metadata-recovery scripts replay the training pipeline's tokenization and packing on the public
corpus. They also load the benchmarks through the pinned `language/` submodule. They train and score
nothing.

```bash
# about 25 min on a CPU and a 5 GB download
python scripts/reproduction/recover_packed_splits.py --out results/recovered/packed_splits.json
# about 15 min on a CPU
python scripts/reproduction/audit_corpora.py --out results/recovered/corpus_audit.json
# about 5 min; downloads the benchmark data
python scripts/reproduction/describe_benchmarks.py --out results/recovered/benchmarks.json
# raw archives -> results bundle
python scripts/reproduction/build_results_bundle.py
```

Recomputing scores from checkpoints needs the `language/` submodule with the patch in `patches/` applied
(`bash setup_env.sh` does both) and the benchmark data it downloads. A GPU is needed for reasonable speed.
Every scoring script keeps the per-fold scores, and benchmarks without a ceiling (Tuckute2024) get null
normalized values.

Each step below names the file of the results archive it produces; `<b>` is a benchmark key and
`<id>` a Brain-Score benchmark identifier.

**Seed-42 per-fold scores at every block** (`official_brainscore/layer_cis.json`,
`multi_bench/<b>/layer_cis_<b>.json`):

```bash
python -m scripts.extract_layer_cis --benchmark <id> --output <json> \
    --models IParraMartin/gpt2-<corpus>=<sha> ...
```

**Seeds 43-46 and random-weight models at every block** (`*/layer_cis_e1_seeds.json`,
`*/layer_cis_randominit.json`, `multi_bench/<b>/layer_cis_<b>_seeds.json`):

```bash
python -m scripts.score_local_checkpoints --benchmark <id> --output <json> \
    --checkpoints <corpus>_seed<s>=<dir> ...
```

**Random-weight models** (`seed_models/random_init_seed<s>`):

```bash
python -m scripts.make_random_init_models --seeds 42 43 44 --out-root <dir> --manifest <json>
```

**Official benchmark score** (`official_brainscore/brain-code-*.json`):

```bash
brain-score --hf_model_id <hub_id> --model_revision <sha> \
    --experiment pereira_243 --save_results
```

**Multi-seed geometry, CKA, RSA and permutation null** (`experiments/19/geometry_<b>.json`):

```bash
python scripts/run_latent_geometry.py --models-dir <seed_models> \
    --hf-model tinystories_seed42=IParraMartin/gpt2-tinystories ... \
    --benchmark <b> --max-length 1024 --batch-size 8 --permutations 200 --out <json>
```

**PC-ablation RSA** (`experiments/19/pc_ablation_rsa_<b>.json`):

```bash
python scripts/run_pc_ablation_rsa.py --models-dir <seed_models> --hf-model ... \
    --benchmark <b> --batch-size 8 --max-length 1024 --out <json>
```

**Progressive transplants** (`official_brainscore/layer_cis_e2b_progressive.json`):

```bash
python -m scripts.run_progressive_transplant --output <json>
```

**Single-block and embedding transplants** (`official_brainscore/layer_cis_exp11.json`,
`exp11_paired_tests.json`): build each transplant with `brain-patch` (`--patching_mode block --layer <k>`
for one block, `--patching_mode embeddings` for the embeddings), score the outputs with
`score_local_checkpoints`, then test them against the receivers:

```bash
brain-patch --patching_mode block --layer <k> --out_dir <dir> \
    --donor_model IParraMartin/gpt2-tinystories --receiver_model IParraMartin/gpt2-<corpus>
python -m scripts.single_block_paired_tests --transplants <json> --baseline layer_cis.json \
    --output <json>
```

**Embedding-only adaptation** (`experiments/09`, `experiments/13`,
`official_brainscore/layer_cis_embedft.json`): adapt each code model on TinyStories and, as the control,
on its own corpus, then score the outputs with `score_local_checkpoints`:

```bash
brain-embed-ft --model_id IParraMartin/gpt2-<corpus> --dataset <tinystories or own corpus> \
    --n_steps 2000 --max_tokens 2000000 --context_length 64 --batch_size 32 --out_dir <dir>
```

**One-checkpoint spectrum audit** (`results/spectrum/`):

```bash
brain-manifold --model <hub_id>
python scripts/compute_spectra.py
python scripts/compute_spectrum_uncertainty.py
```

**Corpus diagnostics** (`confounds/`):

```bash
brain-measure --tokenizer default --corpora_metrics
brain-confounds --benchmark_id Pereira2018.243sentences-ridge
brain-correlate ...
```

Measured costs: the multi-seed geometry took 6 minutes for 27 models on one GPU; the spectrum audit
about 2 and 12 minutes on one GPU. The progressive transplants (672 layer scorings) were estimated at
about an hour on one A5000. Other steps were not timed.

`run_latent_geometry.py` truncates long contexts from the right, unlike the benchmark's encoding path.
This matters only for Blank2014 (see `docs/benchmarks.md`).

BLiMP accuracies (`blimp/<corpus>.txt`) were computed with EleutherAI's lm-evaluation-harness, task
`blimp`, zero-shot accuracy. The exact invocation and harness version were not recorded and that tool is
not a dependency of this repository.

## Level 3: from corpora to checkpoints

```bash
# or use the shipped data/tokenizers/50k_50mb
brain-train-tokenizer --vocab_size 50000 --bytes_per_dataset 50_000_000
brain-train --dataset python --training_args seedrep --model_args small --tokenizer default \
    --seed 43 --run-name gpt2_small_python_seed43 \
    --output-dir checkpoints/python_seed43 --resume
```

Seeds 43-46 used `configs/training/train_seedrep.yaml`, which is the corpus recipe
(`train_<corpus>.yaml`, identical across corpora) without the Hub push:

- batch 16 x 1,024;
- at most 30,517 updates;
- evaluation every 1,000 updates, with patience 5;
- best weights restored at the end;
- bf16, which needs an Ampere or newer GPU.

`--resume` continues from the last `checkpoint-N/` in the output directory, so a job killed at its
walltime can be resubmitted. Comparable 500M-token runs took 10-11 hours of training plus 1.5-6 hours of
tokenization on one A40. Training is not bit-identical across GPUs or library versions.

Seed 42 (the public models) split packed sequences rather than documents and inserted no
end-of-document token; that pipeline is available through
`LMTrainer.train(split_before_pack=False, insert_eos_between_docs=False)`. Exposure is not equalized
across corpora (see `docs/corpora.md`).

The pseudoword and factorial controls train GPT-2 small for 500M tokens on rebuilt TinyStories corpora,
scoring blocks 0, 6 and 11 on Pereira-243 at every milestone through the official harness:

```bash
python -m scripts.dump_corpus --dataset-key tinystories \
    --out data/factorial_corpora/tinystories_corpus.txt
# conditions A (intact), B (shuffled), C (pseudoword) and D
brain-build-corpora --dataset-path data/factorial_corpora/tinystories_corpus.txt \
    --out-dir data/factorial_corpora_500m --seed 42
# conditions Cp and Dp
python -m scripts.build_lexical_permutation --corpora data/factorial_corpora_500m

export BRAIN_CODE_ALIGNMENT_BENCHMARK=Pereira2018.243sentences-ridge
export BRAIN_CODE_ALIGNMENT_LAYERS=first,middle,last
brain-checkpoint-sweep --dataset-path data/factorial_corpora_500m/condition_A.txt \
    --model-args small --training-args milestones --seed 42 \
    --milestones 10000000 50000000 100000000 250000000 500000000 \
    --output-dir results/checkpoint_sweep/run_condA_seed42 --save-checkpoints \
    --alignment-entrypoint brain_code.experiments.milestone_alignment:frozen_alignment
```

Conditions A and C (the pseudoword control) ran first; B, Cp and Dp complete the 2 x 2 design. Each
condition used seeds 42, 43 and 44. `configs/training/train_milestones.yaml` is the seed-replication
recipe with 30,520 updates, because 30,517 updates stop 9,472 tokens short of the 500M milestone.
`data/factorial/pos_map.json` assigns each content-word type its part of speech; it was built with spaCy
`en_core_web_sm` 3.7.5 on 120,000 sampled texts, and the script that built it was not kept.

## What has been run

- **Level 1:** run in a fresh Python 3.11 environment. The outputs are byte-identical across runs.
- **Level 2:** the four metadata-recovery scripts and the bundle builder were run in full. As spot checks,
  the progressive transplant was rebuilt from the public checkpoints for the Python and binary receivers,
  and 28 of 28 arms reproduce the recorded parameter hashes. One archived score (TinyStories seed 42,
  Pereira-243, block 0) was recomputed on a CPU, with a difference of 1.7e-8. The full set of scores was
  not recomputed.
- **Level 3:** not rerun.
