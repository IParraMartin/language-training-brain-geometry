# Neural benchmarks and the encoding pipeline

Implementation: Brain-Score language, the `language/` submodule at commit
`6650678f5f28f3d71ca4a0d086f622834794eaa0`. Counts below are from `scripts/reproduction/describe_benchmarks.py`,
which loads the data exactly as each benchmark does and scores nothing.

| Benchmark | Modality, task | Scored units | Targets | Participants (loaded) | CV groups |
|---|---|---:|---:|---:|---|
| Pereira-243 | fMRI, sentence reading | 243 sentences | 8,031 voxels | 6 | 10-fold GroupKFold over 24 stories |
| Pereira-384 | fMRI, sentence reading | 384 sentences | 12,155 voxels | 9 | 10-fold GroupKFold over 24 stories |
| Tuckute2024 | fMRI, sentence reading | 1,000 sentences | 1 averaged response | (5, averaged) | 10 random 90/10 splits |
| Blank2014 | fMRI, story listening | 1,317 time points | 60 fROIs | 5 | 8-fold GroupKFold over 8 stories |
| Fedorenko2016 | ECoG, sentence reading | 416 words | 97 electrodes | 5 | 10-fold GroupKFold over 52 sentences |

Benchmark and data identifiers in the pinned implementation:

- Pereira-243: `Pereira2018.243sentences-ridge`, data `Pereira2018.language`, experiment `243sentences`.
- Pereira-384: `Pereira2018.384sentences-ridge`, data `Pereira2018.language`, experiment `384sentences`.
- Tuckute2024: `Tuckute2024-ridge`, data `Tuckute2024.language`.
- Blank2014: `Blank2014-ridge`, data `Blank2014.fROI`.
- Fedorenko2016: `Fedorenko2016-ridge`, data `Fedorenko2016.language`.

Original studies, from their methods sections:

- **Pereira et al. (2018)**, doi:10.1038/s41467-018-03068-4. Experiment 2: 384 sentences in 96 passages
  (24 topics, 4 passages each), 8 participants. Experiment 3: 243 sentences in 72 passages (24 topics, 3
  passages each), 6 participants. Sentences read one at a time. The loader contains 9 participants for
  the 384-sentence set and 24 context groups for the 243-sentence set, so a model's cumulative context
  spans a topic rather than a passage there. "243" and "384" are sentence counts, not participant counts.
- **Tuckute et al. (2024)**, doi:10.1038/s41562-023-01783-7. 1,000 six-word sentences read by five
  participants; the left-hemisphere language-network response was averaged within and across
  participants. The benchmark exposes that single averaged response and has no ceiling.
- **Blank et al. (2014)**, doi:10.1152/jn.00884.2013. Ten participants listened to stories (TR 2 s);
  language fROIs were defined in each participant. The loader contains 5 participants, 8 stories and 12
  fROIs per participant; its units are time points, not sentences.
- **Fedorenko et al. (2016)**, doi:10.1073/pnas.1612132113. Six epilepsy patients read eight-word
  sentences word by word; broadband gamma from language-responsive electrodes. The loader contains 5
  patients, 97 electrodes and 52 sentences x 8 words.

## Encoding pipeline

1. **Context.** Stimuli of a passage (Pereira), story (Blank) or sentence (Fedorenko) are presented in
   order; for stimulus k the model reads the space-joined text of stimuli 1..k. Tuckute sentences are read
   alone.
2. **Tokenization and truncation.** The model's own tokenizer; contexts longer than 1,024 tokens are
   truncated from the left (the harness loads the tokenizer with `truncation_side="left"`), keeping the
   most recent tokens.
3. **Readout.** The last token's activation at one block (`transformer.h.k`, the output of block k),
   model in evaluation mode without gradients.
4. **Regression.** Ridge with generalized cross-validation over alpha in {1e-3, ..., 1e3} (7 values), one
   alpha shared across targets, intercept fitted, no feature or target standardization, all fitted inside
   the training fold.
5. **Score.** Pearson r per target on the held-out fold. The benchmark's own score is the median over
   targets of the fold-averaged correlation; the analyses here use the mean over targets within
   each fold, averaged over folds (`scripts/extract_layer_cis.py`). The two differ by 0.005-0.010 for the
   seed-42 models. Benchmark ceilings use the benchmark's aggregation and are not applied.

**Blank2014.** 417 of 1,317 cumulative contexts exceed 1,024 tokens (maximum 1,793). Under left truncation
all 1,317 inputs stay distinct; under right truncation, as used by `scripts/run_latent_geometry.py`, 411
become byte-identical to an earlier input, which is why Blank2014 RSA is not interpreted. Seeds 43-46
were scored on Tuckute2024, Blank2014 and Fedorenko2016 by `scripts/score_local_checkpoints.py`, which
also loads the tokenizer with `truncation_side="left"`.
