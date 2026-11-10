---
myst:
  html_meta:
    "description lang=en": "GPT-2 models trained on English and on four code corpora, compared with matched random-weight models on four neural benchmarks and by their representational geometry."
---

# Language, Not Code Abstraction: Natural‑language Training Shapes Brain‑aligned Geometry

<p class="authors">Iñigo Parra</p>
<p class="affiliation">Department of Linguistics, University of California, Berkeley</p>
<p class="venue">NeurReps 2026 · Proceedings Track</p>

<div class="link-buttons">
<a href="https://arxiv.org/pdf/2611.09027"><i class="fa-solid fa-file-pdf"></i> Paper</a>
<a href="https://arxiv.org/abs/2611.09027"><i class="fa-solid fa-file-lines"></i> arXiv</a>
<a href="https://github.com/IParraMartin/language-training-brain-geometry"><i class="fa-brands fa-github"></i> Code</a>
<a href="https://huggingface.co/datasets/IParraMartin/BrainScore-Code-English"><i class="fa-solid fa-database"></i> Corpora</a>
<a href="https://huggingface.co/collections/IParraMartin/brain-scores"><i class="fa-solid fa-cube"></i> Models</a>
<a href="reproduction.html"><i class="fa-solid fa-book"></i> Reproduction</a>
</div>

```{figure} _static/figures/fig1_design.svg
:target: _static/figures/fig1_design.svg
:figclass: teaser
:alt: Design. Five corpora (TinyStories, Python files, C functions, x86-64 assembly and machine code as hex text) pass through one shared tokenizer and GPT-2-small recipe, giving 25 trained models next to three random-weight models; every model is analysed with encoding at all twelve blocks, geometry and interventions.

Five corpora share one tokenizer and one GPT-2-small recipe, giving 25 trained models (five seeds per
corpus). C, assembly and binary hold the same functions in three representations. The matched reference
is three random-weight models for encoding and two for geometry, and every model is read at all twelve
blocks on four benchmark families, with Pereira contributing two stimulus sets.
```

## Abstract

Code-trained language models produce positive brain scores, which raises the question of whether
abstract structure learned from programming languages can substitute for natural-language training. We
trained 25 GPT-2-small models on TinyStories, Python, C functions, x86-64 assembly, and the corresponding
machine code, and scored them using a shared tokenizer and encoding procedure against matched
random-weight models. On Pereira-243 and Tuckute2024, TinyStories exceeds the mean of the four code models
by *Δr* = 0.053 and 0.059 (Holm-corrected *p* < 0.001), while code training leaves alignment near the
random-weight level. The language advantage is benchmark-specific: Blank2014 shows no corrected
language–code difference, and Fedorenko2016 shows no language benefit. Geometry tracks the encoding
result on Pereira-243: TinyStories seeds converge on the most similar representations, and its fit-free
correspondence with neural dissimilarity peaks at *ρ* = 0.14, twice the code and random-weight levels.
Binary-trained models have about three effective dimensions, so seed agreement alone does not imply
neural correspondence. Updating only the embedding matrix on English improves mean-over-depth alignment
for every code model more than an equal update on its own corpus. Within this architecture and training
setup, natural-language training produces a distinct representational geometry with closer neural
correspondence on Pereira-243.

## Two predictions

If abstraction learned from code drives alignment with the human language network, models trained on
high-level source code should align better than models trained on lower-level representations, giving a
Python > C > assembly > binary ordering. If natural-language learning drives it, a model trained on
English should separate from all code models, with no particular ordering among them.

An absolute score cannot decide between these. A randomly initialized transformer already brings a
learned tokenizer, positional structure, causal attention, random nonlinear features and, under encoding
analyses, a fitted readout, and surface properties of the stimuli add predictive power of their own. A
corpus contributes what it produces relative to an identical model that was never trained, so every
result here is an increment over matched random-weight models, with seeds as the unit of inference.

Each model is summarized by its mean encoding score over all twelve blocks, fixed in advance so that no
favorable layer is chosen after seeing the scores. Because fitted encoders can exploit surface features
of the stimuli, the encoding results are paired with analyses that fit nothing: CKA between models,
covariance spectra and RSA between model and brain. Two interventions, block transplants and
embedding-only retraining, ask which learned components carry the effect.

## Language versus code

Random weights already score above zero, so a positive code score does not show that code training
helped. On Pereira-243 and Tuckute2024, TinyStories exceeds the code mean by *Δr* = 0.053 (95% CI
0.046 to 0.060) and 0.059 (0.043 to 0.074), both surviving correction, while the code mean stays near
random weights (+0.004 and −0.007). TinyStories also exceeds random weights by 0.057 and 0.051, and every
TinyStories seed exceeds every random initialization on both benchmarks (exact rank test, *p* = 0.036),
but with only three initializations these contrasts do not survive correction (Holm *p* = 0.145). On
Blank2014, the gains of TinyStories and code over random weights (+0.014 and +0.009) leave no corrected
language–code difference (+0.005); on Fedorenko2016, both means round to *r* = 0.170, below random
weights.

{.results-table}
| Benchmark | Random | Code | TinyStories | TinyStories − random | Code − random | TinyStories − code |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pereira-243 | 0.064 | 0.069 | 0.122 | +0.057 | +0.004 | **+0.053** |
| Tuckute2024 | 0.274 | 0.266 | 0.325 | +0.051 | −0.007 | **+0.059** |
| Blank2014 | 0.040 | 0.049 | 0.054 | +0.014 | +0.009 | +0.005 |
| Fedorenko2016 | 0.175 | 0.170 | 0.170 | −0.004 | −0.005 | +0.000 |

<p class="table-note">Mean-over-depth encoding scores (raw Pearson <i>r</i> averaged over the twelve
blocks; five seeds per trained condition, three random initializations). Code is the equally weighted mean
of Python, C, assembly and binary. Bold contrasts survive Holm correction over all twelve tests, with the
variance of the random initializations included.</p>

The gap is not confined to a favorable layer. On Pereira-243, TinyStories and code models do not differ
significantly at block 0 (Holm *p* = 0.61), and TinyStories is significantly higher at every later
block, with a selected maximum gap of *Δr* = 0.079 at block 4. Pereira-384 and Tuckute2024 show the same
pattern at 11 of 12 blocks, again sparing block 0.

```{figure} _static/figures/fig2_pereira243.svg
:target: _static/figures/fig2_pereira243.svg
:alt: Pereira-243 encoding score at each block for every corpus, with TinyStories rising well above the code corpora and the random-weight band, and the mean over twelve blocks for each model.

Pereira-243. **(a)** Encoding score at each block: mean of five seeds per corpus, with ±1 SD for
TinyStories; the gray band spans the three random initializations. **(b)** Mean over twelve blocks for
each model; bars are condition means. Seeds are the replicates; folds do not enter the uncertainty.
```

```{figure} _static/figures/fig7_four_benchmarks.svg
:target: _static/figures/fig7_four_benchmarks.svg
:alt: Change relative to the random-weight mean at each block on Pereira-243, Tuckute2024, Blank2014 and Fedorenko2016.

Change relative to the random-weight mean at each block on the four benchmark families (mean of five
seeds per corpus; the gray band spans the random initializations). Vertical scales differ between
benchmarks.
```

## Representational geometry

Encoding scores depend on a fitted readout, so we also compared the representations directly, on the same
Pereira-243 stimuli and contexts. Across blocks 4–11, TinyStories seeds have the highest mutual CKA
(0.90), followed by binary (0.86), C (0.81), Python (0.71), assembly (0.69) and random weights (0.67). CKA
among TinyStories seeds exceeds that between TinyStories and code models by 0.06 at block 0 and by 0.39
from block 4 on. Training compresses every corpus relative to random weights, whose participation ratio
is 83 at block 0 and 62 at block 11: binary occupies about two to three effective dimensions across
blocks, whereas TinyStories expands to about 26.

These properties come apart. Binary is nearly as reproducible as TinyStories while occupying only about
three effective dimensions, so seed agreement alone does not identify a brain-like representation.
Fit-free RSA separates the corpora: TinyStories rises from *ρ* = 0.06 at block 0 to 0.14 at block 6,
while every code model peaks at or below 0.07, the random-weight range, and Python falls below random
weights in later blocks. Removing leading principal components preserves the TinyStories lead. On
Pereira-243, only English training combines high reproducibility, intermediate dimensionality and the
highest fit-free correspondence. The RSA advantage is specific to this stimulus set: on Pereira-384 and
Fedorenko2016, binary reaches the highest RSA.

```{figure} _static/figures/fig3_geometry.svg
:target: _static/figures/fig3_geometry.svg
:alt: Seed-to-seed CKA, participation ratio on a log scale, and brain–model RSA at each block for every corpus.

Representational geometry on Pereira-243 stimuli (cumulative topic context, final token). **(a)** Mean
linear CKA between seeds of the same corpus. **(b)** Participation ratio, log scale. **(c)** Brain–model
RSA (Spearman *ρ* between dissimilarity matrices). Five seeds per corpus and two random initializations;
all panels are descriptive.
```

## No abstraction ordering

The code models are not interchangeable. Their encoding scores differ significantly at 2 of 12
Pereira-243 blocks and 8 of 12 Pereira-384 blocks, and their geometries differ more: at blocks 4–11,
Python–C CKA is 0.50, below the agreement among seeds of either (0.71 and 0.81). Python is the most
distinct code representation (0.23 with binary), and C and assembly, which hold the same functions, are
the most similar pair (0.62).

These differences do not follow the predicted ordering. C leads the code models at 8 of 12 Pereira-243
blocks and all 12 Pereira-384 blocks, assembly leads at the other four Pereira-243 blocks, and Python
never leads; the full Python > C > assembly > binary order occurs in none of the 60 block profiles across
the five benchmark sets. Within the matched triplet, the Pereira depth summaries do decline from C to
assembly to binary (0.079, 0.071 and 0.060 on Pereira-243), so on Pereira the ordering fails because
Python, the highest-level language, scores below C; on the other three benchmarks binary leads the code
models. The triplet holds programs fixed, but the models saw 1.0, 0.35 and 0.61 passes of them, and
Python differs from C in content and selection as well as level. The result argues against this
abstraction hierarchy under these conditions, not against any contribution of abstraction or code
learning to alignment.

```{figure} _static/figures/fig6_depth_replication.svg
:target: _static/figures/fig6_depth_replication.svg
:alt: TinyStories minus code mean at each block on both Pereira sets, and the four code corpora at each block with asterisks for significant heterogeneity.

Depth-resolved replication. **(a)** TinyStories minus the code mean at each block on both Pereira sets;
filled markers are significant after Holm correction over blocks. **(b, c)** Code corpora, mean ±1 SD over
five seeds; asterisks mark blocks where the code corpora differ.
```

## Word forms and lexical exposure

English training also means seeing the stimulus words, but the corpus diagnostics suggest that lexical
exposure alone is insufficient. Binary contains 1% of the Pereira word types and Python 98%, yet both stay
near the random-weight level, and the average stimulus token is more frequent in assembly (6.3M
occurrences) than in TinyStories (4.1M). With five corpora, a correlation between exposure and score
(*r* = 0.38, *p* = 0.53) cannot rule out a relationship, so these counterexamples carry the argument.
BLiMP separates TinyStories (0.64) from the code models (0.53–0.55) but orders the code models
differently from the neural scores.

A form control is stronger. Relabeling every TinyStories content word with a pseudoword leaves alignment
comparable at 500M tokens (mean of blocks 0, 6 and 11: 0.107 intact, 0.115 relabeled; three seeds, Welch
*p* = 0.12). The one-to-one map preserves word types, frequencies, co-occurrences and syntactic frames and
changes only the forms, and a comparable endpoint is not an equivalence test. In a 2 × 2 design that
crosses word-order shuffling with a permutation of words within part-of-speech classes, neither
manipulation produces a detectable decrease at 500M tokens, although combining both gives the lowest cell
(0.096). The pseudoword model retains substantial alignment without familiar content-word forms; whether
the preserved regularities suffice remains open.

## Embeddings and blocks

Transplanting TinyStories blocks alone does not reproduce the donor's alignment. With all twelve blocks
placed over code embeddings, the Python, C and binary receivers stay near *r* = 0.07 at block 9, and
assembly falls to −0.01. When the TinyStories token and position embeddings are copied as well, every
receiver reaches the donor's level (*r* = 0.114) by *k* = 4 and equals it at *k* = 12, where the model is
the donor. Retraining only the tied embedding matrix on TinyStories raises the mean-over-depth score of
every code model (+0.028 on average), compared with a smaller mean change under the same budget on its
own corpus (+0.002; paired *t*(3) = 4.31, *p* = 0.023). This paired comparison describes four fixed
checkpoints rather than generalizing over code languages, and English embeddings alone do not reproduce
the donor. Compatibility between embeddings and downstream blocks therefore matters: a learned stitch
might enable block-only transfer, and the effect is not localized to the embeddings.

```{figure} _static/figures/fig4_interface.svg
:target: _static/figures/fig4_interface.svg
:alt: Encoding r at block 9 after transplanting the first k TinyStories blocks into each code model, without and with the TinyStories embeddings, and the change after embedding-only adaptation on the own corpus or on TinyStories.

Interventions on Pereira-243, one donor (TinyStories, seed 42) and one receiver checkpoint per code
corpus. **(a)** Encoding *r* at block 9 after replacing the receiver's first *k* blocks with the donor's;
*k* = 0 is the untouched receiver and *k* = 12 also copies the final normalization. **(b)** The same
transplants with the donor's token and position embeddings copied as well; *k* = 12 is the donor.
**(c)** Change in mean-over-depth *r* after updating only the tied embedding matrix on each model's own
corpus or on TinyStories, with equal budgets.
```

## Benchmark-specific outcomes

Blank2014's story contexts exceed the 1,024-token window for 417 of 1,317 time points. Encoding keeps the
most recent tokens, but the geometry extraction keeps the first 1,024, so 411 time points have an input
identical to an earlier one and the near-zero Blank2014 RSA says nothing about the neural data. RSA is not
interpreted for Tuckute2024 either, whose benchmark exposes a single averaged response. On Fedorenko2016,
neither TinyStories nor the pooled code mean significantly exceeds random weights on word-by-word ECoG
responses, and binary has the highest descriptive mean. A model is not simply more brain-like: the
dataset, readout and reference determine which differences an analysis can detect.

## Scope

The study uses one 124M-parameter GPT-2-small architecture, one English distribution and one tokenizer
trained on all five corpora. The recipe and maximum budget are shared, but realized exposure is not: it
ranges from 0.35 passes over assembly to 6.8 over a heavily duplicated Python corpus, whose validation
documents largely recur in training. Inference generalizes over optimization runs of this design, not over
participants, stimuli, architectures or training distributions; the random reference has three
initializations, and transplants use one checkpoint pair per code corpus without a learned stitch. The
models are compared with human responses to language, so the results constrain what code training
contributes to language-network alignment, not how people process code.

## Reproducing the results

Every reported statistic is recomputed in seconds on a CPU from the released results bundle:

```bash
git clone --recurse-submodules https://github.com/IParraMartin/language-training-brain-geometry.git
cd language-training-brain-geometry
bash setup_env.sh
make results
```

The [reproduction guide](reproduction.md) gives the commands that rebuild the bundle from checkpoints and
retrain the models from the public corpora, and the [corpora](corpora.md) and [benchmarks](benchmarks.md)
pages document the data as it was measured. The seed-42 models and the training corpus are public on the
Hugging Face Hub; the seed 43–46 checkpoints are not released.

## BibTeX

The paper is available on [arXiv](https://arxiv.org/abs/2611.09027). This entry will be updated when the
proceedings are published.

```text
@misc{parra2026language,
  author        = {Parra, I{\~n}igo},
  title         = {Language, Not Code Abstraction: Natural-language Training Shapes Brain-aligned Geometry},
  year          = {2026},
  eprint        = {2611.09027},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2611.09027}
}
```

```{toctree}
:hidden:
:maxdepth: 2

reproduction
corpora
benchmarks
code_guide
```
