---
myst:
  html_meta:
    "description lang=en": "GPT-2 models trained on English and on four code corpora, compared with matched random-weight models on four neural benchmarks and by their representational geometry."
---

# Language Training Shapes Brain-Aligned Geometry

<p class="authors">Iñigo Parra</p>
<p class="affiliation">Department of Linguistics, University of California, Berkeley</p>
<p class="venue">NeurReps 2026 · Proceedings Track</p>

<div class="link-buttons">
<span class="disabled" aria-disabled="true"><i class="fa-solid fa-file-pdf"></i> Paper (coming soon)</span>
<a href="https://github.com/IParraMartin/language-training-brain-geometry"><i class="fa-brands fa-github"></i> Code</a>
<a href="https://huggingface.co/datasets/IParraMartin/BrainScore-Code-English"><i class="fa-solid fa-database"></i> Corpora</a>
<a href="https://huggingface.co/IParraMartin/gpt2-tinystories"><i class="fa-solid fa-cube"></i> Models</a>
<a href="reproduction.html"><i class="fa-solid fa-book"></i> Reproduction</a>
</div>

```{figure} _static/figures/fig1_design.svg
:target: _static/figures/fig1_design.svg
:figclass: teaser
:alt: Design. Five corpora (TinyStories, Python files, C functions, x86-64 assembly and machine code as hex text) pass through one shared tokenizer and GPT-2-small recipe, giving 25 trained models next to three random-weight models; every model is analysed with encoding at all twelve blocks, geometry and interventions.

Five corpora share one tokenizer and one GPT-2-small recipe, giving 25 trained models (five seeds per
corpus). C, assembly and binary hold the same functions in three representations. Three random-weight
models serve as a matched reference, and every model is read at all twelve blocks on four benchmark
families.
```

## Abstract

Code-trained language models produce positive brain scores, which raises the question of whether
abstract structure learned from programming languages can substitute for natural-language training. We
trained 25 GPT-2-small models, five seeds each, on TinyStories, Python, C functions, the x86-64 assembly
compiled from those functions, and the corresponding machine code written as hexadecimal text, and scored
them with a shared tokenizer and readout against matched random-weight models. On Pereira-243 and
Tuckute2024, TinyStories exceeds the mean of the four code models by *Δr* = 0.053 and 0.059
(Holm-corrected *p* < 0.001), while code training leaves alignment near the random-weight level (+0.004
and −0.007). The language advantage is benchmark-specific: Blank2014 shows no corrected language–code
difference and Fedorenko2016 shows no language benefit. The predicted Python > C > assembly > binary
ordering does not appear at any block. Geometry tracks the encoding result on Pereira-243: TinyStories
seeds converge on the most similar representations, and its fit-free correspondence with neural
dissimilarity peaks at *ρ* = 0.14, twice the code and random-weight levels. Binary-trained models are as
reproducible but collapse to about three dimensions, so seed agreement alone does not imply neural
correspondence. Updating only the embedding matrix on English improves every code model more than an
equal update on its own corpus. Within this architecture and training setup, natural-language training
produces a distinct representational geometry that aligns with two of four neural benchmarks.

## Two predictions

If abstraction learned from code drives alignment with the human language network, models trained on
high-level source code should align better than models trained on lower-level representations, giving a
Python > C > assembly > binary ordering. If natural-language learning drives it, a model trained on
English should separate from all code models, with no particular ordering among them.

Absolute scores cannot decide between these. A randomly initialized transformer already brings a learned
tokenizer, positional structure, causal attention, random nonlinear features and a fitted readout, so
every result here is an increment over matched random-weight models, with seeds as the unit of inference.

## Language versus code

On Pereira-243 and Tuckute2024, TinyStories exceeds the code mean by *Δr* = 0.053 (95% CI
0.046 to 0.060) and 0.059 (0.043 to 0.074), while the code mean stays near random weights. On
Blank2014 both kinds of training sit slightly above random weights and their difference survives no
correction; on Fedorenko2016 trained models are at or below random weights.

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

The gap is not confined to a favorable layer. On Pereira-243, TinyStories and code models coincide at
block 0 and TinyStories is significantly higher at every later block, with a largest gap of
*Δr* = 0.079 at block 4.

```{figure} _static/figures/fig2_pereira243.svg
:target: _static/figures/fig2_pereira243.svg
:alt: Pereira-243 encoding score at each block for every corpus, with TinyStories rising well above the code corpora and the random-weight band, and the mean over twelve blocks for each model.

Pereira-243. **(a)** Encoding score at each block: mean of five seeds per corpus, with ±1 SD for
TinyStories; the gray band spans the three random initializations. **(b)** Mean over twelve blocks for
each model; bars are condition means.
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
Pereira-243 stimuli. Across blocks 4–11, TinyStories seeds have the highest mutual CKA (0.90), and CKA
among TinyStories seeds exceeds that between TinyStories and code models by 0.39 from block 4 on.
Fit-free RSA separates the corpora: TinyStories rises from *ρ* = 0.06 at block 0 to 0.14 at block 6,
while every code model peaks at or below 0.07, the random-weight range.

These properties come apart. Binary models are nearly as reproducible as English ones (CKA 0.86) while
collapsing to about three effective dimensions, so seed agreement alone does not identify a brain-like
representation. Only English training combines high reproducibility, intermediate dimensionality and the
highest fit-free correspondence.

```{figure} _static/figures/fig3_geometry.svg
:target: _static/figures/fig3_geometry.svg
:alt: Seed-to-seed CKA, participation ratio on a log scale, and brain–model RSA at each block for every corpus.

Representational geometry on Pereira-243. **(a)** Mean linear CKA between seeds of the same corpus.
**(b)** Participation ratio, log scale. **(c)** Brain–model RSA (Spearman *ρ* between dissimilarity
matrices). Five seeds per corpus and two random initializations; all panels are descriptive.
```

## No abstraction ordering

The code models are not interchangeable. Their encoding scores differ significantly at 2 of 12
Pereira-243 blocks and 8 of 12 Pereira-384 blocks, and their geometries differ more: at blocks 4–11, Python–C CKA is 0.50,
below the agreement among seeds of either. Yet the full Python > C > assembly > binary order occurs at
none of 60 block profiles across the five benchmark sets. Python, the highest-level language, scores
lowest on Pereira, and binary leads the code models on the other three benchmarks.

```{figure} _static/figures/fig6_depth_replication.svg
:target: _static/figures/fig6_depth_replication.svg
:alt: TinyStories minus code mean at each block on both Pereira sets, and the four code corpora at each block with asterisks for significant heterogeneity.

Depth-resolved replication. **(a)** TinyStories minus the code mean at each block on both Pereira sets;
filled markers are significant after Holm correction over blocks. **(b, c)** Code corpora, mean ±1 SD over
five seeds; asterisks mark blocks where the code corpora differ.
```

## Embeddings and blocks

Transplanting all twelve TinyStories blocks into a code model does not raise its score; adding the
TinyStories token and position embeddings recovers the donor's level by *k* = 4 blocks. Retraining only
the tied embedding matrix on TinyStories raises the mean-over-depth score of every code model
(+0.028 on average), whereas the same budget on its own corpus does not (+0.002; paired
*t*(3) = 4.31, *p* = 0.023). The advantage therefore depends on compatibility between learned embeddings
and downstream blocks; without a learned stitch, these transplants cannot localize it.

```{figure} _static/figures/fig4_interface.svg
:target: _static/figures/fig4_interface.svg
:alt: Encoding r at block 9 after transplanting the first k TinyStories blocks into each code model, without and with the TinyStories embeddings, and the change after embedding-only adaptation on the own corpus or on TinyStories.

Interventions on Pereira-243, one donor and one receiver checkpoint per code corpus. **(a)** Encoding *r*
at block 9 after replacing the receiver's first *k* blocks with the donor's. **(b)** The same with the
donor's embeddings copied as well; *k* = 12 is the donor. **(c)** Change in mean-over-depth *r* after
updating only the tied embedding matrix on each model's own corpus or on TinyStories.
```

## Scope

The study uses one 124M-parameter architecture, one English distribution and one tokenizer trained on all
five corpora. The recipe and maximum budget are shared, but realized exposure is not: from 0.35 passes
over assembly to 6.8 over a heavily duplicated Python corpus. Inference generalizes over optimization runs
of this design, not over participants, stimuli, architectures or training distributions, and the models
are compared with human responses to language, not with how people read code.

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
pages document the data as it was measured.

## BibTeX

The citation will be updated when the proceedings are published.

```text
@inproceedings{parra2026language,
  author    = {Parra, I{\~n}igo},
  title     = {Language Training Shapes Brain-Aligned Geometry},
  booktitle = {Symmetry and Geometry in Neural Representations (NeurReps)},
  series    = {Proceedings of Machine Learning Research},
  year      = {2026}
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
