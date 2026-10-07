# Training corpora and realized exposure

All figures on this page were measured on the stored data, by `scripts/reproduction/audit_corpora.py` and
`scripts/reproduction/recover_packed_splits.py`; the per-run report is `results/summary/training_exposure.csv`, written by `make results`.
Where something could not be recovered, it says so.

## Source

The five corpora are configurations of the Hugging Face dataset `IParraMartin/BrainScore-Code-English`
(revision `cc83566aebbaa2eeb76104cc956e883a722b9ded`, last modified 2026-05-16, before the first model was
pushed on 2026-05-20). The dataset card declares MIT and describes the data only as "clean code and the
TinyStories dataset". A sixth configuration (Brainfuck) exists but was excluded from tokenizer training
and from every model since the first commit of the training code.

| Corpus | Rows | Unit | What a row contains |
|---|---:|---|---|
| tinystories | 2,119,719 | story | the original TinyStories training split (`roneneldan/TinyStories`, CDLA-Sharing-1.0); the first 2,000 rows are identical to the source |
| python | 165,440 | file | Python files regenerated from their syntax tree, which removes comments and leading docstrings |
| source | 2,800,000 | function | one C function |
| assembly | 2,800,000 | function | the same function as x86-64 AT&T assembly |
| binary | 2,800,000 | function | the same function's machine code as hexadecimal text |

**C, assembly and binary are aligned.** The three configurations share the same 2.8M `idx` values; C
function names appear in the assembly of the same row in 1,992 of 1,993 sampled rows, and the
hexadecimal bytes of a row decode (objdump, x86-64) to the instructions of its assembly. The assembly uses
GCC-style directives (`.cfi_*`, `.LFB0` labels), `endbr64`, frame-pointer prologues and argument spills,
consistent with unoptimized compilation; the compiler, its version and flags are not recorded.

**Binary is text, not bytes.** Each row is space-separated two-digit hexadecimal tokens: the function's
machine code with zero call and RIP-relative displacements (unlinked object code), preceded by one token
and interleaved with single tokens after branch and relocation fields that are not part of the instruction
stream. Their encoding is undocumented. Example (row 0, `timespec_get`):

```
C        static inline int timespec_get(struct timespec *ts, int base) { if (!ts) return 0; ...
assembly endbr64 / pushq %rbp / movq %rsp, %rbp / subq $16, %rsp / ...
binary   64 f3 0f 1e fa 55 48 89 e5 48 83 ec 10 48 89 7d f8 89 75 f4 ...
```

**What remains of natural language in code.** Python keeps identifiers and string literals (23% of
characters in a sample of 5,000 files); 99.98% of sampled files are fixed points of
`ast.unparse(ast.parse(x))`. About 6% of C functions contain English `/* ... */` comments (1,259 of
20,000 sampled), so comment removal was not applied to C. Assembly keeps symbol names derived from C
identifiers; binary has none.

**Not recoverable.** The upstream collection of the C functions and Python files, their selection and
filtering criteria, the compilation toolchain and the binary encoding are not described in the dataset or
the training code.

## Duplication and splits

| Corpus | Unique rows | Validation documents also in training (seed-43 split) | Documents truncated at 1,024 tokens |
|---|---:|---:|---:|
| tinystories | 85% | 27.0% | 0.0% |
| python | 36% | 72.8% | 24.1% |
| source | 100% | 0.0% | 2.4% |
| assembly | 96% | 6.7% | 20.7% |
| binary | 73% | 32.1% | 8.1% |

Splits are random over rows with no grouping by repository, program or compilation, and the matched C,
assembly and binary rows are split independently in each corpus.

## Tokenizer

One byte-level BPE tokenizer (vocabulary 50,000, one special token `<|endoftext|>` used as BOS, EOS, UNK
and padding) was trained once on a shuffled 50 MB stream from each of the five corpora (seed 42, shuffle
buffer 10,000 documents). Every checkpoint carries the identical vocabulary and merges
(`data/tokenizers/50k_50mb`).

## Packing and exposure

1. Each document is tokenized and cut to its first 1,024 tokens.
2. Seeds 43-46: 10% of documents are held out (`train_test_split(test_size=0.1, seed=seed)`), and an
   end-of-document token is appended to each document. Seed 42: no separator; the 10% split is drawn from
   packed sequences instead.
3. Consecutive batches of 1,000 documents are concatenated and cut into 1,024-token sequences; each batch's
   remainder is dropped. Sequences contain no padding.
4. The collator masks the separator token from the loss, because it doubles as the padding token.
5. Each update processes 16 x 1,024 = 16,384 input positions. Training stops at 30,517 updates or after
   five validation checks without improvement, and best-validation weights are restored.

| Corpus | Tokens after truncation | Training sequences (5 seeds) | Passes at update 30,000 (logged seeds) |
|---|---:|---:|---:|
| tinystories | 463M | 406.1k-408.0k | 1.18 |
| python | 80.4M | 70.6k-70.8k | 6.78-6.79 |
| source | 547M | 479.6k-482.2k | 1.00 |
| assembly | 1.54B | 1,350.8k-1,353.4k | 0.35 |
| binary | 900M | 789.5k-792.2k | 0.61 |

All logged runs (seeds 43-46) reached the 30,517-update cap; early stopping never triggered, and the
selected checkpoint is update 30,000 (491.5M processed positions; 500.0M at termination). Seed 42's
training logs, step count and selected checkpoint were not retained. Pass equivalents divide processed
positions by the training-split size; they are not counts of distinct documents. Device and GPU-hours were
not recorded in the provenance files.

The pseudoword and factorial runs are separate records (`factorial_milestones.csv`): GPT-2 small,
rebuilt TinyStories corpora, 500M non-padding tokens, no early stopping, three seeds per condition.
