# Patches against the pinned `language/` submodule

`setup_env.sh` applies every `*.patch` here to `language/` (Brain-Score language at commit `6650678`)
after the submodule is checked out; an already-patched tree is left alone. Each patch is a deviation from
upstream that a reader of the results has to be able to see.

## `ridgecv-single-neuroid.patch`

`RidgeGCVTorch.fit` ravelled `coef_` whenever the target had a single column. Tuckute2024 has exactly
one neuroid, so its `(n, 1)` target lost its column axis, `predict()` returned a 1-D vector, and
packaging the prediction failed with `ValueError: different number of dimensions on data and dims: 1 vs 2`.
The patch ravels only a genuinely 1-D target:

| target shape | benchmark | `coef_.ndim` before | after |
|---|---|---|---|
| `(n, 1)` | Tuckute2024 | 1 (crashes) | 2 |
| `(n, 60)` | Blank2014 | 2 | 2 |
| `(n,)` | 1-D caller | 1 | 1 |

Scores of multi-target benchmarks are unchanged. All reported Tuckute2024 scores were computed with this
patch applied.
