"""Strict neural adapters and explicit context construction for custom encoding.

The custom encoding path must never infer scientific dimensions by array order or
silently average subjects, repetitions, or time. Each supported benchmark has an
adapter specification. Unknown layouts fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from brain_code.experiments.linalg_backend import resolve_model_device

_POOLINGS = ("last", "mean", "sum")
_CONTEXT_MODES = ("independent_sentence", "cumulative_passage")


def _model_fingerprint(model) -> str:
    """Short digest of a model's architecture and weights.

    Strided samples rather than whole tensors: hashing every parameter of a 124M-parameter model on each
    call costs more than the forward pass it saves. Every 997th element (a prime, so it does not align
    with row or head boundaries) of every parameter, plus each tensor's name and shape, separates a
    trained model from a randomly initialised one of the same architecture.
    """
    digest = hashlib.sha256()
    cfg = getattr(model, "config", None)
    if cfg is not None:
        digest.update(str(getattr(cfg, "to_diff_dict", lambda: cfg)()).encode())
    try:
        import torch

        with torch.no_grad():
            for name, param in sorted(model.named_parameters(), key=lambda kv: kv[0]):
                digest.update(name.encode())
                digest.update(str(tuple(param.shape)).encode())
                flat = param.detach().reshape(-1)
                digest.update(flat[::997].to(torch.float32).cpu().numpy().tobytes())
    except Exception:  # noqa: BLE001 - a fingerprint must never break extraction
        digest.update(b"__fingerprint_unavailable__")
    return digest.hexdigest()[:16]


def _inputs_hash(contexts: list[str]) -> str:
    """Digest of the built model inputs, which depend on the context mode and grouping."""
    return hashlib.sha256("\x00".join(contexts).encode("utf-8", "surrogatepass")).hexdigest()[:16]


def _activation_cache_path(
    cache_dir: str | Path,
    cache_key: str,
    context_mode: str,
    pooling: str,
    layer_index: int,
    *,
    model_fingerprint: str,
    inputs_hash: str
) -> Path:
    """Canonical path for one cached activation array.

    Format: ``{cache_dir}/{cache_key}_{model_fp}_{inputs_hash}_{context_mode}_{pooling}_L{layer:02d}.npy``

    The model fingerprint and the inputs hash are part of the name because the environment key alone
    cannot tell a trained model from an untrained one of the same architecture, nor two probe sets of
    the same size, and either collision serves one call another call's activations. Callers can still
    bypass the cache by clearing ``BRAIN_CODE_CACHE_KEY``.
    """
    fname = f"{cache_key}_{model_fingerprint}_{inputs_hash}_{context_mode}_{pooling}_L{layer_index:02d}.npy"
    return Path(cache_dir) / fname


def _activation_cache_env() -> tuple[str, str] | None:
    """Cache directory and key from the environment, or None when caching is off."""
    cache_dir = os.environ.get("BRAIN_CODE_ACTIVATION_CACHE", "").strip()
    cache_key = os.environ.get("BRAIN_CODE_CACHE_KEY", "").strip()
    if cache_dir and cache_key:
        return cache_dir, cache_key
    return None


def _ensure_pad_token(tokenizer) -> None:
    """Pads with EOS when the tokenizer has no pad token, as GPT-2 does."""
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token


@dataclass(frozen=True)
class NeuralAdapterSpec:
    benchmark_prefix: str
    presentation_dims: tuple[str, ...] = ("presentation",)
    neuroid_dims: tuple[str, ...] = ("neuroid",)
    stimulus_coords: tuple[str, ...] = (
        "stimulus_id",
        "stimulus",
        "sentence",
        "stimulus_sentence",
        "word"
    )
    text_coords: tuple[str, ...] = (
        "stimulus_sentence",
        "sentence",
        "stimulus",
        "word"
    )
    group_coords: tuple[str, ...] = ()
    context_group_coords: tuple[str, ...] = ()
    permitted_singleton_dims: tuple[str, ...] = ()
    required_filters: dict[str, object] = field(default_factory=dict)
    # the official benchmark's CV grouping (``split_coord`` with kfold="group");
    # the resolved grouping must equal it or fold membership diverges from the
    # official one. None means the benchmark has no grouped ridge CV (e.g.
    # split-halves consistency), so no parity constraint applies
    official_split_coord: str | None = None
    official_context_coord: str | None = None
    verified_against_submodule_sha: str | None = None


# Coordinate names below are read from the pinned brainscore-language submodule
# (SHA a897325), not guessed:
#   - dims: assembly.dims == ("presentation", "neuroid")
#   - text: "stimulus"        (data/pereira2018/data_packaging.py sets it)
#   - stable id: "stimulus_id" (= experiment + stimulus_num; globally unique)
#   - grouping: benchmarks/*/benchmark.py CV kwargs `split_coord` + kfold="group"
# Pereira2018 & Blank2014 group by "story"; Fedorenko2016 by "sentence_id".
_PINNED_SUBMODULE_SHA = "a89732599a5a458baaf1695f3eb6c89056944bd2"

_ADAPTERS = (
    NeuralAdapterSpec(
        benchmark_prefix="Pereira2018.",
        # Official: split_coord="story" (benchmark.py). `story` = experiment +
        # passage_category, a COARSER grouping than per-passage `passage_label`;
        # grouping by passage_label would leak same-category passages across folds.
        group_coords=("story", "passage_label", "passage_index", "passage_id", "passage"),
        context_group_coords=("passage_label", "passage_index", "passage_id", "passage"),
        official_split_coord="story",
        official_context_coord="passage_label",
        verified_against_submodule_sha=_PINNED_SUBMODULE_SHA
    ),
    NeuralAdapterSpec(
        benchmark_prefix="Blank2014",
        # Official: split_coord="story".
        group_coords=("story", "story_id", "passage", "passage_id"),
        context_group_coords=("story", "story_id", "passage", "passage_id"),
        official_split_coord="story",
        official_context_coord="story",
        verified_against_submodule_sha=_PINNED_SUBMODULE_SHA
    ),
    NeuralAdapterSpec(
        benchmark_prefix="Fedorenko2016",
        # Official: split_coord="sentence_id".
        group_coords=("sentence_id", "stimulus_id", "run_id"),
        context_group_coords=("sentence_id", "stimulus_id", "run_id"),
        official_split_coord="sentence_id",
        verified_against_submodule_sha=_PINNED_SUBMODULE_SHA
    ),
    NeuralAdapterSpec(
        # No benchmark source in the pinned submodule — grouping UNVERIFIED.
        benchmark_prefix="Tuckute2024",
        group_coords=("run_id", "sentence_id", "stimulus_id"),
        context_group_coords=("run_id", "sentence_id", "stimulus_id"),
        official_split_coord=None
    ),
    NeuralAdapterSpec(
        # Futrell2018 uses SplitHalvesConsistency over subject_id, not grouped
        # ridge CV — no fold-grouping parity constraint applies to this adapter.
        benchmark_prefix="Futrell2018",
        group_coords=("story_id", "story", "sentence_id"),
        context_group_coords=("story_id", "story", "sentence_id"),
        official_split_coord=None
    )
)


@dataclass
class BenchmarkData:
    sentences: list[str]
    responses: NDArray[np.float64]
    groups: NDArray
    context_groups: NDArray
    stimulus_ids: NDArray
    raw_stimulus_ids: NDArray
    neuroid_ids: NDArray
    stimulus_df: pd.DataFrame
    group_col: str
    context_group_col: str
    adapter_name: str
    reductions: list[dict]


def get_adapter_spec(benchmark_id: str) -> NeuralAdapterSpec:
    for spec in _ADAPTERS:
        if benchmark_id.startswith(spec.benchmark_prefix):
            return spec
    raise RuntimeError(
        f"No benchmark-specific neural adapter is registered for {benchmark_id!r}. "
        "Implement and test an adapter before running inferential analysis."
    )


def resolve_layer_index(n_layers: int, spec: str | int) -> int:
    if n_layers <= 0:
        raise ValueError("n_layers must be positive.")
    aliases = {"first": 0, "middle": n_layers // 2, "last": n_layers - 1}
    if isinstance(spec, str) and spec in aliases:
        return aliases[spec]
    if isinstance(spec, str):
        match = re.fullmatch(r"(?:transformer\.h|model\.layers)\.(\d+)", spec)
        if match:
            spec = match.group(1)
    try:
        index = int(spec)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"layer spec must be 'first', 'middle', 'last', or an integer; got {spec!r}."
        ) from exc
    if not 0 <= index < n_layers:
        raise ValueError(f"layer index {index} out of range for a {n_layers}-layer model.")
    return index


def _pool(hidden: NDArray, mask: NDArray, pooling: str) -> NDArray:
    if pooling not in _POOLINGS:
        raise ValueError(f"pooling must be one of {_POOLINGS}; got {pooling!r}.")
    valid = np.asarray(mask).astype(bool)
    if valid.ndim != 1 or hidden.ndim != 2 or len(valid) != len(hidden):
        raise ValueError("Hidden states and attention mask have incompatible shapes.")
    if not valid.any():
        raise ValueError("A tokenized stimulus contains no non-padding tokens.")
    selected = np.asarray(hidden)[valid]
    if pooling == "last":
        return selected[-1]
    if pooling == "mean":
        return selected.mean(axis=0)
    return selected.sum(axis=0)


def build_contexts(
    sentences: Iterable[str],
    groups: NDArray,
    *,
    context_mode: str,
    separator: str = " "
) -> list[str]:
    """Build contexts with the pinned Brain-Score punctuation normalization.

    Pereira uses ``passage_label`` for cumulative context construction and
    ``story`` for grouped cross-validation. Callers must therefore pass the
    context grouping here, not the CV grouping.
    """
    sentence_list = [str(sentence) for sentence in sentences]
    group_array = np.asarray(groups)
    if len(sentence_list) != len(group_array):
        raise ValueError("sentences and groups must have equal length.")
    if context_mode not in _CONTEXT_MODES:
        raise ValueError(f"context_mode must be one of {_CONTEXT_MODES}.")
    def prepare(parts: list[str]) -> str:
        # matches brainscore_language's prepare_context: only exact empty strings
        # are dropped, whitespace-only parts remain
        nonempty = [part for part in parts if part != ""]
        context = separator.join(nonempty)
        return re.sub(r"\s+([.,!?;:])", r"\1", context)

    if context_mode == "independent_sentence":
        return [prepare([sentence]) for sentence in sentence_list]
    contexts: list[str] = []
    current_group: object | None = None
    pieces: list[str] = []
    closed_groups: set[object] = set()
    for sentence, group in zip(sentence_list, group_array):
        if current_group is None or group != current_group:
            if group in closed_groups:
                raise ValueError(
                    f"Group {group!r} is non-contiguous. Cumulative context requires "
                    "presentations from each passage/story to be contiguous and ordered."
                )
            if current_group is not None:
                closed_groups.add(current_group)
            current_group = group
            pieces = []
        pieces.append(sentence)
        contexts.append(prepare(pieces))
    return contexts


def _model_blocks(model):
    """Return the transformer block sequence used by GPT-style causal LMs."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(
        "Could not locate transformer blocks. Add an architecture-specific adapter "
        "rather than falling back to ambiguous output_hidden_states indices."
    )


def _hook_output_tensor(output):
    value = output[0] if isinstance(output, (tuple, list)) else output
    if not hasattr(value, "detach") or getattr(value, "ndim", None) != 3:
        raise ValueError("Transformer block hook did not return a [batch, token, unit] tensor.")
    return value


def extract_hidden_states(
    model,
    tokenizer,
    sentences: list[str],
    layer: str | int = "last",
    pooling: str = "last",
    batch_size: int = 16,
    device: str | None = None,
    *,
    groups: NDArray | None = None,
    context_mode: str = "independent_sentence",
    separator: str = " "
) -> NDArray[np.float32]:
    """Extract pooled hidden states under an explicitly declared context mode."""
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if context_mode == "cumulative_passage" and groups is None:
        raise ValueError("groups is required for cumulative_passage context.")
    effective_groups = np.arange(len(sentences)) if groups is None else np.asarray(groups)
    contexts = build_contexts(
        sentences,
        effective_groups,
        context_mode=context_mode,
        separator=separator
    )
    _ensure_pad_token(tokenizer)

    chosen_device = resolve_model_device(device)
    model = model.to(chosen_device).eval()
    blocks = _model_blocks(model)
    block_index = resolve_layer_index(len(blocks), layer)

    # patched-model callers clear BRAIN_CODE_CACHE_KEY before each call so
    # transplanted-embedding activations are never served from the baseline cache
    cache_env = _activation_cache_env()
    cache_file: Path | None = None
    if cache_env is not None:
        cache_file = _activation_cache_path(
            *cache_env,
            context_mode,
            pooling,
            block_index,
            model_fingerprint=_model_fingerprint(model),
            inputs_hash=_inputs_hash(contexts)
        )
        if cache_file.exists():
            try:
                cached = np.load(cache_file)
                if cached.ndim == 2 and cached.shape[0] == len(sentences):
                    return cached.astype(np.float32)
            except Exception:
                pass  # a corrupt or truncated file falls through to the forward pass

    # the official subject records the module output of transformer.h.N, which
    # for the final GPT-2 block differs from hidden_states[-1] (already past ln_f)
    captured = {}

    def hook(_module, _inputs, output):
        captured["value"] = _hook_output_tensor(output)

    outputs: list[NDArray] = []
    original_truncation_side = getattr(tokenizer, "truncation_side", None)
    original_padding_side = getattr(tokenizer, "padding_side", None)
    tokenizer.truncation_side = "left"
    # the official subject tokenizes one context at a time; right padding keeps
    # valid token positions at 0..n-1 when contexts are batched
    tokenizer.padding_side = "right"
    handle = blocks[block_index].register_forward_hook(hook)
    try:
        with torch.no_grad():
            for start in range(0, len(contexts), batch_size):
                texts = contexts[start : start + batch_size]
                encoded = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    add_special_tokens=True
                )
                encoded = {key: value.to(chosen_device) for key, value in encoded.items()}
                captured.clear()
                model(**encoded, return_dict=True)
                if "value" not in captured:
                    raise RuntimeError("Transformer block hook did not fire.")
                hidden = captured["value"].detach().float().cpu().numpy()
                mask = encoded["attention_mask"].detach().cpu().numpy()
                for row in range(len(texts)):
                    outputs.append(_pool(hidden[row], mask[row], pooling))
    finally:
        handle.remove()
        if original_truncation_side is not None:
            tokenizer.truncation_side = original_truncation_side
        if original_padding_side is not None:
            tokenizer.padding_side = original_padding_side
    array = np.asarray(outputs, dtype=np.float32)
    if array.shape[0] != len(sentences):
        raise AssertionError("Representation extraction changed presentation count.")
    if cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, array)
        except Exception:
            pass  # a failed cache write must not lose the extracted activations
    return array


def extract_hidden_states_multi(
    model,
    tokenizer,
    sentences: list[str],
    *,
    layers: Iterable[str | int] = ("last",),
    poolings: str | Iterable[str] = ("last",),
    batch_size: int = 16,
    device: str | None = None,
    groups: NDArray | None = None,
    context_mode: str = "independent_sentence"
) -> dict[tuple[int, str], NDArray[np.float32]]:
    """Extract multiple block outputs in one forward pass per batch.

    The returned keys use integer block indices, making aliases such as
    ``middle`` and official layer paths such as ``transformer.h.5`` explicit.
    """
    import torch

    blocks = _model_blocks(model)
    layer_values = list(layers)
    if len(layer_values) == 1 and str(layer_values[0]) == "all":
        indices = list(range(len(blocks)))
    else:
        indices = []
        for value in layer_values:
            index = resolve_layer_index(len(blocks), value)
            if index not in indices:
                indices.append(index)
    pooling_values = [poolings] if isinstance(poolings, str) else list(poolings)
    invalid_poolings = sorted(set(pooling_values) - set(_POOLINGS))
    if invalid_poolings:
        raise ValueError(f"Unsupported pooling values: {invalid_poolings}.")
    if context_mode == "cumulative_passage" and groups is None:
        raise ValueError("groups is required for cumulative_passage context.")
    effective_groups = np.arange(len(sentences)) if groups is None else np.asarray(groups)
    contexts = build_contexts(sentences, effective_groups, context_mode=context_mode)

    # all-or-nothing: the forward pass is skipped only when every
    # (layer, pooling) array is cached
    cache_env = _activation_cache_env()
    cache_files: dict[tuple[int, str], Path] | None = None
    if cache_env is not None:
        fingerprint = _model_fingerprint(model)
        inputs_hash = _inputs_hash(contexts)
        cache_files = {
            (idx, pol): _activation_cache_path(
                *cache_env,
                context_mode,
                pol,
                idx,
                model_fingerprint=fingerprint,
                inputs_hash=inputs_hash
            )
            for idx in indices
            for pol in pooling_values
        }
        if all(cf.exists() for cf in cache_files.values()):
            cached_result: dict[tuple[int, str], NDArray[np.float32]] = {}
            all_valid = True
            for (idx, pol), cf in cache_files.items():
                try:
                    arr = np.load(cf)
                    if arr.ndim == 2 and arr.shape[0] == len(sentences):
                        cached_result[(idx, pol)] = arr.astype(np.float32)
                    else:
                        all_valid = False
                        break
                except Exception:
                    all_valid = False
                    break
            if all_valid:
                return cached_result

    _ensure_pad_token(tokenizer)
    chosen_device = resolve_model_device(device)
    model = model.to(chosen_device).eval()
    captured: dict[int, object] = {}
    handles = []
    for index in indices:
        def make_hook(block_index):
            def hook(_module, _inputs, output):
                captured[block_index] = _hook_output_tensor(output)
            return hook
        handles.append(blocks[index].register_forward_hook(make_hook(index)))
    rows: dict[tuple[int, str], list[NDArray]] = {
        (index, pooling): [] for index in indices for pooling in pooling_values
    }
    original_truncation_side = getattr(tokenizer, "truncation_side", None)
    original_padding_side = getattr(tokenizer, "padding_side", None)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"
    try:
        with torch.no_grad():
            for start in range(0, len(contexts), batch_size):
                texts = contexts[start : start + batch_size]
                encoded = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    add_special_tokens=True
                )
                encoded = {key: value.to(chosen_device) for key, value in encoded.items()}
                captured.clear()
                model(**encoded, return_dict=True)
                missing = sorted(set(indices) - set(captured))
                if missing:
                    raise RuntimeError(f"Transformer hooks did not fire for blocks {missing}.")
                mask = encoded["attention_mask"].detach().cpu().numpy()
                for index in indices:
                    hidden = captured[index].detach().float().cpu().numpy()
                    for row in range(len(texts)):
                        for pooling in pooling_values:
                            rows[(index, pooling)].append(_pool(hidden[row], mask[row], pooling))
    finally:
        for handle in handles:
            handle.remove()
        if original_truncation_side is not None:
            tokenizer.truncation_side = original_truncation_side
        if original_padding_side is not None:
            tokenizer.padding_side = original_padding_side
    multi_result = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in rows.items()
    }
    if cache_files is not None:
        for key, arr in multi_result.items():
            cf = cache_files[key]
            try:
                cf.parent.mkdir(parents=True, exist_ok=True)
                np.save(cf, arr)
            except Exception:
                pass  # a failed cache write must not lose the extracted activations
    return multi_result


def load_model_reps(
    hf_model_id: str,
    sentences: list[str],
    layer: str | int = "last",
    pooling: str = "last",
    batch_size: int = 16,
    device: str | None = None,
    *,
    groups: NDArray | None = None,
    context_mode: str = "independent_sentence",
    model_revision: str | None = None,
    tokenizer_revision: str | None = None,
    trust_remote_code: bool = False
) -> NDArray:
    from brain_code.schema import require_immutable_revision

    model_revision = require_immutable_revision(
        model_revision, field_name="model_revision"
    )
    tokenizer_revision = require_immutable_revision(
        tokenizer_revision or model_revision, field_name="tokenizer_revision"
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_id,
        revision=tokenizer_revision or model_revision,
        trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        revision=model_revision,
        trust_remote_code=trust_remote_code
    )
    return extract_hidden_states(
        model,
        tokenizer,
        sentences,
        layer,
        pooling,
        batch_size,
        device,
        groups=groups,
        context_mode=context_mode
    )


def _find_name(candidates: tuple[str, ...], available: Iterable[str], *, kind: str) -> str:
    available_set = set(available)
    matches = [candidate for candidate in candidates if candidate in available_set]
    if not matches:
        raise RuntimeError(
            f"No supported {kind} found. Tried {candidates}; available={sorted(available_set)}."
        )
    return matches[0]


def _all_coord_names(assembly) -> tuple[str, ...]:
    """All retrievable coordinate names, including MultiIndex level names.

    Brain-Score assemblies pack stimulus/neuroid metadata into pandas MultiIndex
    levels on the ``presentation``/``neuroid`` dims. Under the pinned xarray those
    levels are retrievable via ``assembly[name]`` but do NOT appear in
    ``assembly.coords``; enumerating only ``.coords`` therefore hides them.
    """
    names = {str(name) for name in assembly.coords}
    indexes = getattr(assembly, "indexes", None)
    if indexes is not None:
        for dim in getattr(assembly, "dims", ()):
            try:
                index = indexes[dim]
            except (KeyError, TypeError):
                continue
            level_names = getattr(index, "names", None)
            if level_names:
                names.update(str(level) for level in level_names if level is not None)
    return tuple(sorted(names))


def _coord_vector(assembly, name: str, presentation_dim: str) -> NDArray:
    # assembly[name], unlike assembly.coords[name], also resolves MultiIndex levels
    coord = assembly[name]
    if tuple(coord.dims) != (presentation_dim,):
        raise RuntimeError(
            f"Coordinate {name!r} must vary only over {presentation_dim!r}; "
            f"got dims {tuple(coord.dims)}."
        )
    return np.asarray(coord.values)


def adapt_neural_assembly(assembly, benchmark_id: str) -> BenchmarkData:
    """Converts an xarray/BrainIO assembly using a benchmark-specific spec."""
    spec = get_adapter_spec(benchmark_id)
    dims = tuple(str(dim) for dim in assembly.dims)
    presentation_dim = _find_name(spec.presentation_dims, dims, kind="presentation dimension")
    neuroid_dim = _find_name(spec.neuroid_dims, dims, kind="neuroid dimension")

    extra_dims = [dim for dim in dims if dim not in {presentation_dim, neuroid_dim}]
    reductions: list[dict] = []
    working = assembly
    for dim in extra_dims:
        if dim not in spec.permitted_singleton_dims:
            raise RuntimeError(
                f"Unsupported neural dimension {dim!r} for {benchmark_id!r}. "
                "Subjects, repetitions, and time may not be implicitly averaged."
            )
        if int(working.sizes[dim]) != 1:
            raise RuntimeError(
                f"Permitted dimension {dim!r} is not singleton (size={working.sizes[dim]})."
            )
        working = working.squeeze(dim, drop=True)
        reductions.append({"dimension": dim, "operation": "squeeze_singleton"})

    working = working.transpose(presentation_dim, neuroid_dim)
    values = np.asarray(working.values, dtype=np.float64)
    if values.ndim != 2:
        raise RuntimeError(
            f"Adapter did not produce a 2-D response matrix (ndim={values.ndim}): {values.shape}."
        )
    if not np.isfinite(values).all():
        raise RuntimeError("Neural response matrix contains NaN or infinity after filtering.")

    coord_names = _all_coord_names(working)
    text_coord = _find_name(spec.text_coords, coord_names, kind="stimulus coordinate")
    group_coord = _find_name(spec.group_coords, coord_names, kind="CV grouping coordinate")
    context_candidates = spec.context_group_coords or spec.group_coords
    context_group_coord = _find_name(
        context_candidates, coord_names, kind="context grouping coordinate"
    )
    # custom folds that differ from the official ones would void both the parity
    # and the leakage guarantees
    if spec.official_split_coord is not None and group_coord != spec.official_split_coord:
        raise RuntimeError(
            f"Resolved grouping coordinate {group_coord!r} does not match the official "
            f"benchmark split_coord {spec.official_split_coord!r} for {benchmark_id!r}. "
            f"Grouped cross-validation would diverge from official fold membership. "
            f"Available coordinates: {sorted(coord_names)}."
        )
    if (
        spec.official_context_coord is not None
        and context_group_coord != spec.official_context_coord
    ):
        raise RuntimeError(
            f"Resolved context coordinate {context_group_coord!r} does not match the "
            f"official context coordinate {spec.official_context_coord!r} for "
            f"{benchmark_id!r}. Cumulative-context activations would diverge from "
            f"the official benchmark. Available coordinates: {sorted(coord_names)}."
        )
    sentences = [str(value) for value in _coord_vector(working, text_coord, presentation_dim)]
    groups = _coord_vector(working, group_coord, presentation_dim)
    context_groups = _coord_vector(working, context_group_coord, presentation_dim)

    id_matches = [name for name in spec.stimulus_coords if name in coord_names]
    if not id_matches:
        raise RuntimeError(
            "No stable stimulus identifier coordinate is available. Row positions may not "
            "be used as scientific stimulus IDs."
        )
    raw_stimulus_ids = _coord_vector(working, id_matches[0], presentation_dim)
    if (
        len(sentences) != values.shape[0]
        or len(groups) != values.shape[0]
        or len(context_groups) != values.shape[0]
    ):
        raise RuntimeError(
            "Stimulus, CV group, context group, and response presentation counts do not match."
        )
    stim_to_groups: dict[str, set] = {}
    for sid, grp in zip(map(str, raw_stimulus_ids), map(str, groups)):
        stim_to_groups.setdefault(sid, set()).add(grp)
    multi_group_stims = {
        sid: sorted(gps) for sid, gps in stim_to_groups.items() if len(gps) > 1
    }
    if multi_group_stims:
        examples = list(multi_group_stims.items())[:3]
        raise RuntimeError(
            f"Stimulus IDs appear in more than one group, indicating passage-assignment "
            f"inconsistency. Affected: {examples}"
        )
    stable_ids = np.asarray([
        f"{benchmark_id}:{group}:{value}"
        for group, value in zip(groups, raw_stimulus_ids)
    ])
    if len(np.unique(stable_ids)) != len(stable_ids):
        raise RuntimeError(
            "Stimulus identifiers are not unique after combining benchmark, group, and ID."
        )

    frame_data = {
        "stimulus_id": stable_ids,
        "raw_stimulus_id": np.asarray(raw_stimulus_ids),
        "sentence": sentences,
        group_coord: groups,
        "presentation_index": np.arange(len(sentences), dtype=np.int64)
    }
    if context_group_coord != group_coord:
        frame_data[context_group_coord] = context_groups
    stimulus_df = pd.DataFrame(frame_data)

    if "neuroid_id" in coord_names:
        neuroid_coord = working.coords["neuroid_id"]
        if tuple(neuroid_coord.dims) != (neuroid_dim,):
            raise RuntimeError(
                f"Coordinate 'neuroid_id' must vary only over {neuroid_dim!r}; "
                f"got dims {tuple(neuroid_coord.dims)}."
            )
        neuroid_ids = np.asarray(neuroid_coord.values)
    else:
        neuroid_ids = np.arange(values.shape[1], dtype=np.int64)
    return BenchmarkData(
        sentences=sentences,
        responses=values,
        groups=np.asarray(groups),
        context_groups=np.asarray(context_groups),
        stimulus_ids=stable_ids,
        raw_stimulus_ids=np.asarray(raw_stimulus_ids),
        neuroid_ids=np.asarray(neuroid_ids),
        stimulus_df=stimulus_df,
        group_col=group_coord,
        context_group_col=context_group_coord,
        adapter_name=spec.benchmark_prefix,
        reductions=reductions
    )


def load_neural_benchmark(benchmark_id: str) -> BenchmarkData:
    try:
        from brainscore_language import load_benchmark
    except ImportError as exc:
        raise RuntimeError(
            "brainscore_language is not installed. Run: uv pip install -e language/"
        ) from exc
    benchmark = load_benchmark(benchmark_id)
    assembly = getattr(benchmark, "data", None)
    if assembly is None:
        raise RuntimeError(f"Benchmark {benchmark_id!r} exposes no .data assembly.")
    return adapt_neural_assembly(assembly, benchmark_id)


def load_reps_and_neural(
    hf_model_id: str,
    benchmark_id: str,
    layer: str | int = "last",
    pooling: str = "last",
    batch_size: int = 16,
    device: str | None = None,
    *,
    context_mode: str = "independent_sentence",
    model_revision: str | None = None,
    tokenizer_revision: str | None = None
) -> tuple[NDArray, BenchmarkData]:
    data = load_neural_benchmark(benchmark_id)
    reps = load_model_reps(
        hf_model_id,
        data.sentences,
        layer,
        pooling,
        batch_size,
        device,
        groups=data.context_groups,
        context_mode=context_mode,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision
    )
    return reps, data
