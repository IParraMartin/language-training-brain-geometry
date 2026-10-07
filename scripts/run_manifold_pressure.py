"""Measures the geometric pressure a training corpus puts on the representation manifold.

G1 profiles every layer's covariance spectrum and rank-correlates each metric with that layer's brain
alignment; G2 deletes the top-m principal directions at one prespecified layer and re-scores. The ablation
layer must be given explicitly (choosing it from the sweep would be test-set selection), so results are
written ``exploratory_only`` under results/spectrum/.

Example: brain-manifold --model IParraMartin/gpt2-python --model-revision <sha> --ablation-layer 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_DEFAULT_ALPHAS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5)


def _hash_array(value: np.ndarray) -> str:
    import hashlib

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _toy_inputs(n_layers: int, n_stimuli: int, seed: int):
    """Synthetic reps + responses so the CLI is exercisable with no downloads."""
    rng = np.random.default_rng(seed)
    d = 32
    reps = {}
    for layer in range(n_layers):
        # deeper layers get a steeper spectrum so the toy sweep has structure
        scale = np.arange(1, d + 1, dtype=np.float64) ** (-(0.4 + 0.15 * layer) / 2)
        reps[layer] = rng.standard_normal((n_stimuli, d)) * scale
    responses = reps[n_layers // 2][:, :6] @ rng.standard_normal((6, 12))
    responses += 0.5 * rng.standard_normal((n_stimuli, 12))
    groups = np.repeat(np.arange(n_stimuli // 4 + 1), 4)[:n_stimuli]
    return reps, responses, groups


def _extract_all_layers(args, sentences, context_groups):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from brain_code.experiments.data_loading import extract_hidden_states_multi
    from brain_code.schema import require_immutable_revision

    revision = require_immutable_revision(args.model_revision, field_name="model_revision")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision)

    multi = extract_hidden_states_multi(
        model,
        tokenizer,
        sentences,
        layers=("all",),
        poolings=(args.pooling,),
        batch_size=args.batch_size,
        device=args.device,
        groups=context_groups,
        context_mode=args.context_mode
    )
    reps = {layer: np.asarray(arr, dtype=np.float64) for (layer, _), arr in multi.items()}
    return reps, model, tokenizer


def main() -> None:
    args = _parse_args()

    from brain_code.experiments.manifold_pressure import (
        geometry_alignment_correlation,
        grouped_ridge_fold_scorer,
        layerwise_spectral_profile,
        pc_ablation_curve
    )
    from brain_code.experiments.split_manifests import fold_manifest_hash
    from brain_code.experiments.subspace_erasure import _folds_from_groups, _score_condition

    # inputs
    data = model = tokenizer = None
    if args.toy:
        reps_by_layer, responses, groups = _toy_inputs(args.toy_layers, args.toy_stimuli, args.seed)
        stimulus_count = responses.shape[0]
    else:
        from brain_code.experiments.data_loading import load_neural_benchmark

        data = load_neural_benchmark(args.benchmark)
        reps_by_layer, model, tokenizer = _extract_all_layers(
            args, data.sentences, data.context_groups
        )
        responses, groups = data.responses, data.groups
        stimulus_count = len(data.sentences)

    folds = _folds_from_groups(groups, args.n_outer_folds, args.seed)
    alpha_grid = np.asarray(_DEFAULT_ALPHAS, dtype=np.float64)

    # G1: geometry per layer, alignment per layer
    profiles = layerwise_spectral_profile(
        reps_by_layer, i_min=args.decay_i_min, i_max=args.decay_i_max
    )
    layer_scores = []
    for row in profiles:
        _, _, r_mean = _score_condition(
            reps_by_layer[row["layer"]],
            responses,
            folds,
            alpha_grid=alpha_grid,
            n_inner_folds=args.n_inner_folds,
            groups=groups
        )
        layer_scores.append(r_mean)
        row["alignment_r"] = float(r_mean)
        print(
            f"  layer {row['layer']:>2}  r={r_mean:+.4f}  "
            f"PR={row['participation_ratio']:6.2f}  "
            f"eff_rank={row['effective_rank']:6.2f}  "
            f"alpha={row['spectral_decay']:.3f}",
            flush=True
        )

    coupling = geometry_alignment_correlation(profiles, layer_scores)

    # G2: causal ablation at the prespecified layer
    ablation_layer = int(args.ablation_layer)
    if ablation_layer not in reps_by_layer:
        raise SystemExit(
            f"--ablation-layer {ablation_layer} not among extracted layers "
            f"{sorted(reps_by_layer)}"
        )
    n_features = reps_by_layer[ablation_layer].shape[1]
    m_grid = [m for m in args.m_grid if m <= n_features]

    print(
        f"\nAblating top-m PCs at prespecified layer {ablation_layer} "
        f"(d={n_features})",
        flush=True
    )
    curve = pc_ablation_curve(
        reps_by_layer[ablation_layer],
        responses,
        folds,
        m_grid,
        fold_scorer=grouped_ridge_fold_scorer(
            alpha_grid, args.n_inner_folds, groups=groups
        )
    )
    baseline = curve[0]["score"] if curve and curve[0]["m"] == 0 else float("nan")
    for row in curve:
        row["score_retained"] = (
            float(row["score"] / baseline) if np.isfinite(baseline) and baseline else float("nan")
        )
        print(
            f"  m={row['m']:>3}  r={row['score']:+.4f}  "
            f"retained={row['score_retained']:6.1%}  "
            f"var_removed={row['variance_removed']:6.1%}",
            flush=True
        )

    # provenance
    from brain_code.experiments.brain_score_harness import get_benchmark_meta
    from brain_code.schema import hash_json, make_base_result, write_result

    if args.toy:
        # A synthetic run must never be renderable: `partial` is a blocked status.
        provenance = {"result_status": "partial", "conclusion_status": "exploratory_only"}
        run_id = f"toy_{args.benchmark}_manifold"
    else:
        provenance = {
            "result_status": "exploratory_only",
            "conclusion_status": "exploratory_only",
            "model_revision": args.model_revision,
            "tokenizer_revision": args.model_revision,
            "model_config_hash": hash_json(model.config.to_dict()),
            "checkpoint_hash": f"hf_revision:{args.model_revision}",
            "tokenizer_hash": hash_json({
                "vocab": tokenizer.get_vocab(),
                "special_tokens": tokenizer.special_tokens_map,
                "init_kwargs": tokenizer.init_kwargs
            }),
            "benchmark_modality": get_benchmark_meta(args.benchmark)["modality"],
            "benchmark_data_checksum": _hash_array(data.responses),
            "stimulus_ids_hash": hash_json(sorted(map(str, data.stimulus_ids.tolist()))),
            "stimulus_order_hash": hash_json(list(map(str, data.stimulus_ids.tolist()))),
            "split_manifest_hash": fold_manifest_hash(
                data.stimulus_df,
                data.group_col,
                args.n_outer_folds,
                args.seed
            ),
            "grouping_coordinate": data.group_col
        }
        run_id = f"{args.model.replace('/', '-')}_{args.benchmark}_manifold"

    result = make_base_result(
        experiment_id="exp12_manifold_pressure",
        run_id=run_id,
        model_identifier=args.model,
        benchmark_identifier=args.benchmark,
        repo_root=PROJECT_ROOT,
        tokenizer_identifier=args.model,
        config_hash=hash_json(vars(args)),
        unit_test_status="passed",
        layer=ablation_layer,
        layer_selection_method="prespecified",
        context_mode=args.context_mode,
        pooling=args.pooling,
        regression_method="ridge_oof_grouped",
        alpha_grid=list(_DEFAULT_ALPHAS),
        alpha_selection_strategy="per_target_inner_cv",
        inner_cv_definition=f"{args.n_inner_folds}-fold inner CV on training rows",
        outer_cv_definition=f"{args.n_outer_folds}-fold grouped outer CV, seed={args.seed}",
        primary_metric="pearson_r_oof",
        aggregation_method="mean_over_neuroids",
        random_seeds={"folds": args.seed},
        completeness="complete",
        layer_geometry=_jsonable(profiles),
        geometry_alignment_coupling=_jsonable(coupling),
        pc_ablation_curve=_jsonable(curve),
        ablation_m_grid=list(m_grid),
        n_stimuli=int(stimulus_count),
        n_neuroids=int(responses.shape[1]),
        spectral_decay_fit_range=[args.decay_i_min, args.decay_i_max],
        warnings=[
            "Layer sweep is descriptive; no layer is selected by score.",
            "PC ablation lacks a variance-matched random-direction control; "
            "a causal claim requires one (see subspace_erasure.build_matched_control)."
        ],
        **provenance
    )

    result["benchmark_version"] = result.get("brainscore_submodule_commit")

    out = Path(args.out or (PROJECT_ROOT / "results" / "spectrum" / f"{result['run_id']}.json"))
    write_result(out, result)
    print(f"\nwrote {out}")

    print("\ngeometry vs alignment (Spearman rho across layers):")
    for metric, stat in coupling.items():
        print(f"  {metric:>22}  rho={stat['rho']:+.3f}  (n={stat['n']})")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="IParraMartin/gpt2-python")
    p.add_argument("--model-revision", "--model_revision", dest="model_revision")
    p.add_argument("--benchmark", default="Pereira2018.243sentences-ridge")
    p.add_argument("--pooling", choices=["last", "mean", "sum"], default="last")
    p.add_argument(
        "--context-mode",
        "--context_mode",
        dest="context_mode",
        default="independent_sentence"
    )
    p.add_argument(
        "--ablation-layer",
        "--ablation_layer",
        dest="ablation_layer",
        type=int,
        default=0,
        help="PRESPECIFIED layer for the PC ablation; never chosen by score."
    )
    p.add_argument(
        "--m-grid",
        "--m_grid",
        dest="m_grid",
        type=int,
        nargs="+",
        default=[0, 1, 2, 4, 8, 16, 32, 64],
        help="Numbers of leading PCs to delete (must start at 0 for a baseline)."
    )
    p.add_argument(
        "--decay-i-min",
        "--decay_i_min",
        dest="decay_i_min",
        type=int,
        default=1
    )
    p.add_argument(
        "--decay-i-max",
        "--decay_i_max",
        dest="decay_i_max",
        type=int,
        default=None
    )
    p.add_argument(
        "--n-outer-folds",
        "--n_outer_folds",
        dest="n_outer_folds",
        type=int,
        default=5
    )
    p.add_argument(
        "--n-inner-folds",
        "--n_inner_folds",
        dest="n_inner_folds",
        type=int,
        default=4
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=16
    )
    p.add_argument("--device")
    p.add_argument("--out")
    p.add_argument(
        "--toy",
        action="store_true",
        help="Run on synthetic data: no model download, no benchmark, CPU only."
    )
    p.add_argument(
        "--toy-layers",
        "--toy_layers",
        dest="toy_layers",
        type=int,
        default=6
    )
    p.add_argument(
        "--toy-stimuli",
        "--toy_stimuli",
        dest="toy_stimuli",
        type=int,
        default=120
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
