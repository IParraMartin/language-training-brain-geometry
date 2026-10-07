"""Computes every reported statistic from the results bundle.

Each ``*_results`` function returns plain dicts and lists so the results can be written to JSON and
compared without recomputation.
"""

from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from brain_code.reproduction.bundle import Bundle
from brain_code.reproduction.config import (
    BENCHMARKS,
    CODE_CORPORA,
    CONTRASTS,
    CORPORA,
    N_BLOCKS,
    PRIMARY_BENCHMARKS
)
from brain_code.reproduction.inference import (
    exact_separation_p,
    holm,
    layer_tests,
    order_agreement,
    pairwise_mean,
    primary_contrasts
)

PREDICTED_ORDER = ["python", "source", "assembly", "binary"]
PRIMARY_STEPS_EVALUATED = 30000
TOKENS_PER_UPDATE = 16 * 1024


def primary_results(bundle: Bundle) -> dict:
    """Depth summaries and the twelve-test family, with and without random-initialization variance."""
    out = {"benchmarks": {}, "family": {}}
    for bench in BENCHMARKS:
        grid = bundle.scores(bench)
        summaries = {c: grid.depth_summary(c).tolist() for c in CORPORA}
        summaries["random"] = grid.depth_summary("random").tolist()
        code_mean_by_seed = np.mean([grid.depth_summary(c) for c in CODE_CORPORA], axis=0)
        out["benchmarks"][bench] = {
            "depth_summary": summaries,
            "seeds": grid.seeds["tinystories"],
            "random_seeds": grid.random_seeds,
            "means": {
                "language": float(np.mean(summaries["tinystories"])),
                "code": float(np.mean([np.mean(summaries[c]) for c in CODE_CORPORA])),
                "random": float(np.mean(summaries["random"])),
                **{c: float(np.mean(summaries[c])) for c in CORPORA}
            },
            "code_mean_by_seed": code_mean_by_seed.tolist(),
            "language_random_separation_p": exact_separation_p(
                grid.depth_summary("tinystories"),
                grid.depth_summary("random")
            )
        }
    for label, random_variance in (("sampled_reference", True), ("fixed_reference", False)):
        rows = []
        for bench in PRIMARY_BENCHMARKS:
            grid = bundle.scores(bench)
            tests = primary_contrasts(
                grid.depth_summary("tinystories"),
                {c: grid.depth_summary(c) for c in CODE_CORPORA},
                grid.depth_summary("random"),
                random_variance=random_variance
            )
            rows.extend({"benchmark": bench, "contrast": k, **v.as_dict()} for k, v in tests.items())
        adjusted = holm([r["p"] for r in rows])
        for row, p_holm in zip(rows, adjusted):
            row["p_holm"] = float(p_holm)
            row["significant"] = bool(p_holm < 0.05)
        out["family"][label] = rows
    out["contrasts"] = CONTRASTS
    return out


def layer_results(bundle: Bundle) -> dict:
    """Per-block corpus profiles, the language-code gap and code heterogeneity on every benchmark."""
    out = {}
    for bench in BENCHMARKS:
        grid = bundle.scores(bench)
        tests = layer_tests(grid.trained["tinystories"], grid.code)
        agreement = []
        for block in range(N_BLOCKS):
            means = {c: float(grid.trained[c][:, block].mean()) for c in CODE_CORPORA}
            rho, exact = order_agreement(means, PREDICTED_ORDER)
            agreement.append({"block": block, "spearman_rho": rho, "exact_predicted_order": exact})
        out[bench] = {
            "profile_mean": {c: grid.trained[c].mean(0).tolist() for c in CORPORA},
            "profile_sd": {c: grid.trained[c].std(0, ddof=1).tolist() for c in CORPORA},
            "random_profiles": grid.random.tolist(),
            "random_mean": grid.random.mean(0).tolist(),
            "code_mean_profile": np.mean([grid.trained[c].mean(0) for c in CODE_CORPORA], axis=0).tolist(),
            "gap": tests["gap"].tolist(),
            "p_gap_holm": tests["p_gap_holm"].tolist(),
            "gap_significant_blocks": int(np.sum(tests["p_gap_holm"] < 0.05)),
            "gap_nonsignificant_blocks": [int(b) for b in np.where(tests["p_gap_holm"] >= 0.05)[0]],
            "selected_max_gap": float(tests["gap"].max()),
            "selected_max_gap_block": int(np.argmax(tests["gap"])),
            "mean_gap": float(tests["gap"].mean()),
            "p_heterogeneity_holm": tests["p_heterogeneity_holm"].tolist(),
            "heterogeneity_significant_blocks": [int(b) for b in np.where(tests["p_heterogeneity_holm"] < 0.05)[0]],
            "leader_by_block": tests["leader"].tolist(),
            "order_agreement": agreement,
            "exact_order_blocks": int(sum(a["exact_predicted_order"] for a in agreement)),
            "language_above_random_blocks": int(np.sum(grid.trained["tinystories"].mean(0) > grid.random.mean(0))),
            "code_above_random_blocks": int(np.sum(
                np.mean([grid.trained[c].mean(0) for c in CODE_CORPORA], axis=0) > grid.random.mean(0)
            )),
            "mean_within_corpus_seed_sd": float(np.mean([grid.trained[c].std(0, ddof=1).mean() for c in CORPORA]))
        }
    return out


def _corpus(model: str) -> str:
    return model.rsplit("_seed", 1)[0].replace("random_init", "random")


def geometry_results(bundle: Bundle) -> dict:
    """Seed agreement, corpus separation, manifold dimensionality and fit-free brain RSA."""
    geo = bundle.table("geometry_multiseed")
    cka = bundle.table("cka_pairs")
    pairs = {
        block: {(r.model_a, r.model_b): r.cka for r in sub.itertuples()}
        for block, sub in cka.groupby("block")
    }
    models = sorted(set(cka["model_a"]) | set(cka["model_b"]))
    members = {c: [m for m in models if _corpus(m) == c] for c in CORPORA + ["random"]}

    def mean_pairs(block: int, a: list[str], b: list[str]) -> float:
        return pairwise_mean(pairs[block], a, b)

    within = {c: [mean_pairs(b, members[c], members[c]) for b in range(N_BLOCKS)] for c in members}
    between = {
        f"{a}|{b}": [mean_pairs(k, members[a], members[b]) for k in range(N_BLOCKS)]
        for a, b in combinations(CORPORA + ["random"], 2)
    }
    language_code_between = [
        float(np.mean([between[f"tinystories|{c}"][k] for c in CODE_CORPORA])) for k in range(N_BLOCKS)
    ]
    all_within = [float(np.mean([within[c][k] for c in CORPORA])) for k in range(N_BLOCKS)]
    all_between = [
        float(np.mean([between[f"{a}|{b}"][k] for a, b in combinations(CORPORA, 2)])) for k in range(N_BLOCKS)
    ]
    out = {
        "n_models": {c: len(members[c]) for c in members},
        "seed_cka_blocks_4_11": {c: float(np.mean(within[c][4:])) for c in members},
        "within_cka": within,
        "between_cka": between,
        "language_minus_language_code_cka": [
            within["tinystories"][k] - language_code_between[k]
            for k in range(N_BLOCKS)
        ],
        "within_minus_between_cka_all_corpora": [all_within[k] - all_between[k] for k in range(N_BLOCKS)],
        "benchmarks": {}
    }
    for bench, sub in geo.groupby("benchmark"):
        record = {}
        for corpus in CORPORA + ["random"]:
            rows = sub[sub["corpus"] == corpus]
            rsa = rows.pivot(index="seed", columns="block", values="rsa_pooled").sort_index().to_numpy()
            pr = rows.pivot(index="seed", columns="block", values="participation_ratio").sort_index().to_numpy()
            er = rows.pivot(index="seed", columns="block", values="effective_rank").sort_index().to_numpy()
            z = rows.pivot(index="seed", columns="block", values="rsa_null_z").sort_index().to_numpy()
            profile = rsa.mean(0)
            record[corpus] = {
                "n": int(rsa.shape[0]),
                "rsa_mean": profile.tolist(),
                "rsa_min": rsa.min(0).tolist(),
                "rsa_max": rsa.max(0).tolist(),
                "rsa_peak": float(profile.max()),
                "rsa_peak_block": int(profile.argmax()),
                "participation_ratio_mean": pr.mean(0).tolist(),
                "participation_ratio_min": pr.min(0).tolist(),
                "participation_ratio_max": pr.max(0).tolist(),
                "effective_rank_mean": er.mean(0).tolist(),
                "null_z_mean": np.nanmean(z, axis=0).tolist() if np.isfinite(z).any() else None
            }
        record["brain_rdm_metric"] = str(sub["brain_rdm_metric"].iloc[0])
        out["benchmarks"][bench] = record
    return out


def pc_ablation_results(bundle: Bundle) -> dict:
    """Mean over seeds of each model's maximum RSA over blocks after removing m leading PCs."""
    frame = bundle.table("pc_ablation_rsa")
    out = {}
    for bench, sub in frame.groupby("benchmark"):
        per_model = sub.groupby(["corpus", "seed", "removed_pcs"])["rsa"].max().reset_index()
        table = per_model.groupby(["corpus", "removed_pcs"])["rsa"].mean().unstack()
        out[bench] = {c: {int(m): float(table.loc[c, m]) for m in table.columns} for c in table.index}
    return out


def transplant_results(bundle: Bundle) -> dict:
    """Progressive transplants read at fixed block 9, and the single-block transplant counts."""
    frame = bundle.table("progressive_transplant")
    design = bundle.json("progressive_transplant_design")
    at9 = frame[frame["block"] == 9]
    table = {
        f"{r}|{arm}": {int(k): float(v) for k, v in sub.set_index("k")["raw_r"].items()}
        for (r, arm), sub in at9.groupby(["receiver", "arm"])
    }
    single = bundle.table("single_block_transplant")
    patch_index = single["patch"].map(lambda p: -1 if p == "embeddings" else int(p.split("_")[1]))
    eligible = single[patch_index <= single["readout_block"]]
    sig = eligible[eligible["significant"].astype(str) == "True"]
    return {
        "readout_block": 9,
        "table": table,
        "donor_block9": float(table[f"{CODE_CORPORA[0]}|blocks_and_embeddings"][12]),
        "endpoint_checks": design["endpoint_checks"],
        "design": design["design"],
        "single_block": {
            "cells_eligible": int(len(eligible)),
            "cells_significant": int(len(sig)),
            "significant_cells": sig[[
                "receiver",
                "readout_block",
                "patch",
                "mean_difference"
            ]].to_dict("records")
        }
    }


def adaptation_results(bundle: Bundle) -> dict:
    """Embedding-only adaptation: selected peaks (official scores) and the prespecified depth mean."""
    frame = bundle.table("embedding_adaptation")
    rows = {}
    for receiver in CODE_CORPORA:
        sub = frame[frame["receiver"] == receiver]
        arms = {arm: g.sort_values("block") for arm, g in sub.groupby("arm")}
        official = {arm: g["raw_r_official"].to_numpy() for arm, g in arms.items()}
        folded = {arm: g["raw_r_fold_mean"].to_numpy() for arm, g in arms.items()}
        rows[receiver] = {
            "peak": {arm: float(v.max()) for arm, v in official.items()},
            "peak_block": {arm: int(v.argmax()) for arm, v in official.items()},
            "depth_mean": {arm: float(v.mean()) for arm, v in folded.items()},
            "block9": {arm: float(v[9]) for arm, v in folded.items()}
        }

    def paired(summary: str) -> dict:
        english = np.array([rows[r][summary]["english"] - rows[r][summary]["receiver"] for r in CODE_CORPORA])
        own = np.array([rows[r][summary]["own_corpus"] - rows[r][summary]["receiver"] for r in CODE_CORPORA])
        test = stats.ttest_rel(english, own)
        return {
            "english_delta": english.tolist(),
            "own_delta": own.tolist(),
            "mean_english_delta": float(english.mean()),
            "mean_own_delta": float(own.mean()),
            "t": float(test.statistic),
            "df": len(english) - 1,
            "p": float(test.pvalue)
        }

    return {
        "receivers": rows,
        "paired_peak": paired("peak"),
        "paired_depth_mean": paired("depth_mean")
    }


def factorial_results(bundle: Bundle) -> dict:
    """Endpoints and trajectories of the pseudoword and semantics-by-syntax runs."""
    frame = bundle.table("factorial_milestones")
    out = {}
    for study, sub in frame.groupby("study"):
        per_model = sub.groupby(["condition", "seed", "milestone_tokens"]).agg(
            mean_r=("raw_r", "mean"),
            max_r=("raw_r", "max"),
            eval_loss=("eval_loss", "first"),
            tokens=("actual_nonpadding_tokens", "first"),
            n_layer=("n_layer", "first"),
            n_embd=("n_embd", "first")
        ).reset_index()
        summary = per_model.groupby(["condition", "milestone_tokens"]).agg(
            mean_r=("mean_r", "mean"),
            mean_r_sd=("mean_r", "std"),
            max_r=("max_r", "mean"),
            eval_loss=("eval_loss", "mean"),
            n=("seed", "count")
        ).reset_index()
        out[study] = {
            "blocks_scored": sorted(int(b) for b in sub["block"].unique()),
            "architecture": {
                "n_layer": int(per_model["n_layer"].iloc[0]),
                "n_embd": int(per_model["n_embd"].iloc[0])
            },
            "milestones": sorted(int(m) for m in sub["milestone_tokens"].unique()),
            "summary": summary.to_dict("records"),
            "per_model": per_model.to_dict("records")
        }
    pseudoword = frame[frame["study"] == "pseudoword_gpt2_small"]
    end = pseudoword[pseudoword["milestone_tokens"] == pseudoword["milestone_tokens"].max()]
    by = end.groupby(["condition", "seed"])["raw_r"]
    for summary, values in (("mean_over_blocks", by.mean()), ("max_over_blocks", by.max())):
        a, c = values.loc["A"].to_numpy(), values.loc["C"].to_numpy()
        test = stats.ttest_ind(a, c, equal_var=False)
        out["pseudoword_gpt2_small"][f"endpoint_{summary}"] = {
            "intact": a.tolist(),
            "pseudoword": c.tolist(),
            "difference": float(a.mean() - c.mean()),
            "welch_t": float(test.statistic),
            "p": float(test.pvalue)
        }
    factorial = frame[frame["study"] == "factorial_gpt2_small"]
    cells = pd.concat([pseudoword[pseudoword["condition"] == "A"], factorial])
    out["factorial_gpt2_small"]["main_effects"] = _factorial_tests(cells)
    return out


def _factorial_tests(frame: pd.DataFrame) -> dict:
    """Syntax {A,Cp} vs {B,Dp} and semantics {A,B} vs {Cp,Dp}, Welch 6 vs 6, per milestone."""
    per_model = frame.groupby(["condition", "seed", "milestone_tokens"])["raw_r"].mean().reset_index()
    out = {}
    for milestone, sub in per_model.groupby("milestone_tokens"):
        value = {cond: g["raw_r"].to_numpy() for cond, g in sub.groupby("condition")}
        if not {"A", "B", "Cp", "Dp"} <= set(value):
            continue
        syntax = stats.ttest_ind(np.r_[value["A"], value["Cp"]], np.r_[value["B"], value["Dp"]], equal_var=False)
        semantics = stats.ttest_ind(np.r_[value["A"], value["B"]], np.r_[value["Cp"], value["Dp"]], equal_var=False)
        out[int(milestone)] = {
            "cell_means": {k: float(v.mean()) for k, v in value.items()},
            "cell_sd": {k: float(v.std(ddof=1)) for k, v in value.items()},
            "syntax_delta": float(np.r_[value["A"], value["Cp"]].mean() - np.r_[value["B"], value["Dp"]].mean()),
            "syntax_p": float(syntax.pvalue),
            "semantics_delta": float(np.r_[value["A"], value["B"]].mean() - np.r_[value["Cp"], value["Dp"]].mean()),
            "semantics_p": float(semantics.pvalue),
            "interaction": float((value["A"].mean() - value["B"].mean()) - (value["Cp"].mean() - value["Dp"].mean()))
        }
    return out


def corpus_results(bundle: Bundle) -> dict:
    """Corpus diagnostics and the five-corpus frequency regression against the selected seed-42 peak."""
    diag = bundle.table("corpus_diagnostics").set_index("corpus")
    official = bundle.table("official_seed42")
    peaks = official[official["benchmark"] == "pereira_243"].groupby("corpus")["raw_r"].max()
    freq = np.log10(diag.loc[CORPORA, "mean_corpus_count_per_stimulus_token"].to_numpy() + 1)
    reg = stats.linregress(freq, peaks.loc[CORPORA].to_numpy())
    return {
        "diagnostics": diag.reset_index().to_dict("records"),
        "seed42_selected_peak": peaks.to_dict(),
        "frequency_regression": {
            "r": float(reg.rvalue),
            "p": float(reg.pvalue),
            "n": len(CORPORA)
        }
    }


def exposure_results(bundle: Bundle) -> dict:
    """Realized exposure of the 25 primary runs from provenance records and recovered split sizes."""
    packed = bundle.json("packed_splits")["corpora"]
    provenance = bundle.table("training_provenance").set_index("run_id")
    rows = []
    for corpus in CORPORA:
        for seed in (42, 43, 44, 45, 46):
            split = packed.get(corpus, {}).get("seeds", {}).get(str(seed))
            run_id = f"{corpus}_seed{seed}"
            record = provenance.loc[run_id] if run_id in provenance.index else None
            train_blocks = split["train_blocks"] if split else None
            row = {
                "run_id": run_id if seed != 42 else f"IParraMartin/gpt2-{corpus}",
                "corpus": corpus,
                "seed": seed,
                "pipeline": split["pipeline"] if split else None,
                "train_sequences": train_blocks,
                "train_tokens_available": train_blocks * 1024 if train_blocks else None,
                "max_steps": 30517,
                "steps_completed": int(record["global_step"]) if record is not None else None,
                "evaluated_checkpoint_step": int(record["best_checkpoint_step"]) if record is not None else None,
                "stopping_reason": "max_steps" if record is not None and record["global_step"] == 30517 else None,
                "tokens_processed_total": (
                    int(record["global_step"]) * TOKENS_PER_UPDATE if record is not None else None
                ),
                "tokens_at_evaluated_checkpoint": (
                    int(record["best_checkpoint_step"]) * TOKENS_PER_UPDATE if record is not None else None
                )
            }
            if train_blocks and row["tokens_at_evaluated_checkpoint"]:
                tokens = row["tokens_at_evaluated_checkpoint"]
                row["passes_at_evaluated_checkpoint"] = tokens / row["train_tokens_available"]
            else:
                row["passes_at_evaluated_checkpoint"] = None
            if train_blocks:
                processed = PRIMARY_STEPS_EVALUATED * TOKENS_PER_UPDATE
                row["passes_if_cap_reached_at_30000"] = processed / (train_blocks * 1024)
            else:
                row["passes_if_cap_reached_at_30000"] = None
            rows.append(row)
    return {
        "runs": rows,
        "corpora": {
            c: {k: v for k, v in packed[c].items() if k != "seeds"} for c in CORPORA if c in packed
        }
    }


def all_results(bundle: Bundle) -> dict:
    """Every analysis, in one JSON-serializable dict."""
    return {
        "primary": primary_results(bundle),
        "layers": layer_results(bundle),
        "geometry": geometry_results(bundle),
        "pc_ablation": pc_ablation_results(bundle),
        "transplant": transplant_results(bundle),
        "adaptation": adaptation_results(bundle),
        "factorial": factorial_results(bundle),
        "corpus": corpus_results(bundle),
        "exposure": exposure_results(bundle),
        "benchmarks": bundle.json("benchmarks"),
        "corpus_audit": bundle.json("corpus_audit"),
        "single_checkpoint_spectrum": bundle.table("spectrum_single_checkpoint").drop(
            columns=["cumulative_variance"]
        ).to_dict("records")
    }


def to_json(results: dict) -> str:
    """Serializes results, mapping numpy scalars and NaN to plain JSON."""
    def default(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return None if np.isnan(value) else float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.bool_):
            return bool(value)
        raise TypeError(type(value))

    return json.dumps(
        results,
        indent=2,
        sort_keys=True,
        default=default
    )
