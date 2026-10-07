"""Shared Student-t 95% confidence interval across cross-validation folds.

The official reporting path (``scripts/extract_layer_cis.py``) forms each layer's
uncertainty as a Student-t 95% interval over the per-fold scores (the fold score
is the mean over neuroids; the CI is across folds). The custom encoding
analyses collect those same per-fold scores, so they reuse this
helper to report intervals the same way, keeping every CI in the project on one
method and directly comparable to the official numbers.
"""

from __future__ import annotations

import numpy as np

# Student-t 0.975 (two-sided 95%) critical values by degrees of freedom
# (df = n_folds - 1). 5 outer folds -> df=4; the official 10-fold path -> df=9.
_T975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042
}


def t975(df: int) -> float:
    """Student-t 0.975 quantile for ``df`` degrees of freedom.

    Falls back to the normal approximation (1.96) for df above the table
    (df > 30), where the t and normal quantiles already agree to ~0.02.
    Returns NaN for a nonpositive df (no interval is defined).
    """
    if df <= 0:
        return float("nan")
    return _T975.get(df, 1.96)


def fold_ci95(values) -> dict:
    """Student-t 95% CI of the mean across per-fold scores.

    Args:
        values: one score per CV fold (typically the mean over neuroids within
            that fold). Non-finite entries are dropped.

    Returns:
        dict with ``mean``, ``sem`` (standard error of the mean), ``ci_low``,
        ``ci_high``, and ``n_folds``. With fewer than two finite folds the SEM
        and interval are NaN (an interval is undefined), though ``mean`` is
        still returned when at least one fold is finite.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "mean": float("nan"),
            "sem": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_folds": 0
        }
    mean = float(np.mean(arr))
    if n < 2:
        return {
            "mean": mean,
            "sem": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_folds": n
        }
    sem = float(np.std(arr, ddof=1) / np.sqrt(n))
    half = t975(n - 1) * sem
    return {
        "mean": mean,
        "sem": sem,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "n_folds": n
    }
