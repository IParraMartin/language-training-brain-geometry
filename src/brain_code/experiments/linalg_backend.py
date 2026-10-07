"""GPU-accelerated linear algebra for the encoding models.

The encoding experiments spend nearly all of their wall time in ridge regression,
and overwhelmingly in *alpha selection*: every inner fold refits the model once
per candidate alpha.  At Pereira-243 shapes (243 stimuli x 768 features x 8031
regions) one ``select_alpha`` call costs ~2.6 s on CPU, and the LEACE control search pays that once
per control family per control seed per fold — thousands of times per run.

Two things make this fast:

1. **Eigendecomposition of the Gram matrix.**  Only the diagonal changes with
   alpha, so ``(G + aI)^-1 = V diag(1/(s + a)) V^T`` lets one ``eigh`` serve the
   whole alpha path.  Each additional alpha then costs a matmul instead of a
   full solve.  This alone is the bulk of the speedup and helps on CPU too.
2. **Keeping the operands resident on the GPU.**  Transferring per call is
   slower than staying on CPU — a single ridge fit round-trips more data than it
   saves.  The win only appears when one transfer is amortized over the whole
   fold x alpha sweep.

Measured on an RTX 5070 at the shapes above: ``select_alpha`` drops from 2.62 s
to 0.28 s (9.5x) and picks the identical alpha.

Everything runs in **float64**.  Consumer cards have poor float64 throughput, but
these solves are small and memory-bound rather than FLOP-bound, so float64 costs
roughly 2x float32 here while keeping agreement with the CPU path at ~1e-15.
Scientific results should not depend on the device they were computed on.

Device selection honours ``BRAIN_CODE_DEVICE`` (``cuda``, ``cpu``, or a specific
``cuda:N``); otherwise CUDA is used when available.  Every function falls back to
NumPy when torch or CUDA is missing, so nothing here is a hard dependency.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
from numpy.typing import NDArray


# below this many response-matrix elements (n_rows x n_regions) the host-to-device
# transfer costs more than the solve saves; on an RTX 5070, 100 regions x 243 stimuli
# runs 0.75x on GPU while 8031 regions runs 3.3x faster. override with
# BRAIN_CODE_GPU_MIN_ELEMS
_GPU_MIN_ELEMENTS = 50_000


def _gpu_worth_it(Y: NDArray) -> bool:
    """Whether a problem is large enough for the GPU to pay off.

    Args:
        Y: Target matrix, whose size drives the transfer cost.

    Returns:
        bool: True when the response matrix clears the size threshold.
    """
    threshold = int(os.environ.get("BRAIN_CODE_GPU_MIN_ELEMS", _GPU_MIN_ELEMENTS))
    return int(np.asarray(Y).size) >= threshold


def resolve_device() -> str:
    """Device for the float64 ridge solves: ``'cuda'``/``'cuda:N'`` or ``'cpu'``.

    The ridge path runs in **float64** and uses ``torch.linalg.eigh``. Apple
    ``mps`` supports neither float64 nor ``eigh``, so it is **not** a usable ridge
    accelerator: an ``mps`` request maps to ``cpu`` here (the model forward pass
    uses mps separately — see :func:`resolve_model_device`). Honours
    ``BRAIN_CODE_DEVICE`` otherwise; automatic selection is CUDA when available,
    else CPU.

    Returns:
        str: ``'cuda'``/``'cuda:N'`` or ``'cpu'``.
    """
    override = os.environ.get("BRAIN_CODE_DEVICE")
    if override:
        return "cpu" if override.startswith("mps") else override
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def resolve_model_device(override: str | None = None) -> str:
    """Device for the model forward pass (activation extraction): cuda / mps / cpu.

    Unlike the float64 ridge path, transformer inference runs in the model's
    native precision (float32), which Apple ``mps`` fully supports — so mps is a
    genuine accelerator here. Resolution order: an explicit ``override`` argument
    (e.g. a CLI ``--device``), then ``BRAIN_CODE_DEVICE``, then autodetect
    ``cuda`` -> ``mps`` -> ``cpu``.

    Args:
        override: Explicit device string, or ``None`` to autodetect.

    Returns:
        str: A torch device string suitable for ``model.to(...)``.
    """
    if override:
        return override
    env = os.environ.get("BRAIN_CODE_DEVICE")
    if env:
        return env
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def gpu_available() -> bool:
    """True when the backend will actually run on a CUDA device."""
    if not resolve_device().startswith("cuda"):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def device_report() -> str:
    """One-line description of where the ridge solves will run.

    Returns:
        str: Human-readable device summary for run logs.
    """
    model_dev = resolve_model_device()
    if gpu_available():
        try:
            import torch
            name = torch.cuda.get_device_name(0)
        except Exception:                                   # pragma: no cover
            name = "unknown CUDA device"
        ridge = (
            f"ridge backend: {resolve_device()} ({name}); "
            f"problems under {_GPU_MIN_ELEMENTS:,} response elements stay on CPU"
        )
    else:
        ridge = f"ridge backend: cpu ({resolve_device()})"
    if model_dev.startswith("mps"):
        return ridge + " | model forward: mps (float32); ridge on CPU (mps lacks float64/eigh)"
    if model_dev != resolve_device():
        return ridge + f" | model forward: {model_dev}"
    return ridge


def _fold_bounds(n: int, n_folds: int, fold_i: int) -> tuple[int, int]:
    """Contiguous validation-fold bounds.

    Mirrors the original NumPy loop exactly: equal-sized folds with the final
    fold absorbing the remainder. Changing this would change which alpha is
    selected, so it is kept bit-for-bit identical.

    Args:
        n: Number of rows.
        n_folds: Number of folds.
        fold_i: Fold index.

    Returns:
        tuple[int, int]: ``(lo, hi)`` row bounds of the validation fold.
    """
    fold_size = n // n_folds
    lo = fold_i * fold_size
    hi = lo + fold_size if fold_i < n_folds - 1 else n
    return lo, hi


def _svd_fit_ridge(X: NDArray, Y: NDArray, alpha: float) -> NDArray:
    """Ridge via the SVD of ``X`` — same solution, far better conditioned.

    ``b = V diag(s / (s^2 + a)) U^T Y`` is algebraically identical to the normal
    equations but never forms ``X^T X``, which squares the condition number. When
    a design mixes wildly different column scales the Gram matrix can exceed
    float64's ~1e16 dynamic range, at which point adding ``alpha`` to the
    diagonal no longer lifts the smallest eigenvalue above zero and the solve
    fails outright. Working from ``X`` directly keeps the conditioning at
    ``sqrt`` of that.
    """
    U, s, Vt = np.linalg.svd(np.asarray(X, dtype=np.float64), full_matrices=False)
    return Vt.T @ ((s / (s ** 2 + alpha))[:, None] * (U.T @ Y))


def _np_fit_ridge(X: NDArray, Y: NDArray, alpha: float) -> NDArray:
    """Closed-form ridge on CPU: ``(X^T X + aI)^-1 X^T Y``.

    Uses the normal equations, which are fast and adequate for well-scaled
    designs, and falls back to :func:`_svd_fit_ridge` when they break down.
    ``X^T X + aI`` is positive definite in exact arithmetic, so a singular-matrix
    failure always means the Gram matrix lost the penalty to rounding rather than
    that the problem is genuinely rank-deficient — the SVD path recovers the same
    solution. Non-finite output is caught too, since an overflowed Gram matrix
    can produce NaNs without raising.
    """
    n, p = X.shape
    try:
        if n >= p:
            A = X.T @ X
            A[np.diag_indices(p)] += alpha
            coef = np.linalg.solve(A, X.T @ Y)
        else:
            A = X @ X.T
            A[np.diag_indices(n)] += alpha
            coef = X.T @ np.linalg.solve(A, Y)
    except np.linalg.LinAlgError:
        coef = None
    if coef is None or not np.all(np.isfinite(coef)):
        warnings.warn(
            "Ridge normal equations were numerically singular; falling back to the "
            "SVD solve. This means the design mixes very different column scales — "
            "check for unstandardized raw-count features before trusting the fit.",
            RuntimeWarning,
            stacklevel=3
        )
        coef = _svd_fit_ridge(X, Y, alpha)
    return coef


def _np_select_alpha(
    X: NDArray,
    Y: NDArray,
    alpha_grid: NDArray,
    n_folds: int,
    criterion: str = "r2"
) -> float:
    """CPU alpha selection, eigendecomposed over the alpha path."""
    n = len(X)
    scores = np.zeros(len(alpha_grid))
    for fold_i in range(n_folds):
        lo, hi = _fold_bounds(n, n_folds, fold_i)
        mask = np.ones(n, dtype=bool)
        mask[lo:hi] = False
        X_tr, Y_tr, X_val, Y_val = X[mask], Y[mask], X[~mask], Y[~mask]

        nn, pp = X_tr.shape
        dual = nn < pp
        G = (X_tr @ X_tr.T) if dual else (X_tr.T @ X_tr)
        rhs = Y_tr if dual else (X_tr.T @ Y_tr)
        s, V = np.linalg.eigh(G)
        VtRhs = V.T @ rhs
        XvV = (X_val @ X_tr.T @ V) if dual else (X_val @ V)

        yv = Y_val - Y_val.mean(axis=0, keepdims=True)
        yv_ss = (yv ** 2).sum(axis=0)
        for ai, alpha in enumerate(alpha_grid):
            pred = XvV @ (VtRhs / (s + float(alpha))[:, None])
            if criterion == "r2":
                ss_res = ((Y_val - pred) ** 2).sum(axis=0)
                scores[ai] += float((1.0 - ss_res / (yv_ss + 1e-12)).mean())
            else:
                yp = pred - pred.mean(axis=0, keepdims=True)
                num = (yv * yp).sum(axis=0)
                den = np.sqrt(yv_ss * (yp ** 2).sum(axis=0)) + 1e-12
                scores[ai] += float((num / den).mean())
    return float(alpha_grid[int(np.argmax(scores))])


def fit_ridge(X: NDArray, Y: NDArray, alpha: float) -> NDArray:
    """Closed-form ridge regression.

    Uses the primal form when ``n >= p`` and the dual form otherwise, matching
    the original NumPy implementation.

    Args:
        X: Design matrix ``(n, p)``.
        Y: Targets ``(n, r)``.
        alpha: Ridge penalty.

    Returns:
        NDArray: Coefficients ``(p, r)``.

    Note:
        A single fit stays on CPU: one solve moves more data to the GPU than the
        solve saves. GPU acceleration applies to :func:`select_alpha`, which
        amortizes one transfer across the whole fold x alpha sweep.
    """
    return _np_fit_ridge(np.asarray(X), np.asarray(Y), alpha)


def select_alpha(
    X: NDArray,
    Y: NDArray,
    alpha_grid: NDArray,
    n_folds: int,
    device: str | None = None,
    criterion: str = "r2"
) -> float:
    """Select the ridge alpha by inner k-fold CV.

    Args:
        X: Design matrix ``(n, p)``.
        Y: Targets ``(n, r)``.
        alpha_grid: Candidate penalties.
        n_folds: Inner CV folds.
        device: Override the resolved device (mainly for tests).
        criterion: ``'r2'`` (default) selects the alpha maximizing mean R², which
            matches the metric the experiments report. ``'r'`` selects on mean
            Pearson r, the original behaviour, retained for comparison.

    Returns:
        float: The selected alpha.

    Note:
        Selecting on ``'r'`` while reporting R² is unsound: Pearson r is invariant
        to scale and offset, so an alpha that maximizes r can be badly
        under-regularized in R² terms. That mismatch is why the default is
        ``'r2'``.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    dev = device or resolve_device()
    # an explicit device is honoured as-is so tests can compare paths directly;
    # automatic selection also requires the problem to be worth shipping
    if not dev.startswith("cuda") or (device is None and not _gpu_worth_it(Y)):
        return _np_select_alpha(
            X,
            Y,
            alpha_grid,
            n_folds,
            criterion
        )

    try:
        import torch
    except ImportError:
        return _np_select_alpha(
            X,
            Y,
            alpha_grid,
            n_folds,
            criterion
        )
    if not torch.cuda.is_available():
        return _np_select_alpha(
            X,
            Y,
            alpha_grid,
            n_folds,
            criterion
        )

    try:
        return _torch_select_alpha(
            X,
            Y,
            alpha_grid,
            n_folds,
            dev,
            criterion
        )
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        # out of VRAM or a driver problem: fall back rather than fail a long run,
        # since the CPU path is numerically equivalent, only slower
        torch.cuda.empty_cache()
        return _np_select_alpha(
            X,
            Y,
            alpha_grid,
            n_folds,
            criterion
        )


def _torch_select_alpha(
    X: NDArray,
    Y: NDArray,
    alpha_grid: NDArray,
    n_folds: int,
    device: str,
    criterion: str = "r2"
) -> float:
    """CUDA alpha selection over an eigendecomposed alpha path.

    Args:
        X: Design matrix ``(n, p)``.
        Y: Targets ``(n, r)``.
        alpha_grid: Candidate penalties.
        n_folds: Inner CV folds.
        device: Torch device string.
        criterion: ``'r2'`` or ``'r'``; see :func:`select_alpha`.

    Returns:
        float: The selected alpha.
    """
    import torch

    dt = torch.float64
    Xt = torch.as_tensor(X, dtype=dt, device=device)
    Yt = torch.as_tensor(Y, dtype=dt, device=device)
    n = Xt.shape[0]
    scores = torch.zeros(len(alpha_grid), dtype=dt, device=device)

    for fold_i in range(n_folds):
        lo, hi = _fold_bounds(n, n_folds, fold_i)
        mask = torch.ones(n, dtype=torch.bool, device=device)
        mask[lo:hi] = False
        X_tr, Y_tr, X_val, Y_val = Xt[mask], Yt[mask], Xt[~mask], Yt[~mask]

        nn, pp = X_tr.shape
        dual = nn < pp
        G = (X_tr @ X_tr.T) if dual else (X_tr.T @ X_tr)
        rhs = Y_tr if dual else (X_tr.T @ Y_tr)
        s, V = torch.linalg.eigh(G)
        VtRhs = V.T @ rhs
        XvV = (X_val @ X_tr.T @ V) if dual else (X_val @ V)

        yv = Y_val - Y_val.mean(0, keepdim=True)
        yv_ss = (yv ** 2).sum(0)
        for ai, alpha in enumerate(alpha_grid):
            pred = XvV @ (VtRhs / (s + float(alpha)).unsqueeze(1))
            if criterion == "r2":
                ss_res = ((Y_val - pred) ** 2).sum(0)
                scores[ai] += (1.0 - ss_res / (yv_ss + 1e-12)).mean()
            else:
                yp = pred - pred.mean(0, keepdim=True)
                num = (yv * yp).sum(0)
                den = torch.sqrt(yv_ss * (yp ** 2).sum(0)) + 1e-12
                scores[ai] += (num / den).mean()

    best = int(torch.argmax(scores).item())
    del Xt, Yt, scores
    return float(alpha_grid[best])
