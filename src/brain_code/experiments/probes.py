"""Fresh probes for measuring what a representation linearly/nonlinearly encodes.

Used by the LEACE erasure analysis to verify erasure and to measure selectivity:

* ``linear_probe_accuracy`` — a *fresh* multinomial logistic-regression probe
  (numpy, deterministic).  This is the probe that should drop to chance on the
  ERASED target concept: LEACE guarantees no linear map can recover it.
* ``nonlinear_probe_accuracy`` — a small torch MLP.  Its purpose is the
  opposite: to demonstrate that LEACE does **not** erase non-linear structure.
  If the MLP still recovers the concept after erasure, that is expected and
  must be reported honestly.
* ``selectivity_report`` — probes the erased representation for the target
  attribute (expect ~chance, linearly) and for a set of unrelated attributes
  (expect preserved), returning per-attribute accuracies and a summary.

Probes are trained on a train split and evaluated on a disjoint test split.
The caller is responsible for keeping those splits leak-free (e.g. the probe
train/test split, never the neural stimuli).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _encode_labels(y_train: NDArray, y_test: NDArray) -> tuple[NDArray, NDArray, int]:
    """Maps arbitrary labels to contiguous 0..k-1 using the train classes.

    Test labels unseen in train are mapped to -1 (always counted as wrong).
    """
    classes = np.unique(y_train)
    lookup = {c: i for i, c in enumerate(classes)}
    yt = np.array([lookup[c] for c in y_train], dtype=np.int64)
    ye = np.array([lookup.get(c, -1) for c in y_test], dtype=np.int64)
    return yt, ye, len(classes)


def chance_level(y: NDArray) -> dict[str, float]:
    """Returns both chance floors for a label vector.

    ``uniform`` = 1/k (balanced-guess baseline);
    ``majority`` = frequency of the most common class (the honest floor for
    imbalanced targets — a constant predictor achieves this).
    """
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    k = len(counts)
    return {
        "uniform": float(1.0 / max(k, 1)),
        "majority": float(counts.max() / max(len(y), 1))
    }


def _standardize(X_train: NDArray, X_test: NDArray) -> tuple[NDArray, NDArray]:
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd


def _softmax(Z: NDArray) -> NDArray:
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)


def linear_probe_accuracy(
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    y_test: NDArray,
    *,
    l2: float = 1e-2,
    n_iter: int = 500,
    lr: float = 0.5,
    seed: int = 0  # accepted for API uniformity; init is deterministic (zeros)
) -> float:
    """Returns the top-1 accuracy of a fresh L2-regularised logistic-regression probe.

    Features are standardised on the train split.  Optimisation is full-batch
    gradient descent from a zero initialisation on a convex objective, so the
    result is deterministic (the ``seed`` argument is unused but kept for a
    uniform probe API).
    """
    del seed
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    yt, ye, k = _encode_labels(np.asarray(y_train), np.asarray(y_test))
    if k < 2:
        # one class in train makes the probe a constant predictor
        return float((ye == 0).mean())

    Xtr, Xte = _standardize(X_train, X_test)
    n, d = Xtr.shape
    W = np.zeros((d, k), dtype=np.float64)
    b = np.zeros(k, dtype=np.float64)
    Y = np.eye(k)[yt]

    for _ in range(n_iter):
        P = _softmax(Xtr @ W + b)
        gW = Xtr.T @ (P - Y) / n + l2 * W
        gb = (P - Y).mean(axis=0)
        W -= lr * gW
        b -= lr * gb

    pred = np.argmax(Xte @ W + b, axis=1)
    return float((pred == ye).mean())


def nonlinear_probe_accuracy(
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    y_test: NDArray,
    *,
    hidden: int = 32,
    epochs: int = 150,
    lr: float = 1e-2,
    seed: int = 0
) -> float:
    """Returns the top-1 accuracy of a small one-hidden-layer ReLU MLP (torch, CPU, seeded).

    LEACE guarantees only *linear* erasure, so a positive result here after
    erasure is expected and informative — it quantifies residual non-linear
    decodability, not a failure of LEACE.
    """
    import torch  # local import: torch is heavy and only needed for this probe

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    yt, ye, k = _encode_labels(np.asarray(y_train), np.asarray(y_test))
    if k < 2:
        return float((ye == 0).mean())

    Xtr_np, Xte_np = _standardize(X_train.astype(np.float64), X_test.astype(np.float64))

    torch.manual_seed(seed)
    Xtr = torch.tensor(Xtr_np, dtype=torch.float32)
    Xte = torch.tensor(Xte_np, dtype=torch.float32)
    ytr = torch.tensor(yt, dtype=torch.long)

    model = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, k)
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(dim=1).numpy()
    # unseen test labels are encoded as -1 and never match
    return float((pred == ye).mean())


def selectivity_report(
    orig_train: NDArray,
    orig_test: NDArray,
    erased_train: NDArray,
    erased_test: NDArray,
    attribute_labels_train: dict[str, NDArray],
    attribute_labels_test: dict[str, NDArray],
    *,
    target_attribute: str | None = None,
    preservation_threshold: float = 0.9,
    seed: int = 0
) -> dict:
    """Probes each attribute on the original and the erased representations.

    For every attribute we fit a fresh linear probe on the original and on the
    erased representations, plus a non-linear probe on the erased ones.  The
    ``target_attribute`` (the concept LEACE erased) is expected to fall to
    chance under the linear probe; all other (unrelated) attributes are
    expected to be preserved (erased accuracy >= ``preservation_threshold`` ×
    original accuracy).

    Returns:
        A dict keyed by attribute, plus a ``_summary`` entry with the mean preservation ratio over
        unrelated attributes and the target's linear drop.
    """
    report: dict[str, dict] = {}
    preservation_ratios: list[float] = []
    target_drop = None

    for attr, y_tr in attribute_labels_train.items():
        y_te = attribute_labels_test[attr]
        lin_orig = linear_probe_accuracy(
            orig_train,
            y_tr,
            orig_test,
            y_te,
            seed=seed
        )
        lin_erased = linear_probe_accuracy(
            erased_train,
            y_tr,
            erased_test,
            y_te,
            seed=seed
        )
        nonlin_erased = nonlinear_probe_accuracy(
            erased_train,
            y_tr,
            erased_test,
            y_te,
            seed=seed
        )
        chance = chance_level(y_te)
        ratio = lin_erased / max(lin_orig, 1e-9)

        is_target = attr == target_attribute
        if is_target:
            # within 0.05 of the majority floor counts as erased to chance
            preserved = None
            target_drop = float(lin_orig - lin_erased)
            erased_to_chance = bool(lin_erased <= chance["majority"] + 0.05)
        else:
            preserved = bool(lin_erased >= preservation_threshold * lin_orig)
            erased_to_chance = None
            preservation_ratios.append(ratio)

        report[attr] = {
            "is_target": is_target,
            "linear_orig": float(lin_orig),
            "linear_erased": float(lin_erased),
            "nonlinear_erased": float(nonlin_erased),
            "chance_uniform": chance["uniform"],
            "chance_majority": chance["majority"],
            "preservation_ratio": float(ratio),
            "preserved": preserved,
            "erased_to_chance": erased_to_chance
        }

    report["_summary"] = {
        "target_attribute": target_attribute,
        "target_linear_drop": target_drop,
        "mean_unrelated_preservation_ratio": (
            float(np.mean(preservation_ratios)) if preservation_ratios else None
        ),
        "all_unrelated_preserved": (
            bool(all(
                report[a]["preserved"]
                for a in attribute_labels_train
                if a != target_attribute
            ))
            if len(attribute_labels_train) > (1 if target_attribute else 0)
            else None
        )
    }
    return report
