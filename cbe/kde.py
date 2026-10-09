# -*- coding: utf-8 -*-
"""KDE coverage score: Willimetz & Grajciar, JPCL 2025, Eq. (1).

    rho(q) = (1/k) * sum_{i in k(q)} exp( -||q - x_i||^2 / (2 h^2) )

rho ∈ (0, 1]; 1 = fully covered by the training set, smaller = further outside. Empirical
threshold 0.5.
"""
from __future__ import annotations

import numpy as np


def kde_score(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute the KDE score from the k nearest-neighbour Euclidean distances.

    Parameters:
        distances: (n_query, k) matrix of Euclidean distances
        bandwidth: kernel bandwidth h
    Returns:
        (n_query,) array of rho scores, ∈ (0, 1]
    """
    distances = np.asarray(distances, dtype=np.float64)
    bandwidth = float(bandwidth)
    if bandwidth <= 0:
        raise ValueError("bandwidth must be > 0")
    return np.mean(np.exp(-(distances ** 2) / (2.0 * bandwidth ** 2)), axis=1)


def exp_coverage(distances: np.ndarray, scale: float) -> np.ndarray:
    """Exponential coverage = exp(-d/scale) ∈ (0, 1].

    Smoother than KDE (exp(-d²/2h²)): when the OOD distance is much larger than scale, KDE
    underflows to 0, whereas exp(-d/scale) still gives a continuous, spread-out [0, 1]
    coverage and a more continuous scatter plot. scale is usually the mean k nearest-neighbour
    distance of the training set.
    """
    distances = np.asarray(distances, dtype=np.float64)
    scale = float(scale)
    if scale <= 0:
        raise ValueError("scale must be > 0")
    return np.mean(np.exp(-distances / scale), axis=1)


def estimate_bandwidth(index, X: np.ndarray, k: int, mode: str = "std") -> float:
    """Kernel bandwidth h = a statistic of the k nearest-neighbour distances of the training set.

    mode:
      - "std" (default, h_std of the paper): standard deviation. Suits large, diverse corpora.
      - "mean" / "median": more stable for a compact, single-material training set (avoids
        bandwidth collapse, where every coverage is pushed down to 0).
    Silverman / IQR underestimate h by an order of magnitude on large data (Willimetz SI S1).
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[:, None]
    kk = min(k + 1, len(X)) if k < len(X) else len(X)
    dist, _ = index.search(X, kk)
    if dist.shape[1] > 1:
        dist = dist[:, 1:]  # drop the nearest neighbour (the point itself)
    if mode == "std":
        val = float(np.std(dist))
    elif mode == "mean":
        val = float(np.mean(dist))
    elif mode == "median":
        val = float(np.median(dist))
    else:
        raise ValueError(f"unknown bandwidth mode: {mode}")
    if not np.isfinite(val) or val <= 0:
        raise ValueError("zero variance in the training set: all descriptors are identical, "
                         "cannot estimate a bandwidth")
    return val


def estimate_bandwidth_silverman(X) -> float:
    """Silverman's rule (multivariate Gaussian kernel): h = σ·(4/(d+2))^(1/(d+4))·n^(-1/(d+4)).

    σ is the RMS of the per-dimension std of the descriptors (the global spread), n the number
    of samples and d the dimension. Unlike h_std (the std of the k nearest-neighbour distances,
    which scales with density), Silverman uses the **global spread** and is therefore more
    robust on both dense and sparse datasets (h is not squeezed too small as the index grows
    denser). X may be the full matrix or a sampled one (e.g. 200k points).
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    sig = X.std(axis=0)
    sigma = float(np.sqrt((sig ** 2).mean()))
    return sigma * (4.0 / (d + 2)) ** (1.0 / (d + 4)) * n ** (-1.0 / (d + 4))


def estimate_bandwidth_sampled(index, k: int, n_sample: int = 200000,
                               mode: str = "std", seed: int = 0) -> float:
    """Estimate the KDE bandwidth by sampling the prebuilt index (no descriptor matrix needed).

    Procedure: uniformly sample n_sample indexed vectors (index.reconstruct), search their
    k+1 nearest neighbours, drop the point itself (the one at distance 0) and take the
    std/mean/median of the remaining k distances.

    Difference from estimate_bandwidth: that one needs an explicit X, whereas this function
    consumes a faiss index directly -- convenient once build_index.py has run and the raw
    features were not saved. Returns the bandwidth h.
    """
    rng = np.random.default_rng(seed)
    n = int(index.n)
    n_sample = min(n_sample, n)
    idxs = rng.choice(n, size=n_sample, replace=False)

    X = np.vstack([np.asarray(index.reconstruct(int(i)), dtype=np.float32)
                   for i in idxs])

    kk = min(k + 1, n)
    dist, _ = index.search(X, kk)
    if dist.shape[1] > 1:
        dist = dist[:, 1:]  # drop the nearest neighbour (the point itself, at distance 0)

    if mode == "std":
        val = float(np.std(dist))
    elif mode == "mean":
        val = float(np.mean(dist))
    elif mode == "median":
        val = float(np.median(dist))
    else:
        raise ValueError(f"unknown bandwidth mode: {mode}")
    if not np.isfinite(val) or val <= 0:
        raise ValueError("zero variance in the training set: all descriptors are identical, "
                         "cannot estimate a bandwidth")
    return val
