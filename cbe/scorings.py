# -*- coding: utf-8 -*-
"""Density / novelty scorers (coverage alternatives to kNN-KDE).

Unlike CoverageModel's kde/distance, these are the classic/deep-ML OOD scorers, wrapped in a
uniform "fit(training features) -> score(query features)" interface so that the
coverage-vs-error benchmark can use them directly.

Unified convention (same as CoverageModel.score):
  score(X) returns the "extrapolation risk", **larger = further outside** (kde coverage goes
  the other way; here everything is expressed as a risk).

Intended use: features already reduced by PCA (e.g. MACE 256 -> 16 dims, the same space as
the index). When the training index is large, sample a subset (e.g. 10k-20k points) before
fitting, rather than feeding all 49M points to these methods.
"""
from __future__ import annotations

import numpy as np


class _BaseScorer:
    """Scorer base class: fit(X) returns self, score(X) -> (n,) risk array."""

    def fit(self, X: np.ndarray) -> "_BaseScorer":
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def score_scalar(self, x: np.ndarray) -> float:
        """Risk scalar of a single point (1×d or (d,))."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        return float(np.asarray(self.score(x)).ravel()[0])


class MahalanobisScorer(_BaseScorer):
    """Mahalanobis distance: d² = (x−μ)ᵀ Σ⁻¹ (x−μ).

    A single-Gaussian assumption: cheap and interpretable, but on a multi-modal training
    distribution (e.g. 1.4M materials) it underestimates the density of regions that are
    sparse yet still part of the training set. score = the Mahalanobis distance itself
    (larger = further outside).
    """

    def __init__(self, reg: float = 1e-6):
        self.reg = reg
        self.mu_ = None
        self.prec_ = None

    def fit(self, X: np.ndarray) -> "MahalanobisScorer":
        X = np.asarray(X, dtype=np.float64)
        self.mu_ = X.mean(axis=0)
        cov = np.cov(X, rowvar=False) + self.reg * np.eye(X.shape[1])
        self.prec_ = np.linalg.pinv(cov)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        d = X - self.mu_
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.prec_, d), 0.0))


class GMMScorer(_BaseScorer):
    """Gaussian-mixture density. score = -log p(x) (larger = further outside).

    Multi-modal and gives a genuine density semantics (the same family as KDE); fitting is
    fast in the low-dimensional PCA space.
    """

    def __init__(self, n_components: int = 32, reg_covar: float = 1e-5,
                 random_state: int = 0):
        self.n_components = n_components
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.gmm_ = None

    def fit(self, X: np.ndarray) -> "GMMScorer":
        from sklearn.mixture import GaussianMixture
        X = np.asarray(X, dtype=np.float64)
        self.gmm_ = GaussianMixture(
            n_components=self.n_components, covariance_type="full",
            reg_covar=self.reg_covar, random_state=self.random_state,
            max_iter=200, n_init=1)
        self.gmm_.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.gmm_.score_samples(np.asarray(X, dtype=np.float64))


class LOFScorer(_BaseScorer):
    """Local Outlier Factor (local density ratio). score = -score_samples (larger = further out).

    Density-adaptive: it picks out "locally sparse pockets" (rare materials) inside a dense
    backbone, but on a large training set it needs novelty=True to fit the boundary first and
    score new points afterwards (fitting only needs a training subset).
    """

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.lof_ = None

    def fit(self, X: np.ndarray) -> "LOFScorer":
        from sklearn.neighbors import LocalOutlierFactor
        X = np.asarray(X, dtype=np.float64)
        self.lof_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors, contamination=self.contamination,
            novelty=True, n_jobs=1)
        self.lof_.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.lof_.score_samples(np.asarray(X, dtype=np.float64))


class OneClassSVMScorer(_BaseScorer):
    """One-Class SVM (the SVM implementation of SVDD). score = -decision_function
    (larger = further outside).

    Learns a boundary enclosing the training distribution: no density assumption and friendly
    to high dimensions. The drawbacks are that it gives no gradient inside the boundary (every
    in-distribution point gets a similarly large decision value) and that, with an RBF kernel,
    the fit is fairly sensitive to the number of samples.
    """

    def __init__(self, nu: float = 0.05, gamma: str = "scale", kernel: str = "rbf"):
        self.nu = nu
        self.gamma = gamma
        self.kernel = kernel
        self.svm_ = None

    def fit(self, X: np.ndarray) -> "OneClassSVMScorer":
        from sklearn.svm import OneClassSVM
        X = np.asarray(X, dtype=np.float64)
        self.svm_ = OneClassSVM(kernel=self.kernel, gamma=self.gamma, nu=self.nu)
        self.svm_.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.svm_.decision_function(np.asarray(X, dtype=np.float64))


# unified registry: name -> (constructor, has density semantics)
DENSITY_SCORERS = {
    "mahalanobis": MahalanobisScorer,
    "gmm": GMMScorer,
    "lof": LOFScorer,
    "oneclasssvm": OneClassSVMScorer,
}


def make_scorer(name: str, **kwargs) -> _BaseScorer:
    """Construct a scorer by name (the pluggable entry point)."""
    if name not in DENSITY_SCORERS:
        raise ValueError(f"unknown scorer: {name} (available: {sorted(DENSITY_SCORERS)})")
    return DENSITY_SCORERS[name](**kwargs)
