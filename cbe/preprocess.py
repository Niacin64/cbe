# -*- coding: utf-8 -*-
"""Preprocessing: PCA dimensionality reduction (pure numpy, no hard sklearn dependency).

Reduces high-dimensional descriptors (MACE 256-d / SOAP with thousands of dims) to a low
dimension (16 is recommended in the paper); this saves storage/VRAM and speeds up the
nearest-neighbour query. See Willimetz & Grajciar, JPCL 2025, SI S2.
"""
from __future__ import annotations

import numpy as np


class PCA:
    """Principal component analysis based on the SVD. fit -> transform."""

    def __init__(self, n_components: int):
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt[: self.n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("call fit first")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def save(self, path: str) -> None:
        """Write ``mean`` and ``components`` to an npz file, as ``build_index.py`` does."""
        if self.mean_ is None:
            raise RuntimeError("call fit before save")
        np.savez(path, mean=self.mean_, components=self.components_)

    @classmethod
    def load(cls, path: str) -> "PCA":
        """Load from an npz file (e.g. the pca.npz written by build_index.py)."""
        d = np.load(path)
        pca = cls(int(d["components"].shape[0]))
        pca.mean_ = d["mean"]
        pca.components_ = d["components"]
        return pca
