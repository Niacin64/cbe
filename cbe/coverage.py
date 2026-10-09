# -*- coding: utf-8 -*-
"""Coverage layer: turn the distance to the training points into a coverage/extrapolation score.

Two scorings are supported:
  - "kde"  (default, recommended): KDE coverage rho ∈ (0, 1], 1 = fully covered.
            Structure-level coverage = minimum over the per-atom rho (the weakest link);
            extrapolation risk = 1 - coverage (larger = further outside the training set).
  - "distance": distance to the k-th nearest neighbour (0 = fully covered, larger = further
            outside); structure-level score = aggregation (agg) of the per-atom distances.

Unified convention: score() always returns the "extrapolation risk, larger = further outside",
so that the benchmark can consume it directly.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from ase import Atoms

from .descriptors import Descriptor
from .index import Index, BruteForceIndex, FaissIndex
from .kde import kde_score, exp_coverage, estimate_bandwidth
from .preprocess import PCA


class CoverageModel:
    """Coverage model: fit(training set) -> score/coverage(query set).

    Parameters:
        descriptor: descriptor (per-atom or structure level)
        index: nearest-neighbour index (exact brute force by default)
        k: number of neighbours used for scoring (100 for kde; 1-5 for distance)
        scoring: "kde" or "distance"
        pca_dim: if given and below the feature dimension, PCA down to it (16 is advised)
        bandwidth: kernel bandwidth for kde; None estimates it during fit
        agg: per-atom aggregation in the distance mode: "max"/"mean"/"p95"
    """

    def __init__(self, descriptor: Descriptor, index: Optional[Index] = None,
                 k: int = 100, scoring: str = "kde", pca_dim: Optional[int] = None,
                 bandwidth: Optional[float] = None, bandwidth_mode: str = "std",
                 agg: str = "max", coverage_agg: str = "min"):
        if scoring not in ("kde", "distance", "exp"):
            raise ValueError(f"unknown scoring: {scoring}")
        self.descriptor = descriptor
        self.index = index if index is not None else BruteForceIndex()
        self.k = k
        self.scoring = scoring
        self.pca_dim = pca_dim
        self.bandwidth = bandwidth
        self.bandwidth_mode = bandwidth_mode
        self.agg = agg
        self.coverage_agg = coverage_agg
        self._per_atom = descriptor.level == "per_atom"
        self._pca: Optional[PCA] = None

    # ---- fit ----
    def fit(self, atoms_list: List[Atoms]) -> "CoverageModel":
        feats = [self.descriptor.describe(a) for a in atoms_list]
        if self._per_atom:
            X = np.concatenate(feats, axis=0)
        else:
            X = np.stack(feats, axis=0)
        X = self._maybe_pca_fit(X)
        self.index.build(X)
        if self.scoring in ("kde", "exp") and self.bandwidth is None:
            # kde uses mode; for exp the mean distance gives a smoother scale
            mode = "mean" if self.scoring == "exp" else self.bandwidth_mode
            self.bandwidth = estimate_bandwidth(self.index, X, self.k, mode=mode)
        return self

    # ---- persistence ----
    def save(self, directory) -> None:
        """Write everything needed to reload this model into ``directory``.

        Produces the same layout as the pretrained indexes distributed with the package:
        ``index.faiss`` (or ``vectors.npy`` for a brute-force index), ``pca.npz`` and a
        ``meta.json`` recording the parameters the index was built with. Reload with
        :meth:`load`; note that the *descriptor* is not stored, because a descriptor is
        code, so the caller must pass the same one back.
        """
        import json
        from pathlib import Path as _Path

        d = _Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        if isinstance(self.index, BruteForceIndex):
            self.index.save(d / "vectors.npy")
            index_file = "vectors.npy"
        else:
            self.index.save(d / "index.faiss")
            index_file = "index.faiss"
        if self._pca is not None:
            self._pca.save(d / "pca.npz")
        meta = {
            "k": int(self.k),
            "scoring": self.scoring,
            "agg": self.agg,
            "coverage_agg": self.coverage_agg,
            "pca_dim": None if self._pca is None else int(self._pca.n_components),
            "bandwidth": None if self.bandwidth is None else float(self.bandwidth),
            "bandwidth_mode": self.bandwidth_mode,
            "n_environments": int(self.index.n),
            "descriptor_dim": int(self.descriptor.dim),
            "layout": "single" if index_file == "index.faiss" else "bruteforce",
            "files": [index_file] + (["pca.npz"] if self._pca is not None else []),
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"saved {self.index.n:,} indexed environments "
              f"(k={self.k}, scoring={self.scoring}, h={self.bandwidth}) to {d}")

    @classmethod
    def load(cls, directory, descriptor) -> "CoverageModel":
        """Reload a model written by :meth:`save`.

        ``descriptor`` must be the same descriptor the index was built with.
        """
        import json
        from pathlib import Path as _Path

        d = _Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        idx = BruteForceIndex() if meta.get("layout") == "bruteforce" else FaissIndex()
        idx.load(d / ("vectors.npy" if meta.get("layout") == "bruteforce" else "index.faiss"))
        m = cls(descriptor, index=idx, k=meta.get("k", 30),
                scoring=meta.get("scoring", "kde"), pca_dim=meta.get("pca_dim"),
                bandwidth=meta.get("bandwidth"), bandwidth_mode=meta.get("bandwidth_mode", "std"),
                agg=meta.get("agg", "max"), coverage_agg=meta.get("coverage_agg", "min"))
        if (d / "pca.npz").exists():
            m._pca = PCA.load(d / "pca.npz")
        m._per_atom = descriptor.level == "per_atom"
        return m

    @classmethod
    def from_index(cls, index_path, pca_path, descriptor, k=30,
                   scoring="distance", agg="max", coverage_agg="min",
                   bandwidth=None, bandwidth_mode="std") -> "CoverageModel":
        """Construct a CoverageModel from a prebuilt large index plus its PCA (skips fit).

        index_path: path to index.faiss; pca_path: path to pca.npz (may be None: no
        dimensionality reduction).
        Note: the index was built on PCA-transformed descriptors, so `descriptor` must be the
        same one used at build time (e.g. the 256-d MACE-MP-0 latent invariant descriptor).
        With scoring="kde"/"exp" a bandwidth must be given (otherwise kde_score fails because
        h=None); use kde.estimate_bandwidth_sampled to estimate it by sampling the index.
        """
        idx = FaissIndex()
        idx.load(index_path)
        m = cls(descriptor, index=idx, k=k, scoring=scoring, agg=agg,
                coverage_agg=coverage_agg, bandwidth=bandwidth,
                bandwidth_mode=bandwidth_mode)
        m._pca = PCA.load(pca_path) if pca_path else None
        m._per_atom = descriptor.level == "per_atom"
        return m

    @classmethod
    def from_shards(cls, shard_dir, pca_path, descriptor, k=30,
                    scoring="kde", agg="max", coverage_agg="min",
                    bandwidth=None, bandwidth_mode="std", lazy=True) -> "CoverageModel":
        """Same as from_index, but the index is a sharded directory (shard_*.faiss, see
        ShardedFaissIndex).

        lazy=True (default): shards are loaded one at a time at query time, so memory is about
        one shard (mandatory for large indexes).
        """
        from .index import ShardedFaissIndex

        idx = ShardedFaissIndex().load(shard_dir, lazy=lazy)
        m = cls(descriptor, index=idx, k=k, scoring=scoring, agg=agg,
                coverage_agg=coverage_agg, bandwidth=bandwidth,
                bandwidth_mode=bandwidth_mode)
        m._pca = PCA.load(pca_path) if pca_path else None
        m._per_atom = descriptor.level == "per_atom"
        return m

    def _maybe_pca_fit(self, X: np.ndarray) -> np.ndarray:
        if self.pca_dim is not None and self.pca_dim < X.shape[1]:
            self._pca = PCA(self.pca_dim)
            return self._pca.fit_transform(X)
        self._pca = None
        return X

    def _query_features(self, atoms: Atoms) -> np.ndarray:
        f = self.descriptor.describe(atoms)
        if f.ndim == 1:
            f = f[None, :]
        if self._pca is not None:
            f = self._pca.transform(f)
        return f

    def _kdist(self, atoms: Atoms) -> np.ndarray:
        """k nearest-neighbour distances of the query rows, (n_rows, k)."""
        dist, _ = self.index.search(self._query_features(atoms), self.k)
        return dist

    # ---- per-atom / per-row ----
    def score_per_atom(self, atoms: Atoms) -> np.ndarray:
        """Raw per-atom (or per-row) score.

        The kde/exp modes return coverage (1 = good); the distance mode returns the distance
        (0 = good).
        """
        d = self._kdist(atoms)
        if self.scoring == "kde":
            return kde_score(d, self.bandwidth)
        if self.scoring == "exp":
            return exp_coverage(d, self.bandwidth)
        return d[:, -1]  # distance to the k-th nearest neighbour

    # ---- structure level ----
    def coverage(self, atoms: Atoms, agg: str = None) -> float:
        """Structure-level coverage, 1 = fully covered by the training set (kde/exp modes).

        agg: per-atom aggregation (defaults to self.coverage_agg). "min" = weakest link
             (conservative), "mean" = average (smoother, gives a more continuous scatter plot).
        """
        if self.scoring not in ("kde", "exp"):
            raise ValueError("coverage() is only defined for scoring='kde'/'exp'")
        return float(_aggregate(self.score_per_atom(atoms),
                                agg or self.coverage_agg))

    def score(self, atoms: Atoms) -> float:
        """Structure-level extrapolation risk, larger = further out (usable by the benchmark)."""
        if self.scoring in ("kde", "exp"):
            return 1.0 - self.coverage(atoms)
        return float(_aggregate(self.score_per_atom(atoms), self.agg))

    def is_extrapolated(self, atoms: Atoms, threshold: float = 0.5) -> bool:
        """kde mode: whether coverage falls below the threshold (the paper's empirical 0.5)."""
        return self.coverage(atoms) < threshold

    def score_many(self, atoms_list: List[Atoms]) -> np.ndarray:
        return np.array([self.score(a) for a in atoms_list])

    def coverage_many(self, atoms_list: List[Atoms]) -> np.ndarray:
        return np.array([self.coverage(a) for a in atoms_list])


def _aggregate(d: np.ndarray, agg: str) -> float:
    if agg == "max":
        return float(np.max(d))
    if agg == "min":
        return float(np.min(d))
    if agg == "mean":
        return float(np.mean(d))
    if agg == "p95":
        return float(np.percentile(d, 95))
    raise ValueError(f"unknown aggregation: {agg}")
