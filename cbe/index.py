# -*- coding: utf-8 -*-
"""Index layer: build an (approximate) nearest-neighbour index over the training features.

Pluggable: FaissIndex (GPU / large scale) and BruteForceIndex (pure-numpy fallback / tests).
Convention: distances are always returned as **Euclidean distances** (not squared), ascending.
"""
from __future__ import annotations

from pathlib import Path

from abc import ABC, abstractmethod

import numpy as np


class Index(ABC):
    """Abstract base class for nearest-neighbour indexes."""

    def __init__(self):
        self._n = 0

    @property
    def n(self) -> int:
        """Number of indexed vectors."""
        return self._n

    @abstractmethod
    def build(self, X: np.ndarray) -> None:
        """Build the index from an (n, d) feature matrix."""

    @abstractmethod
    def search(self, Xq: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances, indices), each of shape (n_query, k)."""


class BruteForceIndex(Index):
    """Exact brute-force nearest-neighbour index (scipy cdist); for small or faiss-free setups."""

    def build(self, X: np.ndarray) -> None:
        self._X = np.asarray(X, dtype=np.float32)
        self._n = len(self._X)

    def search(self, Xq: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        from scipy.spatial.distance import cdist
        D = cdist(np.asarray(Xq, dtype=np.float32), self._X, metric="euclidean")
        k = min(k, self._n)
        idx = np.argsort(D, axis=1)[:, :k]
        dist = np.take_along_axis(D, idx, axis=1)
        return dist, idx

    def reconstruct(self, i: int) -> np.ndarray:
        """Return the i-th indexed vector (used to sample-estimate the bandwidth, etc.)."""
        return np.asarray(self._X[i], dtype=np.float32)

    def save(self, path) -> None:
        """Persist the vectors as ``<path>.npy`` (no faiss needed)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self._X)

    def load(self, path) -> "BruteForceIndex":
        """Load vectors written by :meth:`save` (``.npy`` or ``.npz``)."""
        p = Path(path)
        if p.suffix == ".npz":
            z = np.load(p)
            key = "vectors" if "vectors" in z else list(z.keys())[0]
            X = z[key]
        else:
            X = np.load(p)
        self._X = np.asarray(X, dtype=np.float32)
        self._n = len(self._X)
        return self


class FaissIndex(Index):
    """faiss-backed nearest-neighbour index.

    Parameters:
        exact: True -> IndexFlatL2 (exact); False -> HNSW (approximate, large scale)
        hnsw_M: number of HNSW connections (used only when exact=False)
    """

    def __init__(self, exact: bool = True, hnsw_M: int = 32):
        super().__init__()
        self.exact = exact
        self.hnsw_M = hnsw_M
        self._index = None

    def build(self, X: np.ndarray) -> None:
        import faiss
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[:, None]
        d = X.shape[1]
        if self.exact:
            self._index = faiss.IndexFlatL2(d)
        else:
            self._index = faiss.IndexHNSWFlat(d, self.hnsw_M)
        self._index.add(X)
        self._n = len(X)

    def load(self, path) -> "FaissIndex":
        """Load a saved Faiss index from disk (e.g. the index.faiss written by build_index.py)."""
        import faiss
        self._index = faiss.read_index(str(path))
        self._n = int(self._index.ntotal)
        return self

    def search(self, Xq: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        Xq = np.asarray(Xq, dtype=np.float32)
        if Xq.ndim == 1:
            Xq = Xq[:, None]
        k = min(k, self._n)
        # HNSW's default efSearch=16 < k gives low recall (it silently returns worse
        # neighbours). Raise efSearch to >= k before querying (with a floor of 64).
        hnsw = getattr(self._index, "hnsw", None)
        if hnsw is not None and getattr(hnsw, "efSearch", 16) < max(k, 64):
            hnsw.efSearch = max(k, 64)
        D2, I = self._index.search(Xq, k)
        return np.sqrt(np.maximum(D2, 0.0)), I

    def reconstruct(self, i: int) -> np.ndarray:
        """Return the i-th indexed vector (used to sample-estimate the bandwidth, etc.)."""
        return np.asarray(self._index.reconstruct(int(i)), dtype=np.float32)

    def save(self, path) -> None:
        """Write the faiss index to disk (readable again with :meth:`load`)."""
        import faiss
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))


class ShardedFaissIndex(Index):
    """Multi-shard FlatL2 index: search k neighbours per shard, merge, keep the global top-k.

    For cases where a single large index does not fit in local memory (e.g. the Alexandria
    index: 158.6M environments x 16 dims, 10 GB of features plus another 10 GB of index).
    Shard files are loaded in filename order (shard_000.faiss ...) and the returned ids are
    (prefix sum over shards) + (id within the shard), matching the ids obtained by
    concatenating all features in the same order.
    """

    def __init__(self):
        self.shards = []      # [(faiss_index, global_offset)]
        self._n = 0

    def load(self, directory, lazy: bool = False) -> "ShardedFaissIndex":
        """With lazy=True only the shard paths and sizes are recorded; shards are loaded one at
        a time at query time (memory is about one shard).

        A large index (e.g. Alexandria, 4 x 2.5 GB = 10 GB) must be loaded lazily on a machine
        with 15 GB of RAM.
        """
        import faiss as _faiss

        paths = sorted(Path(directory).glob("shard_*.faiss"))
        if not paths:
            raise FileNotFoundError(f"{directory} has no shard_*.faiss files")
        self.lazy = lazy
        self.paths = paths
        if lazy:
            self._sizes = [int(_faiss.read_index(str(p), _faiss.IO_FLAG_MMAP).ntotal)
                           for p in paths]
            self.shards = []
            self._n = int(sum(self._sizes))
            return self
        self.shards = []
        off = 0
        for p in paths:
            idx = _faiss.read_index(str(p))
            self.shards.append((idx, off))
            off += idx.ntotal
        self._n = off
        return self

    @property
    def n(self) -> int:
        return self._n

    def build(self, X: np.ndarray) -> None:      # pragma: no cover - read-only use
        raise NotImplementedError("sharded indexes are produced by make_index_from_npy.py; "
                                  "clients only read them")

    def search(self, Xq: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        import faiss as _faiss

        Xq = np.ascontiguousarray(Xq, dtype=np.float32)
        if getattr(self, "lazy", False):
            Ds, Is, off = [], [], 0
            for p, nt in zip(self.paths, self._sizes):
                # Read a whole shard into memory before searching (random mmap access on a
                # large index is several times slower because of page faults); release it
                # right after the search, so peak memory is about one shard
                idx = _faiss.read_index(str(p))
                d, i = idx.search(Xq, min(k, nt))
                Ds.append(d)
                Is.append(i + off)
                off += nt
                del idx
            D = np.concatenate(Ds, axis=1)
            I = np.concatenate(Is, axis=1)
            order = np.argsort(D, axis=1)[:, :k]
            return (np.sqrt(np.maximum(np.take_along_axis(D, order, 1), 0.0)),
                    np.take_along_axis(I, order, 1))
        Ds, Is = [], []
        for idx, off in self.shards:
            d, i = idx.search(Xq, min(k, idx.ntotal))
            Ds.append(d)
            Is.append(i + off)
        D = np.concatenate(Ds, axis=1)
        I = np.concatenate(Is, axis=1)
        if len(self.shards) == 1:
            return np.sqrt(np.maximum(D, 0.0)), I
        order = np.argsort(D, axis=1)[:, :k]
        D = np.take_along_axis(D, order, 1)
        I = np.take_along_axis(I, order, 1)
        return np.sqrt(np.maximum(D, 0.0)), I     # same as FaissIndex: Euclidean distances

    def reconstruct(self, i: int) -> np.ndarray:
        for idx, off in self.shards:
            if i < off + idx.ntotal:
                return idx.reconstruct(int(i - off))
        raise IndexError(i)
