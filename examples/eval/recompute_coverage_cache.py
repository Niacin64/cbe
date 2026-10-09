# -*- coding: utf-8 -*-
"""Rebuild the coverage cache (with per-atom k-NN distances) and also save structure-level coverage.npy.

Why a separate script: compare_models_4.py's diag stores only force errors, while compare_models_9.py
needs coverage on the same structures as the force errors. This script computes and caches it once:

  --cache  per-atom nearest-neighbour distances (k_sens, so changing k is instant)
  --cov    structure-level coverage npy (same order as refs)

Usage (local grace environment):
  PYTHONPATH=CBE python recompute_coverage_cache.py \
      --index-dir mptrj_index_all/ --refs ../../refs_full.xyz ../../refs_ood.xyz \
      --cache ../../k_sens_dist.npz --cov ../../coverage_836.npy
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mace.calculators import mace_mp
from cbe.index import FaissIndex
from cbe.preprocess import PCA
from cbe.data import read_structures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--cache", required=True, help="output: per-atom nearest-neighbour distances npz")
    ap.add_argument("--cov", required=True, help="output: structure-level coverage npy")
    ap.add_argument("--kmax", type=int, default=500, help="maximum number of neighbours to cache")
    ap.add_argument("--k", type=int, default=30, help="k used for the structure-level coverage")
    ap.add_argument("--bandwidth", type=float, default=0.2184)
    args = ap.parse_args()

    refs = []
    for p in args.refs:
        refs += read_structures(p)
    n_atoms_total = sum(len(a) for a in refs)
    print(f"{len(refs)} structures, {n_atoms_total} atoms", flush=True)

    small = mace_mp(model="small", device="cpu", default_dtype="float64")
    idx = FaissIndex().load(str(Path(args.index_dir) / "index.faiss"))
    pca = PCA.load(str(Path(args.index_dir) / "pca.npz"))

    D_all, bnd = [], [0]
    for i, a in enumerate(refs):
        q = pca.transform(np.asarray(small.get_descriptors(a, invariants_only=True),
                                     dtype=np.float64)).astype(np.float32)
        d, _ = idx.search(q, args.kmax)
        D_all.append(np.sqrt(np.maximum(d, 0)))       # faiss returns squared distances
        bnd.append(bnd[-1] + len(q))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(refs)} structures", flush=True)
    D = np.concatenate(D_all, axis=0)
    np.savez(args.cache, D=D.astype(np.float32), bnd=np.array(bnd))
    per_atom = np.mean(np.exp(-(D[:, :args.k] ** 2) / (2 * args.bandwidth ** 2)), axis=1)
    coverage = np.array([per_atom[b0:b1].min() for b0, b1 in zip(bnd[:-1], bnd[1:])])
    np.save(args.cov, coverage.astype(np.float32))
    print(f"saved -> {args.cache} (D {D.shape}) and {args.cov} (coverage for {len(coverage)} structures)")
    print(f"coverage: min={coverage.min():.3f} median={np.median(coverage):.3f} max={coverage.max():.3f}")


if __name__ == "__main__":
    main()
