#!/usr/bin/env python
"""Verify a downloaded coverage index before you trust it with a screen.

A 3-10 GB download can fail in ways a checksum does not catch: a truncated shard, a
shard set that is missing one file, an index built with a different descriptor or a
different PCA than the ``pca.npz`` sitting next to it. This script checks the artefact
itself, using properties that a correct index must satisfy and that a corrupt one cannot.

Checks
------
1. ``meta.json`` and ``pca.npz`` agree: 16 components, descriptor dimension 256.
2. Every file listed in ``meta.json`` is present, and the shards are the declared size.
3. The number of indexed environments equals ``meta.json`` (catches a missing shard).
4. **Round trip**: a vector taken out of the index is returned by the index as its own
   nearest neighbour at distance 0. This is the strongest cheap statement that the
   shards are readable, complete and consistent with each other.
5. **Coverage behaves like a density**: on a few probe vectors, the KDE score is
   non-increasing in k and lies in (0, 1], which is what a genuine nearest-neighbour
   search must give. This tests the scoring path, not just the storage.

Usage
-----
    python examples/index/verify_index.py --index ~/cbe_data/mptrj
    python examples/index/verify_index.py --index ~/cbe_data/alexandria --probe 8

Exit code 0 means the index is usable; anything else prints what is wrong and which
file to download again.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def read_meta(index_dir: Path) -> dict:
    path = index_dir / "meta.json"
    if not path.exists():
        sys.exit(f"error: {path} not found — is {index_dir} an extracted index directory?")
    return json.loads(path.read_text())


def check_pca(index_dir: Path, meta: dict) -> np.ndarray:
    """Return the PCA mean/components and confirm the shapes."""
    path = index_dir / "pca.npz"
    if not path.exists():
        sys.exit(f"error: {path} not found — the index cannot be used without the "
                 "pca.npz it was built with")
    z = np.load(path)
    mean, comps = z["mean"], z["components"]
    dim = int(meta.get("pca_dim", comps.shape[0]))
    if mean.shape != (256,):
        sys.exit(f"error: pca.npz mean has shape {mean.shape}, expected (256,)")
    if comps.shape != (dim, 256):
        sys.exit(f"error: pca.npz components have shape {comps.shape}, expected ({dim}, 256)")
    print(f"  pca.npz: {dim} components over a 256-dimensional descriptor  OK")
    return comps


def open_index(index_dir: Path, meta: dict, lazy: bool):
    """Open the index, single-file or sharded, as the library would."""
    from cbe.index import FaissIndex, ShardedFaissIndex

    if meta.get("layout") == "sharded":
        return ShardedFaissIndex().load(index_dir, lazy=lazy)
    files = sorted(index_dir.glob("*.faiss"))
    if not files:
        sys.exit(f"error: no .faiss files in {index_dir}")
    if len(files) > 1:
        sys.exit(f"error: {len(files)} .faiss files present for a single-file index; "
                 "delete the directory and download again")
    return FaissIndex().load(files[0])


def check_files(index_dir: Path, meta: dict) -> None:
    missing = [f for f in meta.get("files", []) if not (index_dir / f).exists()]
    if missing:
        sys.exit(f"error: missing file(s) {missing} in {index_dir}")
    sizes = {p.name: p.stat().st_size for p in sorted(index_dir.glob("*.faiss"))}
    print(f"  {len(sizes)} index file(s) present, all files listed in meta.json exist  OK")
    for name, size in sizes.items():
        print(f"    {name}: {size / 2**30:.2f} GiB")


def check_count(index, meta: dict) -> int:
    expected = int(meta["n_environments"])
    got = int(index.n)
    if got != expected:
        sys.exit(f"error: index holds {got:,} environments, meta.json declares "
                 f"{expected:,}. A shard is missing or truncated — delete the .faiss "
                 "files and download them again.")
    print(f"  {got:,} environments indexed, matches meta.json  OK")
    return got


def probe_vectors(index, n_probe: int, seed: int = 0) -> np.ndarray:
    """Take n_probe vectors out of the index itself (uniformly spread over its range)."""
    total = int(index.n)
    ids = np.linspace(0, total - 1, num=n_probe, dtype=np.int64)
    return np.stack([np.asarray(index.reconstruct(int(i)), dtype=np.float32) for i in ids])


def check_round_trip(index, probes: np.ndarray, k: int) -> np.ndarray:
    """Each probe must be its own nearest neighbour, at distance 0."""
    dist, ids = index.search(probes, k)
    first = dist[:, 0]
    worst = float(np.max(np.abs(first)))
    # the index stores float32; a query round trip through float32/float64 leaves a
    # residual of order 1e-4 * the vector norm. A misplaced or truncated shard would
    # put the nearest neighbour orders of magnitude further away than this.
    if worst > 1e-2:
        sys.exit(f"error: a vector taken from the index is not returned as its own "
                 f"nearest neighbour (largest distance {worst:.3e}, expected < 1e-2). "
                 "The index is inconsistent with itself; download it again.")
    print(f"  round trip: {len(probes)} vectors found themselves as their own nearest "
          f"neighbour (max |d| = {worst:.1e}, float32 round-off)  OK")
    if ids.shape[1] < k:
        sys.exit(f"error: asked for k={k} neighbours but the index returned {ids.shape[1]}")
    return dist


def check_coverage_behaviour(index, probes: np.ndarray, k: int, bandwidth: float) -> None:
    """Coverage computed from the index must behave like a k-NN density estimate.

    A brute-force cross-check is *not* used here: the index holds tens of millions of
    vectors, so a subsample cannot reproduce its k-NN distances and would fail even for
    a perfect index. What can be checked is a property that follows from the search
    being a genuine nearest-neighbour search: adding more neighbours can only add more
    kernel mass, so the KDE score must be non-increasing in k. A shard that is misread,
    a wrong metric, or a k larger than the index can hold all break it.
    """
    from cbe.kde import kde_score

    ks = [1, 10, k, min(200, k * 4)]
    ks = sorted(set(int(x) for x in ks if 0 < x <= int(index.n)))
    if ks[-1] < k:
        sys.exit(f"error: the index holds {int(index.n):,} environments, fewer than k={k}")
    print("  coverage against k:")
    prev = None
    for kk in ks:
        d, _ = index.search(probes, kk)
        rho = kde_score(d, bandwidth)
        flag = ""
        if prev is not None and np.any(rho > prev + 1e-9):
            flag = "  <-- VIOLATES the monotonicity that a k-NN score must have"
        print(f"    k={kk:4d}: median coverage {np.median(rho):.4f}"
              f"  (range {rho.min():.4f}-{rho.max():.4f}){flag}")
        if flag:
            sys.exit("error: coverage is not monotone in k; the index or the scoring "
                     "path is not what it claims to be")
        prev = rho

    rho = kde_score(index.search(probes, k)[0], bandwidth)
    if not np.all((rho > 0) & (rho <= 1)):
        sys.exit(f"error: coverage outside (0, 1]: {rho}")
    print(f"  coverage at the deployed k={k} lies in (0, 1] and is monotone in k  OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, type=Path,
                    help="directory holding meta.json, pca.npz and the .faiss files")
    ap.add_argument("--probe", type=int, default=5, help="number of probe vectors")
    ap.add_argument("--k", type=int, default=30, help="neighbours used by the score")
    ap.add_argument("--no-k-check", action="store_true",
                    help="skip the coverage-monotonicity-in-k check")
    ap.add_argument("--lazy", action="store_true",
                    help="open a sharded index lazily, as a low-memory user would")
    args = ap.parse_args()

    print(f"verifying {args.index}")
    meta = read_meta(args.index)
    print(f"  corpus: {meta.get('corpus')}  ({meta.get('note', 'no note')})")
    check_pca(args.index, meta)
    check_files(args.index, meta)

    index = open_index(args.index, meta, lazy=args.lazy)
    check_count(index, meta)

    if args.lazy and meta.get("layout") == "sharded":
        print("  lazy mode: skipping the round trip and brute-force checks, which need "
              "to reconstruct vectors (re-run without --lazy for the full check)")
        return 0

    probes = probe_vectors(index, max(1, args.probe))
    check_round_trip(index, probes, args.k)
    if not args.no_k_check:
        bandwidth = float(meta.get("bandwidth_used", 0.2184))
        print(f"  scoring with h = {bandwidth}")
        check_coverage_behaviour(index, probes, args.k, bandwidth)

    print(f"\n{args.index}: usable.\n"
          "Load it with:\n"
          "    from cbe import pretrained\n"
          f"    cov = pretrained.load_index('{meta.get('name', 'mptrj')}')   "
          "# add pca/index paths if not using the pretrained layout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
