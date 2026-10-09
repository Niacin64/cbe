# -*- coding: utf-8 -*-
"""Per-atom nearest-neighbour distances against any number of corpus indexes (local faiss / CPU).

Purpose: compute k-NN distances "to each corpus index" for calibration, test, cathode or grain-boundary sets,
cached as npz (`D_<tag>` + `bnd`): coverage for any bandwidth h is instant, and corpora can be unioned.

When memory is tight, use --only to run one index at a time (each is loaded, then released when done),
then --merge to combine the partial results.

Usage:
  # all three indexes in one go (when memory >= 8 GB)
  PYTHONPATH=CBE:CBE/examples python coverage_multi.py \
      --refs ../../refs_calib.xyz --out ../../dist_calib.npz \
      --index mp=CBE/examples/mptrj_index_all --index alex=alexandria_index_full \
      --index omat=omat24_index --k 200

  # tight memory: one at a time (each released when done), merge at the end
  ... --only omat --save-partial
  ... --merge dist_calib_mp.npz dist_calib_alex.npz dist_calib_omat.npz --out dist_calib.npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent.parent


def _load_refs(paths, filter_path=None):
    from ase.io import read
    refs = []
    for p in paths:
        pp = Path(p)
        if not pp.exists():
            pp = data(p)
        refs += read(str(pp), index=":")
    if filter_path:
        keep = np.load(filter_path if Path(filter_path).exists() else data(filter_path))
        refs = [a for a, k in zip(refs, keep) if k]
    return refs


def _open_index(d, tag):
    """Supports both a single index.faiss and sharded shard_*.faiss; lazy loading."""
    from cbe.index import FaissIndex, ShardedFaissIndex
    shards = sorted(Path(d).glob("shard_*.faiss"))
    if shards:
        return ShardedFaissIndex().load(str(d), lazy=True)
    return FaissIndex().load(str(Path(d) / "index.faiss"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="+", default=None)
    ap.add_argument("--filter", default=None, help="boolean mask npy (e.g. test_keep.npy)")
    ap.add_argument("--index", action="append", default=[], metavar="TAG=DIR",
                    help="may be given several times, e.g. --index mp=CBE/examples/mptrj_index_all")
    ap.add_argument("--only", default=None, help="run only one tag (use when memory is tight)")
    ap.add_argument("--save-partial", action="store_true", help="with --only, save to <out>_<tag>.npz")
    ap.add_argument("--merge", nargs="+", default=None, help="only merge these partial npz files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # ---------- merge-only mode ----------
    if args.merge:
        parts = [np.load(p if Path(p).exists() else data(p)) for p in args.merge]
        out = {"bnd": parts[0]["bnd"]}
        for p, d in zip(args.merge, parts):
            for key in d.files:
                if key.startswith("D_"):
                    out[key] = d[key]
        np.savez_compressed(args.out, **out)
        print(f"merged {len(parts)} parts -> {args.out} ({sorted(out)})")
        return

    specs = []
    for item in args.index:
        tag, _, d = item.partition("=")
        specs.append((tag, Path(d) if Path(d).is_absolute() else ROOT / d))
    if not specs:
        raise SystemExit('at least one --index tag=dir is required')
    if args.only:
        specs = [s for s in specs if s[0] == args.only]
        if not specs:
            raise SystemExit(f"--only {args.only} is not among the --index entries")

    refs = _load_refs(args.refs, args.filter)
    bnd = np.cumsum([0] + [len(a) for a in refs]).astype(int)
    n_atoms = int(bnd[-1])
    print(f"{len(refs)} structures, {n_atoms} atoms", flush=True)

    from mace.calculators import mace_mp
    from cbe.preprocess import PCA
    t0 = time.time()
    calc = mace_mp(model="small", device=args.device, default_dtype="float64")
    X = np.concatenate([np.asarray(calc.get_descriptors(a, invariants_only=True),
                                   dtype=np.float64) for a in refs], axis=0)
    print(f"descriptors {X.shape} ({time.time()-t0:.0f}s)", flush=True)

    out = {"bnd": bnd, "n_atoms": n_atoms}
    for tag, d in specs:
        pca = PCA.load(str(d / "pca.npz"))
        if not pca:
            raise FileNotFoundError(f"{d}/pca.npz is missing (it must accompany the index)")
        Q = pca.transform(X).astype(np.float32)
        idx = _open_index(d, tag)
        t1 = time.time()
        D, _ = idx.search(Q, args.k)
        out[f"D_{tag}"] = D.astype(np.float32)
        print(f"  {tag}: {d.name} -> D{out[f'D_{tag}'].shape} ({time.time()-t1:.0f}s)", flush=True)

    dest = str(args.out).replace(".npz", f"_{args.only}.npz") if (args.only and args.save_partial) \
        else args.out
    np.savez_compressed(dest, **out)
    print(f"saved -> {dest}")
    for h in (0.2184,):
        for tag, _ in specs:
            pa = np.mean(np.exp(-(out[f"D_{tag}"] ** 2) / (2 * h ** 2)), axis=1)
            cov = np.array([pa[a:b].min() for a, b in zip(bnd[:-1], bnd[1:])])
            print(f"  h={h} {tag:6s}: median coverage {np.median(cov):.3f}")


if __name__ == "__main__":
    main()
