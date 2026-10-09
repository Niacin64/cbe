# -*- coding: utf-8 -*-
"""M4 baseline: multi-model disagreement (ensemble / committee disagreement).

The cache holds only per-structure scalar errors, not per-atom forces, so this baseline has to
run the models again. For the 312 held-out structures, forces are computed once with 3
**MPtrj-only** MACE-MP-0 models (small / medium / large, i.e. the same corpus at different
capacities):

  U_dis = per-atom cross-model force standard deviation, then averaged over atoms   (per structure, eV/Å)
  e_ens = mean per-atom |<F>_ensemble − F_PBE|                                      (ensemble-mean error)
  e_mem = per-atom MAE of each individual member

Missing checkpoints are downloaded from the mace-foundations release (see the URL in the SUMMARY
output; remember to verify the zip first -- this project was once bitten by a truncated
download).

`--save-forces` additionally writes the per-atom forces to disk (`DATA/ensemble_forces.npz`,
float32, ~1.1 GB) for archiving: with it, every ensemble-related number in the paper can be
recomputed straight from the repository without re-running the 936 single points.
Usage:
  PYTHONPATH=CBE:CBE/examples TORCH_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python ensemble_baseline.py --workers 4
"""
import argparse
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- make the sibling example directories importable -------------------------
# this file lives in examples/<group>/; the shared helpers live in examples/common/
# and examples/index/, so both are added to sys.path whatever the working directory.
_HERE = Path(__file__).resolve().parent
for _extra in ("common", "index", _HERE.name):
    _p = _HERE.parent / _extra
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent

from figs_common import data

MEMBERS = [("mace-mp-0-small", "small"), ("mace-mp-0-medium", "medium"),
           ("mace-mp-0-large", "large")]


def load_held_out():
    from ase.io import read
    refs = read(str(data("refs_full.xyz")), index=":") + read(str(data("refs_ood.xyz")), index=":")
    keep = np.load(data("test_keep.npy"))
    refs = [a for a, k in zip(refs, keep) if k]
    held = np.load(data("calib_split.npz"))["held_out"]
    refs = [a for a, k in zip(refs, held) if k]
    return refs


def worker(job):
    lo, hi = job
    from mace.calculators import mace_mp
    refs = load_held_out()[lo:hi]
    out = {}
    for key, tag in MEMBERS:
        p = ROOT / ".cache/mace_foundations" / f"mace-{tag}.model"
        if not p.exists():
            raise FileNotFoundError(f"missing checkpoint {p} (see pull_checkpoints.sh)")
        calc = mace_mp(model=str(p), device="cpu", default_dtype="float64")
        forces = []
        for a in refs:
            a.calc = calc
            forces.append(np.asarray(a.get_forces(), float))     # structures may have different atom counts
        out[key] = forces
    return lo, hi, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(data("ensemble_test.npz")))
    ap.add_argument("--save-forces", action="store_true",
                    help="also write the per-atom forces to DATA/ensemble_forces.npz (for archiving, ~1.1 GB)")
    args = ap.parse_args()

    refs = load_held_out()
    ref_f = [np.asarray(a.arrays["REF_forces"], float) for a in refs]   # structures have different atom counts
    n = len(refs)
    print(f"held-out {n} structures; members {[m[0] for m in MEMBERS]}; workers={args.workers}", flush=True)

    bounds = np.linspace(0, n, args.workers + 1).astype(int)
    jobs = [(int(bounds[i]), int(bounds[i + 1])) for i in range(args.workers)]
    F = {k: [None] * n for k, _ in MEMBERS}
    t0 = time.time()
    with get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for lo, hi, out in pool.imap_unordered(worker, jobs):
            for k in F:
                F[k][lo:hi] = list(out[k])
            print(f"  done {hi}/{n}  ({time.time()-t0:.0f}s)", flush=True)

    Fs = [[F[k][i] for k, _ in MEMBERS] for i in range(n)]           # per structure: [M arrays of (Na,3)]
    dis = np.array([np.mean(np.linalg.norm(np.stack(fs) - np.stack(fs).mean(0, keepdims=True),
                                           axis=-1)) for fs in Fs])
    e_ens = np.array([np.mean(np.linalg.norm(np.stack(fs).mean(0) - ref_f[i], axis=-1))
                      for i, fs in enumerate(Fs)])
    e_mem = np.stack([np.array([np.mean(np.linalg.norm(fs[m] - ref_f[i], axis=-1))
                                for i, fs in enumerate(Fs)]) for m in range(len(MEMBERS))], 0)
    np.savez(args.out, U_dis=dis, e_ens=e_ens, e_members=e_mem,
             members=np.array([k for k, _ in MEMBERS]))
    if args.save_forces:
        # different atom counts -> can only be stored as an object array (one (M, Na, 3) float32 per structure)
        per_struct = np.empty(n, dtype=object)
        for i in range(n):
            per_struct[i] = np.stack([F[k][i] for k, _ in MEMBERS], 0).astype(np.float32)
        natoms = np.array([per_struct[i].shape[1] for i in range(n)], np.int32)
        out_f = Path(str(args.out)).with_name("ensemble_forces.npz")
        np.savez_compressed(out_f, forces=per_struct, natoms=natoms,
                            members=np.array([k for k, _ in MEMBERS]),
                            note=np.array(["object array over the 312 held-out structures in order; "
                                           "entry i has shape (3 members, natoms[i], 3), float32"]))
        print(f"per-atom forces saved -> {out_f} ({per_struct.nbytes/1e6:.0f} MB uncompressed, "
              f"structures of {natoms.min()}–{natoms.max()} atoms)")
    print(f"\ndisagreement U_dis: median {np.median(dis)*1000:.0f} meV/Å, "
          f"range {dis.min()*1000:.0f}–{dis.max()*1000:.0f} meV/Å")
    print(f"ensemble-mean error e_ens: median {np.median(e_ens)*1000:.0f} meV/Å")
    for m, (k, _) in enumerate(MEMBERS):
        print(f"  {k:22s} per-atom MAE median {np.median(e_mem[m])*1000:6.0f} meV/Å")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
