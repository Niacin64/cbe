# -*- coding: utf-8 -*-
"""Calibrate a single θ_f per model on a **cross-difficulty** calib set (force errors only), then reuse it.

Calibration set (disjoint from the evaluation set):
  A. training-corpus samples: 300 MPtrj + 240 Alexandria structures (with PBE forces)
  B. hard structures: **stratified by group** half of the 623 test structures (in-dist / perturbed / random)
Evaluation set: the other half of the test set (312 structures).

Why hard structures are needed: θ = min{c : q̂σ(c) ≤ ε_f} requires the calibration set to contain
structures beyond tolerance; with corpus samples alone the model stays within tolerance and the band always
passes, θ collapses to 0 - the conformal requirement that calibration and deployment be exchangeable.

Coverage is corpus-matched: MPtrj-only uses MPtrj; Alexandria models take the per-structure max of both.

Outputs: calib_split.npz (held_out mask, length 623) + thetas.pkl (per-model θ_f)

Usage:
  PYTHONPATH=CBE:CBE/examples python calibrate_thetas.py [--eps-f 0.5]
"""
import argparse
import pickle
import sys
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

from cbe.calibration import ConformalCalibrator
from coverage_matched import load_dual, per_index_coverage, indexes_for
from figs_common import data, data_glob

H, K = 0.2184, 30


def _dual(npz, keep=None):
    return load_dual(data(npz) if not Path(npz).is_absolute() else npz, keep)


def _cov(dual, corpus, h, k):
    return np.max(np.stack([per_index_coverage(dual, t, h=h, k=k)
                            for t in indexes_for(corpus)], 0), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps-f", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--h", type=float, default=None)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(data("thetas.pkl")))
    ap.add_argument("--out-split", default=str(data("calib_split.npz")))
    args = ap.parse_args()

    from ase.io import read
    meta = pickle.load(open(data("roster_meta.pkl"), "rb"))

    # ---------- A. corpus samples ----------
    calib = read(str(data("refs_calib.xyz")), index=":")
    dual_c = _dual("dist_calib.npz")
    errs_c = {}
    for f in sorted(data_glob("eval_roster_calib_*.pkl")):
        errs_c.update(pickle.load(open(f, "rb")))
    print(f"{len(calib)} corpus-sample structures; {len(errs_c)} models already evaluated")

    # ---------- B. hard structures (stratified halves) ----------
    refs = read(str(data("refs_full.xyz")), index=":") + read(str(data("refs_ood.xyz")), index=":")
    keep = np.load(data("test_keep.npy"))
    refs = [a for a, k in zip(refs, keep) if k]
    groups = np.array([str(a.info.get("group", "?")) for a in refs])
    groups = np.where(np.char.startswith(groups, "rattle"), "perturbed", groups)
    rng = np.random.default_rng(args.seed)
    held = np.zeros(len(refs), bool)           # True = held out for evaluation
    for g in np.unique(groups):
        ii = rng.permutation(np.where(groups == g)[0])
        held[ii[len(ii) // 2:]] = True         # second half evaluates, first half calibrates
    dual_t = _dual("dist_test.npz", keep)
    errs_t = {}
    for f in sorted(data_glob("eval_roster_test_*.pkl")):
        errs_t.update(pickle.load(open(f, "rb")))
    print(f"{len(refs)} hard structures: calibration {int((~held).sum())} / evaluation {int(held.sum())}"
          f" (groups: { {g: int((groups == g).sum()) for g in np.unique(groups)} })")

    # ---------- calibration ----------
    theta = {}
    print(f"\n{'model':22s} {'corpus':26s} {'n_cal':>6s} {'theta_f':>6s}")
    for key, f_t in errs_t.items():
        f_t = np.asarray(f_t, float)
        if key not in errs_c or len(f_t) != len(refs):
            continue
        corpus = meta[key]["corpus"]
        cov_c = _cov(dual_c, corpus, args.h, args.k)
        cov_t = _cov(dual_t, corpus, args.h, args.k)
        cov = np.r_[cov_c, cov_t[~held]]
        f = np.r_[np.asarray(errs_c[key], float), f_t[~held]]
        theta[key] = float(ConformalCalibrator(alpha=args.alpha, seed=args.seed)
                           .fit(cov, f).threshold(args.eps_f))
        print(f"{meta[key]['name']:22s} {corpus:26s} {len(cov):6d} {theta[key]:6.3f}")

    np.savez(data("calib_split.npz"), held_out=held, keep=keep)
    with open(args.out, "wb") as fh:
        pickle.dump({"theta_f": theta, "eps_f": args.eps_f, "alpha": args.alpha,
                     "h": args.h, "k": args.k, "seed": args.seed,
                     "protocol": "corpus samples + stratified half of the hard structures; "
                                 "force errors only"}, fh)
    print(f"\nsaved -> {args.out} and {args.out_split}")
    print(f"theta_f range {min(theta.values()):.3f}-{max(theta.values()):.3f}")


if __name__ == "__main__":
    main()
