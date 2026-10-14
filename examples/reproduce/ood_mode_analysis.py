#!/usr/bin/env python
"""Per-mode evaluation of the extrapolation block, and a group-wise conformal audit.

Three analyses that the external referee asked for, none of which needs new DFT:

**A. Per-mode ranking on the 768 controlled-extrapolation structures.**
The block was built for calibration, so reporting a *calibrated* within-tolerance share on it
would be circular. Ranking is not: whether coverage orders structures by error is a property
of the score, not of theta, so a per-mode AUC can be read as evidence about the method's
central claim on non-bulk structures. Reported here per mode, with a bootstrap interval.

**B. Within-tolerance share by evaluation group.**
The conformal statement is marginal over the calibration distribution. The evaluation set
mixes 169 random cells, which are deliberately *not* in the calibration set, so the aggregate
marginal share (42-89% per model, median 47%) averages over a subgroup that is not exchangeable
with the calibrating one. Splitting the share by group is the honest way to report it, and the
random-cell entry is a result rather than an embarrassment: those cells should be flagged
(100% of them are). Three shares are distinguished in the paper: marginal, conditional on the
flag trusting a structure (90-99%, median 95%), and conditional on it rejecting one (19-73%,
median 31%).

**C. Sensitivity of theta to the calibration composition.**
Bootstrap over structures, and leave-one-mode-out over the ten controlled-extrapolation modes,
to put an interval on theta rather than asserting that it is stable.

Usage
-----
    PYTHONPATH=. python examples/reproduce/ood_mode_analysis.py
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "DATA"
sys.path.insert(0, str(ROOT / "CBE"))
sys.path.insert(0, str(ROOT))
from calibrate_calibration_set import coverage, tags_for       # noqa: E402

EPS = 0.5          # eV/A, the force tolerance
H = 0.2184


def union_auc(score: np.ndarray, bad: np.ndarray) -> float:
    """AUC of `score` separating `bad`; oriented so lower score = more likely bad."""
    r = np.argsort(np.argsort(score)) + 1
    nb, ng = int(bad.sum()), int((~bad).sum())
    if nb == 0 or ng == 0:
        return np.nan
    return float((r[bad].sum() - nb * (nb + 1) / 2) / (nb * ng))


def main() -> int:
    meta = pickle.load(open(DATA / "roster_meta.pkl", "rb"))
    th = pickle.load(open(DATA / "thetas.pkl", "rb"))["theta_f"]
    dn = np.load(DATA / "combined_analysis.npz")
    modes = dn["ood_modes"]
    idx = [str(x) for x in dn["ood_index"]]
    OOD = {"mp": dn["cov_new_mp"], "alex": dn["cov_new_alex"], "omat": dn["cov_new_omat"]}
    e_ood = {k[4:]: dn[k] for k in dn.files if k.startswith("err_")}

    errs_c, errs_t = {}, {}
    for f in sorted(DATA.glob("eval_roster_calib_*.pkl")):
        errs_c.update(pickle.load(open(f, "rb")))
    for f in sorted(DATA.glob("eval_roster_test_*.pkl")):
        errs_t.update(pickle.load(open(f, "rb")))

    keep = np.load(DATA / "test_keep.npy")
    refs = (ase_read(str(DATA / "refs_full.xyz"), index=":")
            + ase_read(str(DATA / "refs_ood.xyz"), index=":"))
    refs = [a for a, k_ in zip(refs, keep) if k_]
    grp = np.array([str(a.info.get("group", "?")).split("_")[0] for a in refs])
    grp = np.where(np.char.startswith(grp, "rattle"), "perturbed", grp)

    models = [k for k in sorted(th) if k in meta and k in e_ood]
    cov_ood = {k: np.max(np.stack([OOD[t] for t in tags_for(meta[k]["corpus"])]), axis=0)
               for k in models}
    print(f"{len(models)} models; {len(idx)} extrapolation structures "
          f"across {len(set(modes))} modes\n")

    # ---- A. per-mode ranking -----------------------------------------------------------
    print("=== A. per-mode ranking on the controlled-extrapolation block ===")
    print(f"  {'mode':12s} {'n':>4s} {'median err':>11s} {'AUC(>tol)':>10s} "
          f"{'AUC 95% CI':>18s}")
    rng = np.random.default_rng(0)
    per_mode = {}
    for m in sorted(set(modes), key=lambda x: -np.nanmedian(
            np.nanmedian(np.stack([e_ood[k] for k in models]), axis=0)[modes == x])):
        sel = modes == m
        # median over models, per structure
        e = np.nanmedian(np.stack([e_ood[k][sel] for k in models]), axis=0)
        c = np.nanmedian(np.stack([cov_ood[k][sel] for k in models]), axis=0)
        ok = np.isfinite(e) & np.isfinite(c)
        bad = e[ok] > EPS
        a = union_auc(-c[ok], bad)
        bs = []
        for _ in range(400):
            j = rng.integers(0, ok.sum(), ok.sum())
            bs.append(union_auc(-c[ok][j], bad[j]))
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        per_mode[m] = dict(n=int(ok.sum()), median_err=float(np.nanmedian(e[ok])),
                           auc=a, lo=float(lo), hi=float(hi))
        print(f"  {m:12s} {ok.sum():>4d} {np.nanmedian(e[ok]):>11.3f} {a:>10.3f} "
              f"   [{lo:.3f}, {hi:.3f}]")

    # also: ranking pooled over the whole block, per model
    print(f"\n  per-model AUC over the whole 768-structure block:")
    aucs = []
    for k in models:
        e, c = e_ood[k], cov_ood[k]
        ok = np.isfinite(e) & np.isfinite(c)
        a = union_auc(-c[ok], e[ok] > EPS)
        aucs.append(a)
    aucs = np.array(aucs)
    print(f"    median {np.nanmedian(aucs):.3f}  range {np.nanmin(aucs):.3f}-"
          f"{np.nanmax(aucs):.3f}   (n={np.isfinite(aucs).sum()} models)")
    # per-mode AUC summarised over models
    print(f"\n  per-mode AUC, median over models:")
    for m in sorted(set(modes)):
        sel = modes == m
        aa = []
        for k in models:
            e, c = e_ood[k][sel], cov_ood[k][sel]
            ok = np.isfinite(e) & np.isfinite(c)
            if ok.sum() > 5:
                aa.append(union_auc(-c[ok], e[ok] > EPS))
        if aa:
            print(f"    {m:12s} {np.nanmedian(aa):.3f}  (range "
                  f"{np.nanmin(aa):.3f}-{np.nanmax(aa):.3f}, n={len(aa)})")

    # ---- B. within-tolerance share by group --------------------------------------------
    print("\n=== B. within-tolerance share on the evaluation set, by group ===")
    print(f"  {'group':12s} {'n':>4s} {'share within eps_f':>19s} "
          f"{'median err':>11s} {'flag rate':>10s}")
    share_tbl = {}
    for g in ("in-dist", "perturbed", "random"):
        sel = grp == g
        within, med, flagged = [], [], []
        for k in models:
            cov = coverage(DATA / "dist_test.npz", tags_for(meta[k]["corpus"]), keep)
            e = np.asarray(errs_t[k], float)
            ok = np.isfinite(cov) & np.isfinite(e) & sel
            if ok.sum() == 0:
                continue
            within.append(np.mean(e[ok] <= EPS) * 100)
            med.append(np.median(e[ok]))
            flagged.append(np.mean(cov[ok] < th[k]) * 100)
        share_tbl[g] = dict(n=int(sel.sum()),
                            within=float(np.median(within)),
                            within_range=(float(np.min(within)), float(np.max(within))),
                            median_err=float(np.median(med)),
                            flag=float(np.median(flagged)))
        print(f"  {g:12s} {sel.sum():>4d} {np.median(within):>18.0f}% "
              f"{np.median(med):>11.3f} {np.median(flagged):>9.0f}%")
    w = np.array([share_tbl[g]["within"] for g in share_tbl])
    print(f"\n  aggregate over the three groups would be "
          f"{np.average(w, weights=[share_tbl[g]['n'] for g in share_tbl]):.0f}%; "
          f"the groups differ by {w.max()-w.min():.0f} points.")

    # ---- C. sensitivity of theta -------------------------------------------------------
    print("\n=== C. sensitivity of theta to the calibration composition ===")
    print("  bootstrap over calibration structures (200 resamples):")
    boot = {k: [] for k in models}
    cov_c, err_c = {}, {}
    for k in models:
        c = np.r_[coverage(DATA / "dist_calib.npz", tags_for(meta[k]["corpus"])),
                  cov_ood[k]]
        e = np.r_[np.asarray(errs_c[k], float), e_ood[k]]
        ok = np.isfinite(c) & np.isfinite(e)
        cov_c[k], err_c[k] = c[ok], e[ok]
    import sys as _s
    _s.path.insert(0, str(ROOT / "CBE"))
    from cbe.calibration import ConformalCalibrator
    for k in models:
        c, e = cov_c[k], err_c[k]
        n = len(c)
        for _ in range(200):
            j = rng.integers(0, n, n)
            try:
                boot[k].append(float(ConformalCalibrator(alpha=0.1, seed=0)
                                     .fit(c[j], e[j]).threshold(EPS)))
            except Exception:
                pass
    widths = [(np.percentile(boot[k], 97.5) - np.percentile(boot[k], 2.5))
              for k in models if len(boot[k]) > 50]
    print(f"    theta 95% bootstrap interval width: median {np.median(widths):.3f}, "
          f"max {np.max(widths):.3f}")

    print("\n  leave-one-mode-out (theta recomputed without one mode at a time):")
    print(f"    {'dropped mode':14s} {'median |dtheta|':>16s} {'max |dtheta|':>14s}")
    for m in sorted(set(modes)):
        dr = []
        for k in models:
            c = np.r_[coverage(DATA / "dist_calib.npz", tags_for(meta[k]["corpus"])),
                      cov_ood[k][modes != m]]
            e = np.r_[np.asarray(errs_c[k], float), e_ood[k][modes != m]]
            ok = np.isfinite(c) & np.isfinite(e)
            try:
                t = float(ConformalCalibrator(alpha=0.1, seed=0)
                          .fit(c[ok], e[ok]).threshold(EPS))
                dr.append(abs(t - th[k]))
            except Exception:
                pass
        if dr:
            print(f"    {m:14s} {np.median(dr):>16.3f} {np.max(dr):>14.3f}")

    json.dump(dict(per_mode=per_mode, within_tolerance=share_tbl,
                   per_model_auc_ood=list(map(float, np.nan_to_num(aucs, nan=-1))),
                   theta_boot_width_median=float(np.median(widths))),
              open(DATA / "ood_mode_analysis.json", "w"), indent=1)
    print(f"\nwrote {DATA / 'ood_mode_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
