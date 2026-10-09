# -*- coding: utf-8 -*-
"""Cross-difficulty transfer experiment for θ (JCTC main text section 3.4 / npj SI).

Calibrate each model's θ_f on four calibration sets of different composition, then evaluate on
the fixed 312 held-out structures:
  A corpus-only        : 540 corpus structures (300 MPtrj + 240 Alexandria)
  B corpus + mild      : A + half of the "mildly perturbed" structures (force error below the
                         median of the perturbed regime)
  C corpus + hard + rnd: A + half of the strongly perturbed + half of the random   <- main-text protocol
  D hard + random only : half of the strongly perturbed + half of the random (no corpus structures)
All four configurations contain the same number of difficult structures; only the
**mixing ratio** changes.

Reported: θ_f, flagged rate, empirical coverage (overall and by group), and the agreement
between the flagged rate and the true out-of-tolerance rate.
Everything comes from the cache; no model runs are needed (<2 min).
Usage:
  PYTHONPATH=CBE:CBE/examples python theta_transfer.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

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
from cbe.calibration import ConformalCalibrator
from cbe.data import read_structures
from coverage_matched import load_dual, slice_dual, matched_coverage
from figs_common import data, data_glob

EPS, ALPHA = 0.5, 0.1


def main():
    refs = read_structures(str(data("refs_full.xyz"))) + read_structures(str(data("refs_ood.xyz")))
    keep = np.load(data("test_keep.npy"))
    refs = [a for a, k in zip(refs, keep) if k]
    grp = np.array([str(a.info.get("group", "?")) for a in refs])
    grp = np.where(np.char.startswith(grp, "rattle"), "perturbed", grp)
    dual = load_dual(data("dist_test.npz"), keep)
    errs = {}
    for f in sorted(data_glob("eval_roster_test_*.pkl")):
        errs.update(pickle.load(open(f, "rb")))
    meta = pickle.load(open(data("roster_meta.pkl"), "rb"))

    held = np.load(data("calib_split.npz"))["held_out"]       # True = evaluation set (312)
    cal_dual = load_dual(data("dist_calib.npz"))              # 540 corpus structures
    cal_errs = {}
    for f in sorted(data_glob("eval_roster_calib_*.pkl")):
        cal_errs.update(pickle.load(open(f, "rb")))
    n_cal = 540                                              # number of corpus structures in refs_calib.xyz

    ev, add = held, ~held
    g_add, g_ev = grp[add], grp[ev]
    dual_ev, dual_add = slice_dual(dual, ev), slice_dual(dual, add)

    probe = np.asarray(errs["mace-mp-0-medium"], float)[add]
    pert = np.where(g_add == "perturbed")[0]
    med = np.nanmedian(probe[pert])
    mild, hard = pert[probe[pert] <= med], pert[probe[pert] > med]
    rand = np.where(g_add == "random")[0]
    rng = np.random.RandomState(0)
    sel = {name: rng.choice(idx, size=len(idx) // 2, replace=False)
           for name, idx in [("mild", mild), ("hard", hard), ("rand", rand)]}

    def mk(extra):
        m = np.zeros(n_cal + len(g_add), bool)
        m[:n_cal] = True
        m[n_cal + extra] = True
        return m

    sets = {
        "A corpus-only": mk(np.array([], int)),
        "B corpus+mild": mk(sel["mild"]),
        "C corpus+hard+rnd": mk(np.r_[sel["hard"], sel["rand"]]),
        "D hard+rnd only": np.r_[np.zeros(n_cal, bool),
                                 np.isin(np.arange(len(g_add)), np.r_[sel["hard"], sel["rand"]])],
    }
    print("calibration sets: " + ", ".join(f"{k}={int(v.sum())}" for k, v in sets.items()) + f"; evaluation {ev.sum()}")

    models = [k for k in errs if len(np.asarray(errs[k], float)) == len(refs) and k in cal_errs]
    rows = {}
    for key in models:
        corpus = meta[key]["corpus"]
        cov_all = np.r_[matched_coverage(cal_dual, corpus), matched_coverage(dual_add, corpus)]
        f_all = np.r_[np.asarray(cal_errs[key], float), np.asarray(errs[key], float)[add]]
        cov_ev = matched_coverage(dual_ev, corpus)
        f_ev = np.asarray(errs[key], float)[ev]
        ok = f_ev <= EPS
        out = {}
        for name, mask in sets.items():
            th = float(ConformalCalibrator(alpha=ALPHA, seed=0).fit(cov_all[mask], f_all[mask]).threshold(EPS))
            out[name] = dict(theta=th, flag=float((cov_ev < th).mean()) * 100,
                             cov=float(ok.mean()) * 100,
                             cov_in=float(ok[g_ev == "in-dist"].mean()) * 100,
                             cov_p=float(ok[g_ev == "perturbed"].mean()) * 100,
                             cov_r=float(ok[g_ev == "random"].mean()) * 100)
        rows[key] = out

    print(f"\n{'config':18s} {'θ med':>7s} {'θ range':>14s} {'flag% med':>10s} {'cov% med':>9s} "
          f"{'cov(in)':>8s} {'cov(pert)':>10s} {'cov(rand)':>10s} {'r(flag,miss)':>13s} {'dev pp':>7s}")
    for name in sets:
        th = np.array([rows[k][name]["theta"] for k in models])
        fl = np.array([rows[k][name]["flag"] for k in models])
        cv = np.array([rows[k][name]["cov"] for k in models])
        ci = np.median([rows[k][name]["cov_in"] for k in models])
        cp = np.median([rows[k][name]["cov_p"] for k in models])
        cr = np.median([rows[k][name]["cov_r"] for k in models])
        miss = 100 - cv
        print(f"{name:18s} {np.median(th):7.3f} {th.min():6.3f}-{th.max():6.3f} {np.median(fl):10.0f} "
              f"{np.median(cv):9.1f} {ci:8.0f} {cp:10.0f} {cr:10.0f} "
              f"{spearmanr(fl, miss).statistic:+13.3f} {np.mean(np.abs(fl - miss)):7.1f}")
    print("\n(cov% = fraction of the held-out set with e_F ≤ 0.5 eV/Å, nominal 90%; dev pp = mean absolute deviation between the flagged rate and the true out-of-tolerance rate)")


if __name__ == "__main__":
    main()
