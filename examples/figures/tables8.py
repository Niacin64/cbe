# -*- coding: utf-8 -*-
"""Generate Table 1 (test set, grouped by corpus) and Table 2 (cathode, dual Spearman + MAE
above/below θ).

Usage:
  PYTHONPATH=CBE:CBE/examples python tables8.py [--table 1|2]
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

from scipy.stats import spearmanr
from coverage_matched import load_dual, slice_dual, matched_coverage, H_DEFAULT
from figs_common import CATHODE_DROP, GROUP_ORDER, corpus_group, data, data_glob
from fig_cathode8 import model_voltage, LEGACY


def test_data():
    from cbe.data import read_structures
    refs = read_structures(str(data("refs_full.xyz"))) + read_structures(str(data("refs_ood.xyz")))
    keep = np.load(data("test_keep.npy"))
    refs = [a for a, k in zip(refs, keep) if k]
    dual = load_dual(data("dist_test.npz"), keep)
    errs = {}
    for f in sorted(data_glob("eval_roster_test_*.pkl")):
        errs.update(pickle.load(open(f, "rb")))
    return refs, dual, errs


def cathode_data():
    d = pickle.load(open(ROOT / "case_cathode/mlip_vs_dft.pkl", "rb"))
    vr = pickle.load(open(ROOT / "case_cathode/voltage_results.pkl", "rb"))
    tpl = np.asarray(d["templates"])
    dual = load_dual(data("dist_cathode.npz"))
    errs, volt = {}, {}
    for f in sorted(data_glob("eval_roster_cathode_*.pkl")):
        errs.update(pickle.load(open(f, "rb")))
    for p in ["case_cathode/mlip_voltage_grace.pkl", "case_cathode/mlip_voltage_mlip7.pkl",
              "case_cathode/voltage_roster_grace.pkl", "case_cathode/voltage_roster_mlip7a.pkl",
              "case_cathode/voltage_roster_mlip7b.pkl"]:
        for k, v in pickle.load(open(ROOT / p, "rb")).items():   # p looks like case_cathode/xxx.pkl
            volt[LEGACY.get(k, k)] = v
    keep = np.ones(len(tpl), bool); keep[CATHODE_DROP] = False
    return tpl, dual, errs, volt, np.asarray(vr["V_dft"]["DFT"], float), keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="all", choices=["1", "2", "all"])
    ap.add_argument("--eps-f", type=float, default=0.2)
    ap.add_argument("--eps-v", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--h", type=float, default=None)
    ap.add_argument("--k", type=int, default=30)
    args = ap.parse_args()

    meta = pickle.load(open(data("roster_meta.pkl"), "rb"))
    TH = pickle.load(open(data("thetas.pkl"), "rb"))
    split = np.load(data("calib_split.npz"))

    if args.table in ("1", "all"):
        refs, dual, errs = test_data()
        n_all = len(refs)
        held = np.load(data("calib_split.npz"))["held_out"]   # held-out half only for the statistics
        refs = [a for a, k in zip(refs, held) if k]
        dual = slice_dual(dual, held)
        errs = {k: np.asarray(v, float)[held] for k, v in errs.items()}
        print(f"% held-out test set {len(refs)}/{n_all} (θ_f fixed by the calibration set)")
        print("% ===== Table 1 (test set 623, grouped by corpus) =====")
        rows = []
        for key, f in errs.items():
            f = np.asarray(f, float)
            if len(f) != len(refs):
                continue
            g = corpus_group(meta[key]["corpus"])
            cov = matched_coverage(dual, meta[key]["corpus"], h=args.h, k=args.k)
            th = TH["theta_f"][key]
            lo, hi = cov < th, cov >= th
            rows.append((g, meta[key]["name"], th, 100 * lo.mean(),
                         float(np.nanmean(f[hi])) * 1000 if hi.sum() else np.nan,
                         float(np.nanmean(f[lo])) * 1000 if lo.sum() else np.nan))
        for gi, g in enumerate(GROUP_ORDER):
            sub = sorted([r for r in rows if r[0] == g], key=lambda r: r[4])
            for r in sub:
                print(f"{r[1]} & {r[2]:.2f} & {r[3]:.0f}\\% & {r[4]:.0f} & {r[5]:.0f} \\\\")
            if gi < len(GROUP_ORDER) - 1:
                print("\\hline")
        print("% within each group, sorted by MAE(ρ≥θ) ascending")

    if args.table in ("2", "all"):
        tpl, dual, errs, volt, Vd, keep = cathode_data()
        print("\n% ===== Table 2 (cathode 74, dual Spearman + MAE above/below θ) =====")
        rows = []
        for key, f in errs.items():
            f = np.asarray(f, float)[keep] * 1000
            cov = matched_coverage(dual, meta[key]["corpus"], h=args.h, k=args.k)[keep]
            V = model_voltage(volt[key], tpl)[keep]
            dv = np.abs(V - Vd[keep]) * 1000
            g = corpus_group(meta[key]["corpus"])
            thf = TH["theta_f"][key]
            thv = TH["theta_f"][key]      # shares the force θ_f
            held_v = np.ones(len(f), bool)          # voltage also uses all candidates (θ_f is unrelated to voltage)
            m = {}
            for tag, th, y, mask in [("f", thf, f, np.ones(len(f), bool)),
                                     ("v", thv, dv, held_v)]:
                lo, hi = (cov < th) & mask, (cov >= th) & mask
                m[tag] = (float(np.nanmean(y[hi])) if hi.sum() else np.nan,
                          float(np.nanmean(y[lo])) if lo.sum() else np.nan)
            rows.append((g, meta[key]["name"], spearmanr(cov, f).statistic,
                         spearmanr(cov, dv).statistic, m["f"][0], m["f"][1], m["v"][0], m["v"][1]))
        for gi, g in enumerate(GROUP_ORDER):
            sub = sorted([r for r in rows if r[0] == g], key=lambda r: r[4])
            for r in sub:
                print(f"{r[1]} & {r[2]:.2f} & {r[3]:.2f} & {r[4]:.0f} & {r[5]:.0f} & "
                      f"{r[6]:.0f} & {r[7]:.0f} \\\\")
            if gi < len(GROUP_ORDER) - 1:
                print("\\hline")
        print("% columns: Model & Sp_F & Sp_V & MAE_f(ρ≥θ) & MAE_f(ρ<θ) & MAE_V(ρ≥θ) & MAE_V(ρ<θ)")


if __name__ == "__main__":
    main()
