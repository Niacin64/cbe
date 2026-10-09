# -*- coding: utf-8 -*-
"""M4 baseline analysis: coverage vs multi-model disagreement (the "Why this score" section of
the main text).

Requires running ensemble_baseline.py first to produce DATA/ensemble_test.npz (per-atom forces
of the 3 MACE-MP-0 members on the 312 held-out structures). This script reports:
  - the AUC of the two scores (in-dist vs perturbed);
  - precision / recall at a matched 65% flagged rate (for each member's own force error, with
    >0.5 eV/Å as the positive class);
  - the overlap of the two flagged sets (Jaccard);
  - the "consistency trap" on layered LiFeO₂: how much the three members differ in voltage and
    how far they are from PBE.
Usage:
  PYTHONPATH=CBE:CBE/examples python \
      ensemble_baseline_analysis.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
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
from cbe.data import read_structures
from coverage_matched import load_dual, slice_dual, matched_coverage
from figs_common import CATHODE_DROP, data, data_glob
from fig_cathode8 import load_cathode, model_voltage

MEMBERS = ["mace-mp-0-small", "mace-mp-0-medium", "mace-mp-0-large"]


def main():
    keep = np.load(data("test_keep.npy"))
    held = np.load(data("calib_split.npz"))["held_out"]
    refs = read_structures(str(data("refs_full.xyz"))) + read_structures(str(data("refs_ood.xyz")))
    refs = [a for a, k in zip(refs, keep) if k]
    grp = np.array([str(a.info.get("group", "?")) for a in refs])
    grp = np.where(np.char.startswith(grp, "rattle"), "perturbed", grp)[held]
    cov = matched_coverage(slice_dual(load_dual(data("dist_test.npz"), keep), held), "MPtrj")
    errs = {}
    for f in sorted(data_glob("eval_roster_test_*.pkl")):
        errs.update(pickle.load(open(f, "rb")))
    f = {k: np.asarray(errs[k], float)[held] * 1000 for k in MEMBERS}
    U = np.asarray(np.load(data("ensemble_test.npz"), allow_pickle=True)["U_dis"], float) * 1000

    pos = grp == "in-dist"
    print("AUC (in-dist vs perturbed):  coverage %.3f | disagreement %.3f"
          % (roc_auc_score(pos, cov), roc_auc_score(pos, -U)))
    print("Spearman(disagreement, coverage) = %.3f" % spearmanr(U, cov).statistic)
    print("\nmatched 65% flagged rate (positive class = that member's e_F > 0.5 eV/Å):")
    flags = {"coverage": -cov > np.quantile(-cov, 0.65), "disagreement": U > np.quantile(U, 0.65)}
    for name, fl in flags.items():
        pr, rc = [], []
        for k in MEMBERS:
            y = f[k] > 500
            tp = (fl & y).sum()
            pr.append(tp / max(fl.sum(), 1))
            rc.append(tp / max(y.sum(), 1))
        print("  %-13s precision %.2f/%.2f/%.2f  recall %.2f/%.2f/%.2f"
              % (name, *pr, *rc))
    inter = (flags["coverage"] & flags["disagreement"]).sum()
    union = (flags["coverage"] | flags["disagreement"]).sum()
    print("\nflagged-set overlap: coverage %d, disagreement %d, intersection %d, Jaccard %.2f"
          % (flags["coverage"].sum(), flags["disagreement"].sum(), inter, inter / union))

    # ---- layered LiFeO2: the consistency trap ----
    tpl, _, _, volt, _, _, _ = load_cathode()
    i = int(np.ravel(CATHODE_DROP)[0])
    V_dft = float(np.ravel(np.asarray(pickle.load(
        open("case_cathode/voltage_results.pkl", "rb"))["V_dft"]["DFT"], float))[i])
    Vs = np.array([float(np.ravel(model_voltage(volt[k], tpl))[i]) for k in MEMBERS])
    allV = np.stack([np.ravel(model_voltage(volt[k], tpl)) for k in errs], 0)
    print("\nlayered LiFeO2 (candidate %d): PBE %.2f V" % (i, V_dft))
    print("  three members %.2f / %.2f / %.2f V, cross-member std %.0f mV, mean deviation %+.2f V"
          % (*Vs, Vs.std() * 1000, Vs.mean() - V_dft))
    print("  voltage std across the 18 models: this candidate %.0f mV, median of the other 74 %.0f mV"
          % (allV.std(0)[i] * 1000, np.median(np.delete(allV.std(0), i)) * 1000))


if __name__ == "__main__":
    main()
