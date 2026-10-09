# -*- coding: utf-8 -*-
"""8-model version of main-text Fig 2 (cathode force error) and Fig 3 (voltage).

The same 8 models as Fig 1; every panel draws the θ line; **no bar summary**;
layered LiFeO2 is dropped (idx 8, a phase that does not exist experimentally).

  --quantity force   : y = per-atom force error vs the PBE single point (meV/Å), θ_f calibrated from ε_f=0.2 eV/Å
  --quantity voltage : y = |ΔV| vs the PBE average voltage (mV), θ_V calibrated from ε_V=0.2 V

Usage:
  PYTHONPATH=CBE:CBE/examples python fig_cathode8.py \
      --quantity force --out ../../fig_cathode8.pdf
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12, "axes.titlesize": 12.5, "axes.labelsize": 14,
                     "xtick.labelsize": 11, "ytick.labelsize": 11})

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

from coverage_matched import load_dual, matched_coverage, H_DEFAULT
from figs_common import CATHODE_DROP, GROUP_COLOR, MODELS8, TCOLORS, corpus_group, data, data_glob

ELEMS = ["Li", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Zr", "Nb",
         "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "Hf", "Ta", "W", "Re", "Os", "Ir",
         "Pt", "Au", "O", "P"]
N_LI = {"layered": 3, "spinel": 2, "olivine": 4}
LEGACY = {"MatterSim-v1": "mattersim-v1", "SevenNet-0": "sevennet-0", "ORB-v2": "orb-v2"}


def load_cathode():
    d = pickle.load(open(ROOT / "case_cathode/mlip_vs_dft.pkl", "rb"))
    vr = pickle.load(open(ROOT / "case_cathode/voltage_results.pkl", "rb"))
    tpl = np.asarray(d["templates"])
    cov_dual = load_dual(data("dist_cathode.npz"))
    errs, volt = {}, {}
    for f in sorted(data_glob("eval_roster_cathode_*.pkl")):
        errs.update(pickle.load(open(f, "rb")))
    for p in ["case_cathode/mlip_voltage_grace.pkl", "case_cathode/mlip_voltage_mlip7.pkl",
              "case_cathode/voltage_roster_grace.pkl", "case_cathode/voltage_roster_mlip7a.pkl",
              "case_cathode/voltage_roster_mlip7b.pkl"]:
        for k, v in pickle.load(open(ROOT / p, "rb")).items():   # p looks like case_cathode/xxx.pkl
            volt[LEGACY.get(k, k)] = v
    meta = pickle.load(open(data("roster_meta.pkl"), "rb"))
    keep = np.ones(len(tpl), bool)
    keep[CATHODE_DROP] = False
    Vd = np.asarray(vr["V_dft"]["DFT"], float)
    return tpl, cov_dual, errs, volt, meta, Vd, keep


def model_voltage(energies, tpl):
    from cbe.data import read_structures
    structs = read_structures(str(ROOT / "case_cathode/relaxed_all.xyz"))[:75]
    n_elem = np.array([int((ROOT / "voltage_vasp/elements" / f"{i:04d}/POSCAR")
                           .read_text().splitlines()[6].split()[0]) for i in range(28)], float)
    ee = np.asarray(energies["elements_e"], float) / n_elem
    ref = {s: ee[i] for i, s in enumerate(ELEMS)}
    n_li = np.array([N_LI[t] for t in tpl])
    return (np.asarray(energies["delith_e"], float) + n_li * ref["Li"]
            - np.asarray(energies["lith_e"], float)) / n_li


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantity", choices=["force", "voltage"], default="force")
    ap.add_argument("--out", default=None)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--h", type=float, default=None)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--eps-force", type=float, default=0.2)      # eV/Å
    ap.add_argument("--eps-volt", type=float, default=0.2)       # V
    args = ap.parse_args()
    out = args.out or str(ROOT / f"DOC/figures/fig_cathode8_{args.quantity}.pdf")

    tpl, cov_dual, errs, volt, meta, Vd, keep = load_cathode()
    TH = pickle.load(open(data("thetas.pkl"), "rb"))
    split = np.load(data("calib_split.npz"))
    # forces: θ_f comes from the test set's calibration split and the cathodes are held out as
    # a whole -> use all candidates
    # voltage: θ_V was calibrated on half of the cathodes -> use only that held-out half
    n = int(keep.sum())                   # voltage also uses all candidates: θ_f is calibrated on the training corpus, so there is no leakage
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 11.5), sharex=False, sharey=True)
    axes = axes.ravel()
    rows = []
    for i, (key, label, corpus) in enumerate(MODELS8):
        ax = axes[i]
        cov = matched_coverage(cov_dual, meta[key]["corpus"], h=args.h, k=args.k)[keep]
        if args.quantity == "force":
            y = np.asarray(errs[key], float)[keep] * 1000
            ylabel, unit = "force error vs PBE (meV/Å)", "meV/Å"
            eps = args.eps_force * 1000
            ymax = 600.0
        else:
            V = model_voltage(volt[key], tpl)
            y = np.abs(V - Vd)[keep] * 1000
            ylabel, unit = "|ΔV| vs PBE (mV)", "mV"
            eps = args.eps_volt * 1000
            ymax = 2500.0
        for t in TCOLORS:
            m = tpl[keep] == t
            if m.any():
                ax.scatter(cov[m], y[m], s=26, alpha=0.75, c=TCOLORS[t], edgecolors="none")
        th = TH["theta_f"][key]          # every quantity shares the same force-calibrated θ_f
        ax.axvline(th, color="0.25", ls="--", lw=1.5)
        lo, hi = cov < th, cov >= th
        mae_lo = float(np.nanmean(y[lo])) if lo.sum() else np.nan
        mae_hi = float(np.nanmean(y[hi])) if hi.sum() else np.nan
        rows.append((label, corpus_group(corpus), th, mae_lo, mae_hi))
        ax.set_ylim(0, ymax); ax.set_xlim(-0.02, 1.02); ax.grid(alpha=0.2)
        theta_sym = "$\\theta_f$"
        cshort = {"MPtrj + Alexandria + OMat24": "MPtrj+Alex+OMat24",
                  "MPtrj + Alexandria": "MPtrj+Alex", "MPtrj": "MPtrj"}[corpus_group(corpus)]
        ax.set_title(f"{label}\n{cshort}  |  {theta_sym}={th:.2f}", fontsize=13,
                     color=GROUP_COLOR[corpus_group(corpus)])
        if i % 3 == 0:
            ax.set_ylabel(ylabel)
        if i // 3 == 2:
            ax.set_xlabel("coverage ρ")

    axl = axes[8]; axl.axis("off")
    h1 = [plt.Line2D([], [], marker="o", ls="", color=TCOLORS[t], label=t) for t in TCOLORS]
    h1.append(plt.Line2D([], [], color="0.25", ls="--", label="threshold"))
    axl.legend(handles=h1, loc="center", frameon=False, fontsize=14, ncol=2,
               title="cathode framework", title_fontsize=12)

    q = "the delithiation voltage" if args.quantity == "voltage" else "the force error"
    fig.suptitle(f"Cathode screen: coverage against {q} (layered LiFeO$_2$ excluded); "
                 f"thresholds fixed by calibration", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(Path(out).with_suffix(".png")), dpi=130, bbox_inches="tight")
    print(f"saved -> {out}")
    unit = "meV/Å" if args.quantity == "force" else "mV"
    print(f"\n{'model':22s} {'corpus':30s} {'theta':>6s} {'MAE(<θ)':>9s} {'MAE(≥θ)':>9s}")
    for nm, cp, th, lo_, hi_ in rows:
        print(f"{nm:22s} {cp:30s} {th:6.2f} {lo_:9.0f} {hi_:9.0f}   ({unit})")


if __name__ == "__main__":
    main()
