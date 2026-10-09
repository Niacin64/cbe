# -*- coding: utf-8 -*-
"""Main-text Fig 1 (8-model version): coverage vs force-error scatter + an "MAE of ρ<θ" summary.

Layout 3x3 + a full-width bottom row:
  rows 1-3: the 8 representative models + a legend cell
  row 4   : extrapolation penalty = **the MAE over the structures whose coverage falls below
            that model's θ_f** (lower is better), together with the "fraction within tolerance"
            (higher is better) as a reference number.

Usage:
  PYTHONPATH=CBE:CBE/examples python fig_test8.py --out ../../fig_test8.pdf
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 15.5,
                     "xtick.labelsize": 12.5, "ytick.labelsize": 12.5})

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

from coverage_matched import load_dual, slice_dual, matched_coverage, H_DEFAULT
from figs_common import GCOLORS, GROUP_COLOR, MODELS8, corpus_group, data, data_glob


def load():
    from cbe.data import read_structures
    refs = read_structures(str(data("refs_full.xyz"))) + read_structures(str(data("refs_ood.xyz")))
    keep = np.load(data("test_keep.npy"))
    refs = [a for a, k in zip(refs, keep) if k]
    groups = np.array([str(a.info.get("group", "?")) for a in refs])
    groups = np.where(np.char.startswith(groups, "rattle"), "perturbed", groups)
    errs = {}
    for f in sorted(data_glob("eval_roster_test_*.pkl")):
        errs.update(pickle.load(open(f, "rb")))
    meta = pickle.load(open(data("roster_meta.pkl"), "rb"))
    th = pickle.load(open(data("thetas.pkl"), "rb"))
    held = np.load(data("calib_split.npz"))["held_out"]        # plot the held-out half only
    dual = slice_dual(load_dual(data("dist_test.npz"), keep), held)
    refs = [a for a, k in zip(refs, held) if k]
    groups = groups[held]
    errs = {k: np.asarray(v, float)[held] for k, v in errs.items()}
    return refs, groups, dual, errs, meta, th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "DOC/figures/fig_test8.pdf"))
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--h", type=float, default=None)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--f-ymax", type=float, default=3.0)
    args = ap.parse_args()

    refs, groups, dual, errs, meta, TH = load()
    n_struct = len(refs)
    present = [g for g in GCOLORS if g in set(groups)]
    print(f"held-out test set: {n_struct} structures, {len(errs)} models (θ fixed by the calibration set)")

    # Full-width layout (JCTC two-column \textwidth = 17.8 cm): 2x4 model panels + a bottom
    # penalty plot; panels are ~4.2 x 3.4 cm, so 14 pt reads as ~7.7 pt at print size.
    fig = plt.figure(figsize=(17.0, 14.4))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.0, 1.0, 1.05], hspace=0.50, wspace=0.26)

    table = []
    for i, (key, label, corpus) in enumerate(MODELS8):
        r, c = divmod(i, 4)
        ax = fig.add_subplot(gs[r, c])
        f = np.asarray(errs[key], float)
        cov = matched_coverage(dual, corpus_of := meta[key]["corpus"], h=args.h, k=args.k)
        for g in present:
            m = groups == g
            ax.scatter(cov[m], f[m], s=9, alpha=0.5, c=GCOLORS[g], edgecolors="none")
        th = TH["theta_f"][key]        # fixed: calibrated once on the calibration set
        ax.axvline(th, color="0.25", ls="--", lw=1.5)
        lo, hi = cov < th, cov >= th
        mae_lo = float(np.nanmean(f[lo])) * 1000 if lo.sum() else np.nan
        mae_hi = float(np.nanmean(f[hi])) * 1000 if hi.sum() else np.nan
        within = float(np.mean(f <= args.eps)) * 100
        table.append((label, corpus_group(corpus), th, mae_lo, mae_hi, within))
        cshort = {"MPtrj + Alexandria + OMat24": "MPtrj+Alex+OMat24",
                  "MPtrj + Alexandria": "MPtrj+Alex", "MPtrj": "MPtrj"}[corpus_group(corpus)]
        ax.set_title(f"{label}\n{cshort}  |  $\\theta_f$={th:.2f}", fontsize=14,
                     color=GROUP_COLOR[corpus_group(corpus)])
        ax.set_xlim(0, 1.02); ax.set_ylim(0, args.f_ymax); ax.grid(alpha=0.2)
        if c == 0:
            ax.set_ylabel("force error (eV/Å)")
        if r == 1:
            ax.set_xlabel("coverage ρ")

    axl = fig.add_subplot(gs[2, 0]); axl.axis("off")
    h = [plt.Line2D([], [], marker="o", ls="", color=GCOLORS[g], label=g) for g in present]
    h.append(plt.Line2D([], [], color="0.25", ls="--", label="$\\theta_f$ threshold"))
    axl.legend(handles=h, loc="center", frameon=False, fontsize=13.5)

    # ---- penalty: MAE over ρ<θ (all 18 models) ----
    axp = fig.add_subplot(gs[2, 0:])
    rows = []
    for k, f in errs.items():
        f = np.asarray(f, float)
        if len(f) != n_struct:
            continue
        corpus = meta[k]["corpus"]
        cov = matched_coverage(dual, corpus, h=args.h, k=args.k)
        th = TH["theta_f"][k]          # fixed: calibrated once on the calibration set (note k, not a leftover key)
        lo = cov < th
        rows.append((meta[k]["name"], corpus_group(corpus),
                     float(np.nanmean(f[lo])) * 1000 if lo.sum() else np.nan,
                     float(np.mean(f <= args.eps)) * 100))
    rows.sort(key=lambda r: r[2])
    y = np.arange(len(rows))[::-1]
    axp.barh(y, [r[2] for r in rows], height=0.68,
             color=[GROUP_COLOR[r[1]] for r in rows])
    for yy, (nm, cp, v, w) in zip(y, rows):
        axp.text(v * 1.02, yy, f"{v:.0f}", va="center", fontsize=11)
    axp.set_yticks(y)
    short = {"MPtrj + Alexandria + OMat24": "MPtrj+Al+OMat24",
             "MPtrj + Alexandria": "MPtrj+Alex",
             "MPtrj": "MPtrj"}
    axp.set_yticklabels([f"{nm}  |  {short[cp]}" for nm, cp, _, _ in rows], fontsize=10)
    axp.set_xlabel("MAE on the configurations flagged by the guard, $\\rho<\\theta_f$ (meV/Å)")
    axp.set_title(f"extrapolation penalty: all {len(rows)} models "
                  f"(lower is better; numbers are this MAE)", fontsize=14)
    axp.set_xscale("log")
    axp.grid(alpha=0.2, axis="x", which="both")
    from matplotlib.patches import Patch
    axp.legend(handles=[Patch(color=GROUP_COLOR[g], label=g) for g in
                        ["MPtrj", "MPtrj + Alexandria", "MPtrj + Alexandria + OMat24"]],
               loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
               frameon=False, fontsize=11.5)

    fig.suptitle("Coverage against per-atom force error on the held-out half of the test set;\n"
                 "coverage from each model's own training corpus, $\\theta_f$ fixed by the calibration set",
                 fontsize=15, y=0.995)
    fig.subplots_adjust(left=0.165, right=0.985, top=0.905, bottom=0.09)
    fig.savefig(args.out)
    fig.savefig(str(Path(args.out).with_suffix(".png")), dpi=130, bbox_inches="tight")
    print(f"saved -> {args.out}")
    print(f"\n{'model':22s} {'corpus':30s} {'theta':>6s} {'MAE(<θ)':>9s} {'MAE(≥θ)':>9s} {'in-tol%':>8s}")
    for nm, cp, th, lo_, hi_, w in table:
        print(f"{nm:22s} {cp:30s} {th:6.2f} {lo_:9.0f} {hi_:9.0f} {w:8.1f}")
    print(f"\npenalty (MAE over ρ<θ, ascending):")
    for nm, cp, v, w in rows:
        print(f"  {nm:22s} {cp:30s} {v:8.0f} meV/Å   within tol {w:5.1f}%")


if __name__ == "__main__":
    main()
