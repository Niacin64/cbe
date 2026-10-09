# -*- coding: utf-8 -*-
"""Case study 2: parse cathode DFT (vasprun.xml under project/) + compare MLIPs, plot coverage vs error.

Alignment: project/{i:04d} == relaxed_all.xyz[i] == mlip_energy.pkl.meta[i] (same order).
Errors: force error = mean||F_MLIP - F_DFT||; energy error = |E_MLIP - E_DFT|/N.

Outputs: case_cathode/mlip_vs_dft.pkl + case_cathode/cathode_case_study.png (2x2, 4 models)
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# for paper layout: fonts stay readable after the figure is scaled to \textwidth (~6.8 in)
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 15, "axes.labelsize": 15,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 8.5,
    "figure.titlesize": 15, "axes.linewidth": 0.8,
})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cbe.vasp import collect_vasp_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vasp-dir", required=True, help="project/ (75 VASP result directories)")
    ap.add_argument("--mlip", required=True, help="mlip_energy.pkl")
    ap.add_argument("--out-pkl", default="case_cathode/mlip_vs_dft.pkl")
    ap.add_argument("--out-png", default="case_cathode/cathode_case_study.png")
    args = ap.parse_args()

    # ---- parse DFT ----
    dirs = sorted([p for p in Path(args.vasp_dir).iterdir() if p.is_dir()])
    dft_structs, dft_energies, dft_forces = collect_vasp_results(dirs)
    n = len(dft_structs)
    print(f"DFT parsed {n} structures")

    # ---- load MLIP ----
    with open(args.mlip, "rb") as fh:
        mlip = pickle.load(fh)
    meta, results = mlip["meta"], mlip["results"]
    coverage = np.asarray(meta["coverage"])
    templates = np.array(meta["template"])
    assert n == len(coverage), "DFT and MLIP counts do not match"

    # ---- compute errors ----
    errs = {}
    for model, r in results.items():
        fe, ee = [], []
        for i in range(n):
            N = len(dft_structs[i])
            f_mlip = np.asarray(r["forces"][i])
            f_dft = np.asarray(dft_forces[i])
            fe.append(float(np.mean(np.linalg.norm(f_mlip - f_dft, axis=1))))
            ee.append(abs(r["energy"][i] / N - dft_energies[i] / N))
        errs[model] = {"f_err": np.array(fe), "e_err": np.array(ee)}

    with open(args.out_pkl, "wb") as fh:
        pickle.dump({"coverage": coverage, "templates": templates,
                     "tm": meta["tm"], "composition": meta["composition"],
                     "errs": errs}, fh)
    print(f"errors saved -> {args.out_pkl}")

    # ---- summary printout ----
    from scipy.stats import spearmanr
    print(f"\n{'model':14s} {'Sp(cov,Ferr)':>12s} {'Ferr@high_cov':>13s} {'Ferr@low_cov':>12s}")
    for model, r in errs.items():
        sp = spearmanr(coverage, r["f_err"])[0]
        hi = r["f_err"][coverage > 0.5]
        lo = r["f_err"][coverage < 0.05]
        print(f"{model:14s} {sp:12.3f} {np.median(hi)*1000 if len(hi) else np.nan:13.0f} "
              f"{np.median(lo)*1000 if len(lo) else np.nan:12.0f}")

    # ---- plot: 2x2 grid, coverage vs force error, colored by template + per-model conformal θ_f ----
    from cbe.calibration import ConformalCalibrator
    tcolors = {"layered": "#1a4d8f", "spinel": "#c0392b", "olivine": "#2e7d32"}
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 8.2))
    axes = axes.ravel()
    # compare MLIPs on equal footing: all panels share the same y-axis range
    f_ymax = max(np.max(r["f_err"]) for r in errs.values()) * 1.05
    print(f"\nshared y-limit for the force-error plots: {f_ymax * 1000:.0f} meV/A")
    print(f"\nper-model conformal threshold theta_f (eps_f=0.5 eV/A):")
    summary = []  # (model, theta, low_cov_err)
    for ax, (model, r) in zip(axes[:len(errs)], errs.items()):
        for t in ["layered", "spinel", "olivine"]:
            m = templates == t
            ax.scatter(coverage[m], r["f_err"][m], s=18, alpha=0.6,
                       c=tcolors[t], edgecolors="none", label=t)
        sp = spearmanr(coverage, r["f_err"])[0]
        # per-model conformal threshold (calibrated on the 75 cathode candidates)
        theta = ConformalCalibrator(alpha=0.1, seed=0).fit(
            coverage, r["f_err"]).threshold(0.5)
        ax.axvline(theta, color="k", lw=1.5, ls="--", label=f"θ_f={theta:.3f}")
        ax.set_xlabel("coverage ρ (1=covered)")
        ax.set_ylabel("force error (eV/Å)")
        ax.set_title(f"{model}\n(Sp={sp:.2f}, $\\theta_f$={theta:.3f})", fontsize=14)
        ax.set_ylim(0, f_ymax)
        ax.grid(alpha=0.2, axis="y")
        print(f"  {model:14s} θ_f={theta:.3f}")
        summary.append((model, theta,
                        float(np.median(r["f_err"][coverage < 0.05])) * 1000))

    # panel 8: extrapolation penalty summary (median force error in the low-coverage region, ascending)
    ax = axes[len(errs)]
    names = [s[0] for s in summary]
    low_err = np.array([s[2] for s in summary])
    idx = np.argsort(low_err)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(summary)))
    ax.barh(range(len(summary)), low_err[idx], color=colors)
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels([names[i] for i in idx], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("F_err, $\\rho$<0.05 (meV/Å)", fontsize=13)
    ax.set_title("extrapolation penalty", fontsize=15)
    for i, j in enumerate(idx):
        ax.text(low_err[j] + 4, i, f"{low_err[j]:.0f}", va="center", fontsize=11)
    # panel 9: legend + how to read the figure
    ax = axes[len(errs) + 1]
    ax.axis("off")
    proxies = [Line2D([], [], marker="o", ls="none", ms=9, c=tcolors[k], label=k)
               for k in ["layered", "spinel", "olivine"]]
    proxies.append(Line2D([], [], color="k", lw=1.5, ls="--",
                          label="θ_f threshold"))
    ax.legend(handles=proxies, loc="center", fontsize=13, frameon=False, ncol=2,
              columnspacing=1.4, handletextpad=0.5, labelspacing=0.9, borderpad=0.2)

    fig.suptitle("coverage vs MLIP-vs-DFT force error (Li cathode candidates)", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(args.out_png, dpi=150)
    fig.savefig(str(Path(args.out_png).with_suffix('.pdf')))
    print(f"figure saved -> {args.out_png}")


if __name__ == "__main__":
    main()
