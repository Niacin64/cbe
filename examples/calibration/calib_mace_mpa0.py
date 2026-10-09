# -*- coding: utf-8 -*-
"""Run one full calibration for a single model (default MACE-MPA-0, corpus MPtrj + sAlex).

Three calibration-set definitions (--calib-mode):
  corpus      training-corpus samples only (300 MPtrj + 240 Alexandria, with PBE forces)
  corpus+ood  corpus samples + half of the hard structures (stratified by group)  <- recommended
  ood         half of the hard structures only

Why hard structures are needed: the conformal threshold θ = min{c : q̂σ(c) ≤ ε_f} requires the set to contain
structures **beyond tolerance**. Corpus samples alone leave the model within tolerance, the band always
passes, θ collapses to 0 and no alarm ever fires - the conformal exchangeability requirement.

Usage:
  PYTHONPATH=CBE:CBE/examples python calib_mace_mpa0.py \
      --model mace-mpa-0 --calib-mode corpus+ood --out ../../case_calib_mace_mpa0.pdf
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 12.5,
                     "xtick.labelsize": 10.5, "ytick.labelsize": 10.5})

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
from coverage_matched import per_index_coverage, indexes_for

H, K = 0.2184, 30


def _dual(npz_mp, npz_alex, keep=None):
    mp = np.load(data(npz_mp))
    al = np.load(data(npz_alex)) if data(npz_alex).exists() else mp
    if keep is None:
        return {"bnd": mp["bnd"], "D_mp": mp["D_mp"], "D_alex": al["D_alex"]}
    cnt = np.diff(mp["bnd"])
    return {"bnd": np.concatenate([[0], np.cumsum(cnt[keep])]),
            "D_mp": mp["D_mp"][np.repeat(keep, cnt)],
            "D_alex": al["D_alex"][np.repeat(keep, cnt)]}


def _cov(dual, corpus, h, k):
    return np.max(np.stack([per_index_coverage(dual, t, h=h, k=k)
                            for t in indexes_for(corpus)], 0), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mace-mpa-0")
    ap.add_argument("--calib-mode", choices=["corpus", "corpus+ood", "ood"], default="corpus+ood")
    ap.add_argument("--eps-f", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--h", type=float, default=H)
    ap.add_argument("--out", default=str(ROOT / "DOC/figures/case_calib_mace_mpa0.pdf"))
    args = ap.parse_args()

    from ase.io import read as ase_read
    meta = pickle.load(open(data("roster_meta.pkl"), "rb"))
    name, corpus = meta[args.model]["name"], meta[args.model]["corpus"]
    print(f"model {name} (corpus {corpus}); calibration mode {args.calib_mode}\n")

    # ---- 1) training-corpus samples ----
    calib = ase_read(str(data("refs_calib.xyz")), index=":")
    cg = np.array([a.info.get("group", "?") for a in calib])
    cov_cal = _cov(_dual("dist_calib_mp.npz", "dist_calib_alex.npz"), corpus, args.h, K)
    f_cal = np.asarray(pickle.load(open(data("eval_roster_calib_grace.pkl"), "rb"))[args.model],
                       float)
    print(f"{len(calib)} corpus samples: median coverage {np.median(cov_cal):.3f}"
          f" (MPtrj {np.median(cov_cal[cg == 'mptrj-calib']):.3f} / "
          f"Alexandria {np.median(cov_cal[cg == 'alexandria-calib']):.3f}); "
          f"median force error {np.median(f_cal) * 1000:.0f} meV/A, p90 {np.percentile(f_cal, 90) * 1000:.0f}")

    # ---- 2) hard structures (physically filtered test set, stratified halves) ----
    refs = (ase_read(str(data("refs_full.xyz")), index=":")
            + ase_read(str(data("refs_ood.xyz")), index=":"))
    keep = np.load(data("test_keep.npy"))
    refs = [a for a, k in zip(refs, keep) if k]
    g = np.array([str(a.info.get("group", "?")) for a in refs])
    g = np.where(np.char.startswith(g, "rattle"), "perturbed", g)
    cov_t = _cov(_dual("dist_test.npz", "dist_test.npz", keep), corpus, args.h, K)
    f_t = np.asarray(pickle.load(open(data("eval_roster_test_grace.pkl"), "rb"))[args.model], float)
    rng = np.random.default_rng(0)
    half = np.zeros(len(refs), bool)
    for gg in np.unique(g):
        ii = rng.permutation(np.where(g == gg)[0])
        half[ii[:len(ii) // 2]] = True
    print(f"{len(refs)} hard structures: median coverage {np.median(cov_t):.3f}; "
          f"median force error {np.median(f_t) * 1000:.0f} meV/A"
          f" (halves: calibration {int(half.sum())} / evaluation {int((~half).sum())})")

    # ---- 3) assemble the calibration set ----
    if args.calib_mode == "corpus":
        cov_c, f_c, ev = cov_cal, f_cal, np.ones(len(refs), bool)
    elif args.calib_mode == "ood":
        cov_c, f_c, ev = cov_t[half], f_t[half], ~half
    else:
        cov_c, f_c, ev = np.r_[cov_cal, cov_t[half]], np.r_[f_cal, f_t[half]], ~half
    cov_ev, f_ev, g_ev = cov_t[ev], f_t[ev], g[ev]

    # ---- 4) conformal calibration ----
    cal = ConformalCalibrator(alpha=args.alpha, seed=0).fit(cov_c, f_c)
    theta = cal.threshold(args.eps_f)
    theta_far = float(np.quantile(cov_cal, args.alpha))
    print(f"\ncalibration set: {len(cov_c)} structures -> theta_f = {theta:.3f}"
          + ("   <- all good structures, band always passes, no alarm fires" if theta <= 1e-9 else ""))
    print(f"(reference: alpha-quantile of training-distribution coverage = {theta_far:.3f}, "
          f"i.e. '{100 * args.alpha:.0f}% false alarms on the training set'; model-independent, not primary)")

    # ---- 5) evaluation ----
    lo, hi = cov_ev < theta, cov_ev >= theta
    print(f"\nevaluation set: {int(ev.sum())} structures (disjoint from the calibration set)")
    if lo.sum():
        print(f"  flagged rho<theta: {int(lo.sum())} ({100 * lo.mean():.0f}%)")
        print(f"  MAE  ρ>=θ {np.mean(f_ev[hi]) * 1000:7.1f} | ρ<θ "
              f"{np.mean(f_ev[lo]) * 1000:7.1f} meV/A | ratio "
              f"{np.mean(f_ev[lo]) / np.mean(f_ev[hi]):.1f}x")
        for gg in ["in-dist", "perturbed", "random"]:
            m = g_ev == gg
            if m.any():
                print(f"    {gg:10s} n={m.sum():3d}  rho median {np.median(cov_ev[m]):.3f}  "
                      f"median force error {np.median(f_ev[m]) * 1000:7.1f} meV/A  "
                      f"flagged {100 * np.mean(cov_ev[m] < theta):3.0f}%")
    else:
        print("  no structure was flagged (theta=0)")
    print(f"  fraction within tolerance {100 * np.mean(f_ev <= args.eps_f):.1f}%")

    # ---- 6) figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    ax = axes[0]
    ax.scatter(cov_cal[cg == "mptrj-calib"], f_cal[cg == "mptrj-calib"] * 1000, s=15, alpha=0.7,
               c="#1a4d8f", label=f"MPtrj sample ({int((cg == 'mptrj-calib').sum())})")
    ax.scatter(cov_cal[cg == "alexandria-calib"], f_cal[cg == "alexandria-calib"] * 1000, s=15,
               alpha=0.7, c="#c0392b",
               label=f"Alexandria sample ({int((cg == 'alexandria-calib').sum())})")
    if args.calib_mode != "corpus":
        ax.scatter(cov_t[half], f_t[half] * 1000, s=12, alpha=0.45, c="#2e7d32",
                   label=f"held-out hard structures ({int(half.sum())})")
    ax.axhline(args.eps_f * 1000, color="0.3", ls=":", lw=1.4,
               label=f"$\\varepsilon_f$={args.eps_f} eV/Å")
    if theta > 0:
        ax.axvline(theta, color="0.25", ls="--", lw=1.6, label=f"$\\theta_f$={theta:.2f}")
    ax.set_yscale("log"); ax.set_xlabel("coverage ρ"); ax.set_ylabel("force error (meV/Å)")
    ax.set_title(f"A  calibration set ({args.calib_mode})\n{name}")
    ax.legend(fontsize=8.5, frameon=False); ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    order = np.argsort(cov_c); cs = np.sort(cov_c)
    ax.plot(cs, f_c[order] * 1000, ".", ms=4, color="#1a4d8f", alpha=0.5,
            label="calibration errors")
    ax.plot(cs, np.array([cal.q_ * cal.sigma_(c) for c in cs]) * 1000, "-", color="#c0392b",
            lw=2, label="conformal band $\\hat q\\,\\sigma(c)$")
    ax.axhline(args.eps_f * 1000, color="0.3", ls=":", lw=1.4, label="$\\varepsilon_f$")
    if theta > 0:
        ax.axvline(theta, color="0.25", ls="--", lw=1.6)
    ax.set_yscale("log"); ax.set_xlabel("coverage ρ"); ax.set_ylabel("force error (meV/Å)")
    ax.set_title("B  conformal band vs tolerance"); ax.legend(fontsize=8.5, frameon=False)
    ax.grid(alpha=0.25, which="both")

    ax = axes[2]
    for gg, cc in [("in-dist", "#1a4d8f"), ("perturbed", "#e08a00"), ("random", "#2e7d32")]:
        m = g_ev == gg
        if m.any():
            ax.scatter(cov_ev[m], f_ev[m] * 1000, s=15, alpha=0.65, c=cc, label=gg)
    if theta > 0:
        ax.axvline(theta, color="0.25", ls="--", lw=1.6)
    ax.axhline(args.eps_f * 1000, color="0.3", ls=":", lw=1.2)
    ax.set_yscale("log"); ax.set_xlabel("coverage ρ"); ax.set_ylabel("force error (meV/Å)")
    ax.set_title((f"C  held-out evaluation\nMAE {np.mean(f_ev[hi]) * 1000:.0f} vs "
                  f"{np.mean(f_ev[lo]) * 1000:.0f} meV/Å") if lo.sum()
                 else "C  held-out evaluation\n(no structure flagged)")
    ax.legend(fontsize=8.5, frameon=False); ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(str(Path(args.out).with_suffix(".png")), dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {args.out} (+ .png)")


if __name__ == "__main__":
    main()
