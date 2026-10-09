# -*- coding: utf-8 -*-
"""Baseline ablation: show that each of the two design choices earns its keep -- (i) kernel
density vs raw distance; (ii) per-model calibration vs a single global threshold.
Two trivial geometric baselines are added as negative controls.

Four "scores" (smaller = more suspicious, all oriented the same way):
  A  KDE coverage ρ           <- this work's method (k=30, h=0.2184, per-structure min)
  B  raw distance -d30        <- kernel density removed; the 30th nearest-neighbour distance
                                 used directly (negated to align the direction)
  C  volume per atom (negated)          <- trivial negative control (geometry, no chemical information)
  D  mean nearest-neighbour distance (negated)  <- trivial negative control (purely geometric)

Three calibration/decision schemes:
  (1) per-model conformal θ_f                 <- what this work does
  (2) single global threshold = median of all models' θ_f  <- measures what "per-model
      calibration" is worth
  (3) per-model conformal, but with score B/C/D            <- measures what "kernel density"
      is worth

Metrics: AUC (separating in-dist vs perturbed+random, threshold-free) +
      penalty (the MAE of the flagged structures) and the within-tolerance fraction
      (operational metrics).

Usage:
  PYTHONPATH=CBE:CBE/examples python baselines_ablation.py \
      --out ../../DOC/figures/fig_baselines.pdf
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from coverage_matched import load_dual, slice_dual, per_index_coverage, indexes_for
from figs_common import data, data_glob, MODELS8, GROUP_COLOR, corpus_group

H, K, EPS, ALPHA = 0.2184, 30, 0.5, 0.1
plt.rcParams.update({"font.size": 11.5, "axes.titlesize": 12.5, "axes.labelsize": 12.5})


def auc(pos, neg):
    """P(score is higher in the positive class) (the normalised Mann-Whitney U)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = np.argsort(np.argsort(np.r_[pos, neg])) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def geometry_scores(structs):
    """The two trivial baselines: volume per atom and mean nearest-neighbour distance (neither
    carries chemical information)."""
    from ase.neighborlist import neighbor_list
    vol, nnd = [], []
    for a in structs:
        vol.append(a.get_volume() / len(a))
        d = neighbor_list("d", a, 3.5)
        # no neighbour within 3.5 Å (very sparse/isolated) -> treat as the most anomalous:
        # fill in a value below every finite one
        nnd.append(float(np.mean(d)) if len(d) else np.nan)
    nnd = np.asarray(nnd, float)
    if not np.isfinite(nnd).all():
        fin = nnd[np.isfinite(nnd)]
        nnd = np.where(np.isfinite(nnd), nnd,
                       (fin.max() + 1.0) if len(fin) else 1.0)
    return -np.asarray(vol), -nnd                  # negate to align the direction: smaller = more suspicious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "DOC/figures/fig_ablation_main.pdf"))
    args = ap.parse_args()

    from ase.io import read
    meta = pickle.load(open(data("roster_meta.pkl"), "rb"))
    th = pickle.load(open(data("thetas.pkl"), "rb"))["theta_f"]
    held, keep = np.load(data("calib_split.npz"))["held_out"], np.load(data("calib_split.npz"))["keep"]

    refs = read(data("refs_full.xyz"), index=":") + read(data("refs_ood.xyz"), index=":")
    refs = [a for a, k in zip(refs, keep) if k]
    grp = np.array([str(a.info.get("group", "?")) for a in refs])
    grp = np.where(np.char.startswith(grp, "rattle"), "perturbed", grp)
    pos, neg = grp == "in-dist", grp != "in-dist"
    eval_refs = [a for a, h in zip(refs, held) if h]
    ev_pos, ev_neg = pos[held], neg[held]
    pert_pos, pert_neg = pos[held], (grp == "perturbed")[held]     # in-dist vs mild perturbation (screening-relevant)
    rand_pos, rand_neg = pos[held], (grp == "random")[held]         # in-dist vs random cells (trivial)
    pert_mask = (grp != "in-dist")[held] & (grp != "random")[held]  # perturbed group only

    # ---- scores: A coverage / B raw distance (evaluation half + calibration half) ----
    dual_ev = slice_dual(load_dual(data("dist_test.npz"), keep), held)
    dual_cal = slice_dual(load_dual(data("dist_test.npz"), keep), ~held)
    errs = {}
    for f in data_glob("eval_roster_test_*.pkl"):
        errs.update(pickle.load(open(f, "rb")))

    def score_A(dual, corpus, tag="kde"):
        if tag == "kde":
            return np.max(np.stack([per_index_coverage(dual, t, h=H, k=K)
                                    for t in indexes_for(corpus)], 0), axis=0)
        # raw distance: min over atoms of the K-th NN distance, negated
        arrs = [dual[f"D_{t}"][:, min(K - 1, dual[f"D_{t}"].shape[1] - 1)]
                for t in indexes_for(corpus)]
        d = np.min(np.stack(arrs, 0), axis=0)
        cnt = np.diff(dual["bnd"])
        return -np.array([d[a:b].min() for a, b in zip(dual["bnd"][:-1], dual["bnd"][1:])])

    keys = [k for k in errs if k in meta and k in th]
    A_ev = {k: score_A(dual_ev, meta[k]["corpus"], "kde") for k in keys}
    A_ca = {k: score_A(dual_cal, meta[k]["corpus"], "kde") for k in keys}
    B_ev = {k: score_A(dual_ev, meta[k]["corpus"], "dist") for k in keys}
    B_ca = {k: score_A(dual_cal, meta[k]["corpus"], "dist") for k in keys}
    err_ev = {k: np.asarray(errs[k], float)[held] for k in keys}
    err_ca = {k: np.asarray(errs[k], float)[~held] for k in keys}

    # ---- trivial baselines ----
    g_ev = geometry_scores(eval_refs)
    g_ca = geometry_scores([a for a, h in zip(refs, ~held) if h])
    C_ev, D_ev = g_ev[0], g_ev[1]
    C_ca, D_ca = g_ca[0], g_ca[1]

    # ---- per-model conformal (scores A / B) ----
    # calibration set = corpus samples (540) + the difficult half, matching the paper
    dual_corpus = load_dual(data("dist_calib.npz"))
    errs_cal = {}
    for f in data_glob("eval_roster_calib_*.pkl"):
        errs_cal.update(pickle.load(open(f, "rb")))
    A_co = {k: score_A(dual_corpus, meta[k]["corpus"], "kde") for k in keys if k in errs_cal}
    B_co = {k: score_A(dual_corpus, meta[k]["corpus"], "dist") for k in keys if k in errs_cal}
    cal_refs = read(data("refs_calib.xyz"), index=":")
    g_co = geometry_scores(cal_refs)
    keys = [k for k in keys if k in errs_cal]

    def _fit(cov, err):
        return float(ConformalCalibrator(alpha=ALPHA, seed=0).fit(cov, err).threshold(EPS))

    thA, thB, thC, thD = {}, {}, {}, {}
    for k in keys:
        ec = np.asarray(errs_cal[k], float)
        thA[k] = _fit(np.r_[A_co[k], A_ca[k]], np.r_[ec, err_ca[k]])
        thB[k] = _fit(np.r_[B_co[k], B_ca[k]], np.r_[ec, err_ca[k]])
        thC[k] = _fit(g_co[0], ec)          # the geometric baseline also gets its own per-model θ (best chance)
        thD[k] = _fit(g_co[1], ec)
    th_global_A = float(np.median([thA[k] for k in keys]))     # single global threshold (using this work's score)

    def penalty(score, err, t):
        m = score < t
        return (float(np.mean(err[m])) if m.sum() else np.nan), float(np.mean(err[~m])) if (~m).sum() else np.nan, float(m.mean())

    def decision_metrics(score, err, frac, tol=EPS):
        """Call the lowest-scoring frac suspicious -> precision / recall / F1 for "does the error exceed tolerance"."""
        n = max(1, int(round(frac * len(score))))
        idx = np.argsort(score)[:n]
        flag = np.zeros(len(score), bool); flag[idx] = True
        bad = err > tol
        tp = int((flag & bad).sum()); fp = int((flag & ~bad).sum()); fn = int((~flag & bad).sum())
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        return prec, rec, (2 * prec * rec / max(1e-9, prec + rec))

    def risk_at(score, err, frac):
        """Fix the flagged rate at frac and return the mean error of the flagged structures
        (comparable only at an equal flagged rate)."""
        n = max(1, int(round(frac * len(score))))
        idx = np.argsort(score)[:n]
        return float(np.mean(err[idx]))

    rows = []
    for k in keys:
        p_own, h_own, f_own = penalty(A_ev[k], err_ev[k], thA[k])
        p_gl, h_gl, f_gl = penalty(A_ev[k], err_ev[k], th_global_A)
        p_B, h_B, f_B = penalty(B_ev[k], err_ev[k], thB[k])
        p_C, h_C, f_C = penalty(C_ev, err_ev[k], thC[k])
        p_D, h_D, f_D = penalty(D_ev, err_ev[k], thD[k])
        rows.append(dict(key=k, name=meta[k]["name"], corpus=corpus_group(meta[k]["corpus"]),
                         aucA=auc(A_ev[k][ev_pos], A_ev[k][ev_neg]),
                         aucB=auc(B_ev[k][ev_pos], B_ev[k][ev_neg]),
                         penalty_own=p_own, penalty_glob=p_gl, penalty_B=p_B,
                         penalty_C=p_C, penalty_D=p_D, flagged=f_own,
                         fA=f_own, fB=f_B, fC=f_C, fD=f_D,
                         rA=risk_at(A_ev[k], err_ev[k], 0.65), rB=risk_at(B_ev[k], err_ev[k], 0.65),
                         rC=risk_at(C_ev, err_ev[k], 0.65), rD=risk_at(D_ev, err_ev[k], 0.65),
                         rG=risk_at(A_ev[k], err_ev[k], f_gl) if f_gl > 0 else np.nan,
                         aucA_pert=auc(A_ev[k][pert_pos], A_ev[k][pert_neg]),
                         aucB_pert=auc(B_ev[k][pert_pos], B_ev[k][pert_neg]),
                         aucA_rand=auc(A_ev[k][rand_pos], A_ev[k][rand_neg]),
                         aucB_rand=auc(B_ev[k][rand_pos], B_ev[k][rand_neg]),
                         rA_pert=risk_at(A_ev[k][pert_mask], err_ev[k][pert_mask], 0.65),
                         rB_pert=risk_at(B_ev[k][pert_mask], err_ev[k][pert_mask], 0.65),
                         rC_pert=risk_at(C_ev[pert_mask], err_ev[k][pert_mask], 0.65),
                         rD_pert=risk_at(D_ev[pert_mask], err_ev[k][pert_mask], 0.65),
                         decG=decision_metrics(A_ev[k], err_ev[k], f_gl) if f_gl > 0 else (np.nan,)*3,
                         **{f"dec{c}": decision_metrics(*args_, 0.65)
                            for c, args_ in (("A", (A_ev[k], err_ev[k])), ("B", (B_ev[k], err_ev[k])),
                                             ("C", (C_ev, err_ev[k])), ("D", (D_ev, err_ev[k])))}))
    aucC, aucD = auc(C_ev[ev_pos], C_ev[ev_neg]), auc(D_ev[ev_pos], D_ev[ev_neg])
    print(f"(the volume/NN baselines are structure-level scores, independent of the model)")

    # ---- print ----
    print(f"{'model':22s} {'AUC(KDE)':>9s} {'AUC(dist)':>10s} {'penalty(own)':>13s} {'penalty(global θ)':>18s} {'penalty(dist)':>14s}")
    for r in sorted(rows, key=lambda r: r["aucA"]):
        print(f"{r['name']:22s} {r['aucA']:9.3f} {r['aucB']:10.3f} {r['penalty_own']*1000:11.0f} meV/Å "
              f"{r['penalty_glob']*1000:16.0f} meV/Å {r['penalty_B']*1000:12.0f} meV/Å")
    # The caption/main text use in-dist vs **perturbed** (the screening-relevant regime), so the
    # bars use *_pert; the negative-control lines must use the same regime too (the bars
    # originally used aucA/aucB, which include random).
    aA = np.array([r["aucA_pert"] for r in rows]); aB = np.array([r["aucB_pert"] for r in rows])
    pO = np.array([r["penalty_own"] for r in rows]); pG = np.array([r["penalty_glob"] for r in rows])
    pB = np.array([r["penalty_B"] for r in rows]); pD = np.array([r["penalty_D"] for r in rows])
    pC = np.array([r["penalty_C"] for r in rows])
    print(f"\nAUC: KDE {aA.min():.3f}–{aA.max():.3f} (median {np.median(aA):.3f}) | "
          f"raw distance {aB.min():.3f}–{aB.max():.3f} (median {np.median(aB):.3f})")
    print(f"      negative controls: volume per atom AUC={aucC:.3f}, mean NN distance AUC={aucD:.3f}")
    print(f"median penalty: per-model θ {np.median(pO)*1000:.0f} | global θ {np.median(pG)*1000:.0f} "
          f"| raw distance {np.median(pB)*1000:.0f} | volume baseline {np.nanmedian(pC)*1000:.0f} meV/Å")
    print(f"median flagged rate: ours θ {np.median([r['fA'] for r in rows]):.2f} | global θ "
          f"{np.median([r['fB'] for r in rows]):.2f} (same score) | raw distance {np.median([r['fB'] for r in rows]):.2f}"
          f" | volume {np.median([r['fC'] for r in rows]):.2f}")
    rA = np.array([r["rA"] for r in rows]); rB = np.array([r["rB"] for r in rows])
    rC = np.array([r["rC"] for r in rows]); rD = np.array([r["rD"] for r in rows])
    print(f"\nrisk at an equal 65% flagged rate (mean force error of the flagged structures, lower is better):")
    print(f"  ours (KDE + per-model θ) {np.median(rA)*1000:.0f} meV/Å | raw distance {np.median(rB)*1000:.0f} "
          f"| volume baseline {np.median(rC)*1000:.0f} | NN distance baseline {np.median(rD)*1000:.0f}")
    print(f"global threshold θ_global={th_global_A:.3f} (median of the per-model θ_f); per-model θ range "
          f"{min(thA.values()):.3f}–{max(thA.values()):.3f}")

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))
    order = np.argsort(aA); labels = [rows[i]["name"] for i in order]; y = np.arange(len(order))
    gcol = [GROUP_COLOR[rows[i]["corpus"]] for i in order]

    ax = axes[0]
    ax.barh(y - 0.2, aA[order], height=0.4, color="#1a4d8f", label="KDE coverage (this work)")
    ax.barh(y + 0.2, aB[order], height=0.4, color="#e08a00", label="raw $k$-NN distance")
    ax.axvline(auc(C_ev[pert_pos], C_ev[pert_neg]), color="#2e7d32", ls=":", lw=1.6,
               label="volume baseline")
    ax.axvline(auc(D_ev[pert_pos], D_ev[pert_neg]), color="#7b1fa2", ls="-.", lw=1.4,
               label="geometric NN baseline")
    ax.axvline(0.5, color="0.45", lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5); ax.set_xlim(0.4, 1.0)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("AUC, in-distribution vs perturbed")
    ax.set_title("A  ranking quality\n(the regime a screen operates in)")
    ax.legend(fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.grid(alpha=0.25, axis="x"); ax.set_xlim(0.35, 1.02)

    ax = axes[1]
    names = ["ours", "distance", "volume", "geom. NN"]
    prec = [np.median([r[f"dec{c}"][0] for r in rows]) for c in "ABCD"]
    rec = [np.median([r[f"dec{c}"][1] for r in rows]) for c in "ABCD"]
    x = np.arange(4)
    ax.bar(x - 0.2, prec, 0.4, color="#1a4d8f", label="precision")
    ax.bar(x + 0.2, rec, 0.4, color="#c0392b", label="recall")
    for i, (p_, r_) in enumerate(zip(prec, rec)):
        ax.text(i - 0.2, p_ + 0.02, f"{p_:.2f}", ha="center", fontsize=8.5)
        ax.text(i + 0.2, r_ + 0.02, f"{r_:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9.5)
    ax.set_xlabel("score (circles = this work's pipeline)")
    ax.set_ylim(0, 1.12); ax.set_ylabel("share at a matched 65% flagged rate")
    ax.set_title("B  does the flag find the failures?\n(positive = error exceeds $\\varepsilon_f$)")
    ax.legend(fontsize=9, frameon=False, loc="upper right"); ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(str(Path(args.out).with_suffix(".png")), dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
