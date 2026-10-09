# -*- coding: utf-8 -*-
"""θ for the relative criterion (a complement to the paper's absolute-tolerance criterion, in the SI).

Absolute criterion: θ_abs = min{c : q̂σ(c) ≤ ε_f}
  Problem: when ε_f is smaller than the model's own in-distribution error, the band exceeds ε_f
  everywhere -> θ=1 -> everything is flagged (the tolerance floor).

Relative criterion: θ_rel = min{c : q̂σ(c) ≤ κ · q̂σ(1)}, with κ = 3 in this work
  Meaning: "the band is more than 3x wider than this model's own in-distribution band", i.e.
  the model is 3x worse than usual.
  Advantage: it always yields a meaningful θ and is unaffected by the tolerance floor; it
  answers "where is the model clearly worse than usual".

Diagnostic set: 540 corpus samples + 311 (half of the difficult structures); evaluation: 312 held-out.
"""
import argparse, pickle, sys
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
from coverage_matched import load_dual, slice_dual, per_index_coverage, indexes_for
from figs_common import data, data_glob, corpus_group

K, ALPHA, KAPPA = 30, 0.1, 3.0
CS = np.linspace(0, 1, 4001)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, default=KAPPA)
    ap.add_argument("--out", default=str(ROOT / "DATA/thetas_rel.pkl"))
    args = ap.parse_args()

    meta = pickle.load(open(data("roster_meta.pkl"), "rb"))
    th_abs = pickle.load(open(data("thetas.pkl"), "rb"))["theta_f"]
    held, keep = np.load(data("calib_split.npz"))["held_out"], np.load(data("calib_split.npz"))["keep"]
    d_cal = load_dual(data("dist_calib.npz")); d_test = load_dual(data("dist_test.npz"))
    dh_full = slice_dual(d_test, keep)
    cab = {}; cte = {}
    for f in data_glob("eval_roster_calib_*.pkl"): cab.update(pickle.load(open(f, "rb")))
    for f in data_glob("eval_roster_test_*.pkl"): cte.update(pickle.load(open(f, "rb")))
    keys = [k for k in cab if k in cte]

    def cov(dual, k):
        return np.max(np.stack([per_index_coverage(dual, t, k=K)
                                for t in indexes_for(meta[k]["corpus"])], 0), 0)

    rel, rows = {}, []
    for k in keys:
        C = np.r_[cov(d_cal, k), cov(dh_full, k)[~held]]
        E = np.r_[np.asarray(cab[k], float), np.asarray(cte[k], float)[~held]]
        cal = ConformalCalibrator(alpha=ALPHA, seed=0).fit(C, E)
        # compare like with like: the difficulty function σ(c) (a median-like scale) against
        # κ x the median in-distribution force error.
        # q̂σ(c) cannot be used here: it is a conservative upper bound of 90%-quantile character,
        # not the same kind of quantity as "κ x the median".
        sig = np.array([float(cal.sigma_(c)) for c in CS])
        base = float(np.median(E[C > 0.7]))          # the accuracy the model actually attains
        ok = np.where(sig <= args.kappa * base)[0]
        rel[k] = float(CS[ok.min()]) if len(ok) else 1.0
        # evaluation
        Ce = cov(dh_full, k)[held]; Ee = np.asarray(cte[k], float)[held]
        def at(t):
            m = Ce < t
            return (float(np.mean(Ee[m])) * 1000 if m.sum() else np.nan, float(m.mean()))
        pa, fa = at(th_abs[k]); pr, fr = at(rel[k])
        rows.append(dict(key=k, name=meta[k]["name"], corpus=corpus_group(meta[k]["corpus"]),
                         th_abs=th_abs[k], th_rel=rel[k], flagged_abs=fa, flagged_rel=fr,
                         pen_abs=pa, pen_rel=pr,
                         ind=float(np.median(E[C > 0.7])) * 1000))

    print(f"relative criterion: θ_rel = min{{c : σ(c) ≤ {args.kappa:.0f} × median in-distribution force error}} (σ is a median-like scale, so both sides are the same kind of quantity)\n")
    print(f"{'model':22s} {'corpus':26s} {'in-dist MAE':>10s} {'θ_abs':>6s} {'θ_rel':>6s} "
          f"{'flagged(abs/rel)':>14s} {'penalty abs/rel (meV/Å)':>24s}")
    for r in sorted(rows, key=lambda r: r["th_rel"]):
        print(f"{r['name']:22s} {r['corpus']:26s} {r['ind']:8.0f} meV/Å {r['th_abs']:6.3f} {r['th_rel']:6.3f} "
              f"{r['flagged_abs']:6.0%}/{r['flagged_rel']:<6.0%} {r['pen_abs']:10.0f}/{r['pen_rel']:<10.0f}")
    ta = np.array([r["th_abs"] for r in rows]); tr = np.array([r["th_rel"] for r in rows])
    fr = np.array([r["flagged_rel"] for r in rows])
    print(f"\nθ_abs {ta.min():.3f}–{ta.max():.3f} (median {np.median(ta):.3f})"
          f" | θ_rel {tr.min():.3f}–{tr.max():.3f} (median {np.median(tr):.3f})")
    print(f"flagged rate of the relative criterion {fr.min():.0%}–{fr.max():.0%} (median {np.median(fr):.0%})")
    pickle.dump({"kappa": args.kappa, "theta_rel": rel,
                 "note": "theta_rel = min{c : sigma(c) <= kappa * median(error | coverage>0.7)}; sigma is the isotonic difficulty scale (median-like), so both sides are the same kind of quantity"},
                open(args.out, "wb"))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
