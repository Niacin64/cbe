# -*- coding: utf-8 -*-
"""Table 4 (JCTC version): what the force-calibrated θ_f actually delivers on a "voltage tolerance".

For the 74 cathode candidates (layered LiFeO₂ excluded), each model's own θ_f and a voltage
tolerance of 300 mV:
  - the fraction of candidates kept;
  - the mean |ΔV| of the kept set / dropped set / all candidates;
  - the number of candidates with |ΔV| > 300 mV, and how many of them the guard "misses" (keeps),
    with the corresponding fraction.

Everything comes from the cache (DATA/dist_cathode.npz, case_cathode/*.pkl); no model runs are
needed.
Usage:
  PYTHONPATH=CBE:CBE/examples python voltage_triage.py
"""
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

from coverage_matched import matched_coverage
from figs_common import data
from fig_cathode8 import load_cathode, model_voltage

TOL_MV = 300.0


def main():
    tpl, cov_dual, errs, volt, meta, Vd, keep = load_cathode()
    TH = pickle.load(open(data("thetas.pkl"), "rb"))["theta_f"]
    rows = []
    for k in errs:
        cov = matched_coverage(cov_dual, meta[k]["corpus"])[keep]
        dV = np.abs(model_voltage(volt[k], tpl)[keep] - Vd[keep]) * 1000.0
        trust = cov >= TH[k]
        bad = dV > TOL_MV
        rows.append((meta[k]["name"], trust.mean() * 100, float(np.mean(dV[trust])),
                     float(np.mean(dV[~trust])), float(np.mean(dV)),
                     int(bad.sum()), int((bad & trust).sum()),
                     (bad & trust).sum() / max(bad.sum(), 1) * 100))
    rows.sort(key=lambda r: r[2])

    print(f"{'model':22s} {'kept%':>6s} {'|dV| kept':>10s} {'dropped':>8s} {'all':>6s} "
          f"{'>tol':>5s} {'missed':>7s} {'miss%':>6s}")
    for r in rows:
        print(f"{r[0]:22s} {r[1]:6.0f} {r[2]:10.0f} {r[3]:8.0f} {r[4]:6.0f} {r[5]:5d} {r[6]:7d} {r[7]:6.0f}")
    a = np.array(rows, dtype=object)
    kept = np.array([r[1] for r in rows]); mk = np.array([r[2] for r in rows])
    ma = np.array([r[4] for r in rows]); nm = np.array([r[6] for r in rows]); pc = np.array([r[7] for r in rows])
    print(f"\n18 models: kept {kept.min():.0f}–{kept.max():.0f}% (median {np.median(kept):.0f}%); "
          f"kept-set mean|ΔV| {mk.min():.0f}–{mk.max():.0f} mV (all {ma.min():.0f}–{ma.max():.0f} mV); "
          f"missed {nm.min()}–{nm.max()} ({pc.min():.0f}–{pc.max():.0f}%, median {np.median(pc):.0f}%)")
    print(f"kept/dropped mean|ΔV| ratio range {np.min(mk/np.array([r[3] for r in rows])):.2f}–"
          f"{np.max(mk/np.array([r[3] for r in rows])):.2f}")


if __name__ == "__main__":
    main()
