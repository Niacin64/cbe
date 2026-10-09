# -*- coding: utf-8 -*-
"""Case study 2: MACE-small relaxation + coverage for cathode candidates.

Pipeline (mimicking step 3 of screening + CBE triage):
  1. read candidates.xyz (unrelaxed candidates)
  2. MACE-small atomic relaxation (fixed cell, BFGS)
  3. coverage on the relaxed structures (MACE-small latent + 49M MPtrj index, k=30, Silverman h)
  4. write relaxed.xyz (with coverage/converged info) + coverage.csv

Usage (local):
  PYTHONPATH=CBE python cathode_relax_coverage.py \
      --candidates ../../case_cathode/candidates.xyz --index-dir mptrj_index_all/ \
      --out ../../case_cathode/relaxed.xyz
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mace.calculators import mace_mp
from cbe.coverage import CoverageModel
from cbe.descriptors import LatentDescriptor
from cbe.data import read_structures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--out", default="relaxed.xyz")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--bandwidth", type=float, default=0.2184)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=150)
    args = ap.parse_args()

    small = mace_mp(model="small", device="cpu", default_dtype="float64")
    desc = LatentDescriptor(
        lambda a: small.get_descriptors(a, invariants_only=True), dim=256,
        level="per_atom")
    cov = CoverageModel.from_index(
        str(Path(args.index_dir) / "index.faiss"),
        str(Path(args.index_dir) / "pca.npz"),
        desc, k=args.k, scoring="kde", coverage_agg="min",
        bandwidth=args.bandwidth)

    candidates = read_structures(args.candidates)
    print(f"{len(candidates)} candidates; starting MACE-small relaxation + coverage ...", flush=True)

    from ase.optimize import BFGS
    from ase.io import write

    rows = []
    relaxed = []
    for a in candidates:
        cid = a.info.get("candidate_id", "?")
        am = a.copy()
        am.calc = small
        try:
            opt = BFGS(am)
            opt.run(fmax=args.fmax, steps=args.max_steps)
            converged = bool(opt.converged())
        except Exception as e:
            print(f"[relax fail] {cid}: {e!r}", flush=True)
            converged = False
        c = float(cov.coverage(am))
        am.calc = None
        am.info["coverage"] = c
        am.info["converged"] = int(converged)
        relaxed.append(am)
        rows.append([cid, am.info.get("template", ""), am.info.get("tm", ""),
                     am.info.get("composition", ""), f"{c:.4f}", converged])
        print(f"  {cid:16s} cov={c:.3f}  {'ok' if converged else 'NOT converged'}", flush=True)

    write(args.out, relaxed)
    out_csv = Path(args.out).with_suffix(".csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate_id", "template", "tm", "composition",
                    "coverage", "converged"])
        w.writerows(rows)

    covs = np.array([a.info["coverage"] for a in relaxed], dtype=float)
    print(f"\ndone: {len(relaxed)} relaxed structures -> {args.out}")
    print(f"coverage range [{covs.min():.3f}, {covs.max():.3f}], "
          f"mean {covs.mean():.3f}")
    print(f"coverage gradient (sorted by tm) written to {out_csv}")


if __name__ == "__main__":
    main()
