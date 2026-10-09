"""Recompute the delithiated phases with a fixed lithiated cell and atomic relaxation.

The earlier full relaxation with FrechetCellFilter let the delithiated phases
transform, which lowered the voltage by about 2 V. This script relaxes the atoms
only (the cell is locked to the lithiated one) and updates ``delith_e`` inside the
``mlip_voltage_*.pkl`` files, which is the standard protocol for an average voltage.

Usage (run once in each environment):

  python mlip_relax_delith_fixed.py --pkl mlip_voltage_mace.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- make the sibling example directories importable -------------------------
# this file lives in examples/<group>/; the shared helpers live in examples/common/
# and examples/index/, so both are added to sys.path whatever the working directory.
_HERE = Path(__file__).resolve().parent
for _extra in ("common", "index", "eval", _HERE.name):
    _p = _HERE.parent / _extra
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
# ----------------------------------------------------------------------------

from mlip_relax_refs import load_models, read_structures_from_dirs


def relax_atoms_only(calc, atoms, fmax=0.05, steps=200):
    from ase.optimize import BFGS
    am = atoms.copy()
    am.calc = calc
    try:
        opt = BFGS(am)          # atoms only, cell fixed
        opt.run(fmax=fmax, steps=steps)
        conv = bool(opt.converged())
    except Exception:
        conv = False
    try:
        e = float(am.get_potential_energy())
    except Exception:
        e = np.nan
    return e, conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="the mlip_voltage_*.pkl file to update")
    ap.add_argument("--vasp-dir", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    delith = read_structures_from_dirs(args.vasp_dir, "delithiated", 75)
    print(f"delithiated structures {sum(x is not None for x in delith)}/75", flush=True)

    data = pickle.load(open(args.pkl, "rb"))
    models = load_models(args.device)
    for name, calc in models.items():
        if name not in data:
            print(f"[skip] {name} not in the pkl", flush=True)
            continue
        print(f"running {name} ...", flush=True)
        es = []
        for a in delith:
            e, _ = relax_atoms_only(calc, a) if a is not None else (np.nan, False)
            es.append(e)
        es = np.array(es)
        # drop diverged points (collapse artefacts with |E| > 500 eV)
        es = np.where(np.abs(es) > 500, np.nan, es)
        data[name]["delith_e"] = es
        print(f"  {name} done (delith mean={np.nanmean(es):.1f} eV, NaN={np.isnan(es).sum()})", flush=True)

    pickle.dump(data, open(args.pkl, "wb"))
    print(f"updated -> {args.pkl}")


if __name__ == "__main__":
    main()
