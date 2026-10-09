"""Average delithiation voltage for the full model roster.

Each model relaxes the structures independently, following the same protocol as
the earlier seven-model run (see mlip_relax_refs.py and mlip_relax_delith_fixed.py):
  28 elemental references: relax (O2 moves atoms only)
  75 lithiated phases:     relax
  75 delithiated phases:   fixed cell, atoms only (the ISIF=2 protocol)
  V = (E_del + n*E_Li - E_lith) / n, with n = 3/2/4 for layered/spinel/olivine
Note: the elemental energies are whole-cell totals; the consumer converts them to
per-atom references using the atom count in each POSCAR (Li bcc = 2, fcc = 4,
O2 = 2, P4 = 4).

Usage:

  python voltage_roster.py --vasp-dir voltage_vasp/ --out mlip_voltage_roster.pkl
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
ROOT = Path(__file__).resolve().parent.parent.parent

from cbe.data import read_structures
from eval_roster import ROSTER, load_model


def read_dirs(base, kind, n):
    """Read POSCARs from voltage_vasp/<kind>/0000.. (returns None for missing ones)."""
    from ase.io import read
    out = []
    for i in range(n):
        p = Path(base) / kind / f"{i:04d}" / "POSCAR"
        out.append(read(str(p)) if p.exists() else None)
    return out


def relax_atoms_only(calc, atoms, fmax=0.05, steps=200):
    from ase.optimize import BFGS
    am = atoms.copy(); am.calc = calc
    try:
        opt = BFGS(am); opt.run(fmax=fmax, steps=steps)
    except Exception:
        pass
    try:
        return float(am.get_potential_energy())
    except Exception:
        return np.nan


def relax_full(calc, atoms, fmax=0.05, steps=200, fix_cell=False):
    from ase.optimize import BFGS
    am = atoms.copy(); am.calc = calc
    try:
        if fix_cell:
            opt = BFGS(am)
        else:
            from ase.constraints import ExpCellFilter
            opt = BFGS(ExpCellFilter(am))
        opt.run(fmax=fmax, steps=steps)
    except Exception:
        pass
    try:
        return float(am.get_potential_energy())
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--vasp-dir", default=str(ROOT / "voltage_vasp"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    elem = read_dirs(args.vasp_dir, "elements", 28)
    delith = read_dirs(args.vasp_dir, "delithiated_fixed", 75)
    if all(x is None for x in delith):
        delith = read_dirs(args.vasp_dir, "delithiated", 75)
    lith = read_structures(str(ROOT / "case_cathode/relaxed_all.xyz"))[:75]
    print(f"elements {sum(x is not None for x in elem)}/28; delithiated {sum(x is not None for x in delith)}/75; "
          f"lithiated {len(lith)}/75", flush=True)

    out = {}
    if Path(args.out).exists():
        out = pickle.load(open(args.out, "rb"))
        print(f"resuming: {list(out)} already present", flush=True)

    for key in args.models:
        if key in out:
            print(f"[skip] {key} already present", flush=True)
            continue
        name, corpus, _, _ = ROSTER[key]
        try:
            calc = load_model(key, args.device)
        except Exception as e:
            print(f"[skip] {name}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            continue
        t0 = time.time()
        ee = [relax_full(calc, a, args.fmax, args.steps, fix_cell=(len(a) <= 2))
              if a is not None else np.nan for a in elem]
        print(f"  {name}: elements done ({time.time()-t0:.0f}s)", flush=True)
        le = [relax_atoms_only(calc, a, args.fmax, args.steps) for a in lith]
        print(f"  {name}: lithiated done ({time.time()-t0:.0f}s)", flush=True)
        de = [relax_atoms_only(calc, a, args.fmax, args.steps) if a is not None else np.nan
              for a in delith]
        out[key] = {"elements_e": np.array(ee), "lith_e": np.array(le), "delith_e": np.array(de)}
        with open(args.out, "wb") as fh:
            pickle.dump(out, fh)
        print(f"  {name} done ({time.time()-t0:.0f}s) -> {args.out}", flush=True)

    print("all models done")


if __name__ == "__main__":
    main()
