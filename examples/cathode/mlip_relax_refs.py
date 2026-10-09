"""Energy calculations for the voltage and formation-energy pipeline (MLIP side).

For the three groups of structures in ``voltage_vasp/`` (28 elemental references,
75 delithiated and 75 lithiated phases), every MLIP performs a full relaxation
(FrechetCellFilter, atoms and cell, matching the DFT ISIF=3 protocol; O2 molecules
are the exception and relax the atoms only), and the relaxed energies are stored.

Run once in each environment, so that the models each one can load are kept
(try/except). Merging the outputs of the two runs gives everything needed to
compare V and E_form against DFT.

Usage:

  python mlip_relax_refs.py --vasp-dir voltage_vasp/ --out mlip_voltage_refs.pkl
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

from cbe.eval import load_calculator


def load_models(device):
    models = {}
    for name in ["mace-mp-0-small", "mace-mp-0-medium", "mace-mp-0-large", "chgnet"]:
        try:
            models[name] = load_calculator(name, device=device)
        except Exception as e:
            print(f"[skip] {name}: {e!r}", flush=True)
    # the three models that only the mlip7 environment can load
    for label, loader in [
            ("SevenNet-0", lambda: __import__("sevenn.sevennet_calculator",
             fromlist=["SevenNetCalculator"]).SevenNetCalculator(model="7net-0", device=device)),
            ("ORB-v2", lambda: _load_orb(device)),
            ("MatterSim-v1", lambda: _load_mattersim(device))]:
        try:
            models[label] = loader()
        except Exception as e:
            print(f"[skip] {label}: {e!r}", flush=True)
    return models


def _load_orb(device):
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator
    return ORBCalculator(model=pretrained.orb_v2(device=device), device=device)


def _load_mattersim(device):
    from mattersim.forcefield import MatterSimCalculator
    ckpt = ".cache/mattersim/mattersim-v1.0.0-1M.pth"
    return MatterSimCalculator(device=device, load_path=ckpt)


def relax(calc, atoms, full=True, fmax=0.05, steps=200):
    from ase.optimize import BFGS
    from ase.filters import FrechetCellFilter
    am = atoms.copy()
    am.calc = calc
    try:
        if full:
            opt = BFGS(FrechetCellFilter(am))
        else:
            opt = BFGS(am)
        opt.run(fmax=fmax, steps=steps)
        conv = bool(opt.converged())
    except Exception:
        conv = False
    try:
        e = float(am.get_potential_energy())
    except Exception:
        e = np.nan
    return e, conv


def read_structures_from_dirs(root, sub, n):
    from ase.io import read
    out = []
    for i in range(n):
        p = Path(root) / sub / f"{i:04d}" / "POSCAR"
        try:
            out.append(read(str(p)))
        except Exception:
            out.append(None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vasp-dir", required=True, help="the voltage_vasp/ directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    elem = read_structures_from_dirs(args.vasp_dir, "elements", 28)
    delith = read_structures_from_dirs(args.vasp_dir, "delithiated", 75)
    lith = read_structures_from_dirs(args.vasp_dir, "lithiated_relax", 75)
    print(f"structures: elements {sum(x is not None for x in elem)}/28, "
          f"delithiated {sum(x is not None for x in delith)}/75, "
          f"lithiated {sum(x is not None for x in lith)}/75", flush=True)

    models = load_models(args.device)
    if not models:
        print("no usable model in this environment")
        return

    results = {}
    for name, calc in models.items():
        print(f"running {name} ...", flush=True)
        ee, de, le = [], [], []
        for a in elem:
            e, _ = relax(calc, a, full=(a is not None and len(a) > 2),  # O2 relaxes the atoms only
                         fmax=args.fmax, steps=args.steps) if a is not None else (np.nan, False)
            ee.append(e)
        for a in delith:
            e, _ = relax(calc, a, full=True, fmax=args.fmax, steps=args.steps) if a is not None else (np.nan, False)
            de.append(e)
        for a in lith:
            e, _ = relax(calc, a, full=True, fmax=args.fmax, steps=args.steps) if a is not None else (np.nan, False)
            le.append(e)
        # NOTE: elements_e holds whole-cell totals (Li bcc cell = 2 atoms, fcc = 4, O2 = 2, P4 = 4);
        #     the consumer voltage_compare.py converts them to per-atom references using
        #     the atom count in each elemental POSCAR.
        results[name] = {"elements_e": np.array(ee),
                         "delith_e": np.array(de),
                         "lith_e": np.array(le)}
        print(f"  {name} done (elements mean={np.nanmean(ee):.2f} eV)", flush=True)

    with open(args.out, "wb") as fh:
        pickle.dump(results, fh)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
