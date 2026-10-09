# -*- coding: utf-8 -*-
"""Case study 2: Li-ion battery cathode candidate generation (element substitution).

Starting from two classic templates, substitute the transition metal TM to make new-composition candidates:
  - layered Li(TM)O2 (R-3m, template = LiCoO2)
  - spinel Li(TM)2O4 (Fd-3m, template = LiMn2O4)

The substituted TMs span a "common in MPtrj -> rare" spectrum, building a coverage gradient:
  Ti V Cr Mn Fe Co Ni Cu Zn Zr Nb Mo W Ta Hf Ag Ru

Candidates are **unrelaxed** structures (equivalent to what a generative model has just produced); then:
  1. MLIP relaxation (mimicking step 3 of screening)
  2. coverage (MACE-small + 49M MPtrj index)
  3. DFT single points (on a subset) as reference

Outputs: xyz + a candidate list csv under the case_cathode/ directory.
"""
import argparse
from pathlib import Path

import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


TM_LIST = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
           "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
           "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au"]


def build_layered(tm: str) -> Structure:
    """O3-type layered Li(TM)O2, R-3m (#166). a=2.815, c=14.05 Å."""
    return Structure.from_spacegroup(
        "R-3m", Lattice.hexagonal(2.815, 14.05),
        ["Li", tm, "O"],
        [[0, 0, 0], [0, 0, 0.5], [0, 0, 0.24]])


def build_spinel(tm: str) -> Structure:
    """Spinel Li(TM)2O4, Fd-3m (#227). a=8.24 Å.

    Note pymatgen's origin choice for Fd-3m: (0.5,0.5,0.5) is the 8-fold site, where Li goes;
    (0.125,0.125,0.125) is the 16-fold site, where TM goes, giving Li:TM:O = 1:2:4.
    """
    return Structure.from_spacegroup(
        "Fd-3m", Lattice.cubic(8.24),
        ["Li", tm, "O"],
        [[0.5, 0.5, 0.5], [0.125, 0.125, 0.125], [0.262, 0.262, 0.262]])


def build_olivine(tm: str) -> Structure:
    """Olivine Li(TM)PO4, Pnma (#62). a=10.33, b=6.01, c=4.69 Å."""
    return Structure.from_spacegroup(
        "Pnma", Lattice.from_parameters(10.33, 6.01, 4.69, 90, 90, 90),
        ["Li", tm, "P", "O", "O", "O"],
        [[0, 0, 0], [0.2822, 0.25, 0.9747], [0.0947, 0.25, 0.4183],
         [0.0967, 0.25, 0.7427], [0.4569, 0.25, 0.2060],
         [0.1655, 0.0465, 0.2854]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="case_cathode")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from ase.io import write
    rows = []
    atoms_list = []

    for tm in TM_LIST:
        for template, builder, comp_fmt in [
                ("layered", build_layered, "Li{}O2"),
                ("spinel", build_spinel, "Li{}2O4"),
                ("olivine", build_olivine, "Li{}PO4")]:
            s = builder(tm)
            comp = comp_fmt.format(tm)
            # pymatgen -> ase
            from pymatgen.io.ase import AseAtomsAdaptor
            atoms = AseAtomsAdaptor.get_atoms(s)
            atoms.info["template"] = template
            atoms.info["tm"] = tm
            atoms.info["composition"] = s.composition.reduced_formula
            atoms.info["group"] = f"{template}-{tm}"
            atoms.info["candidate_id"] = f"{template}_{tm}"
            atoms_list.append(atoms)
            rows.append((template, tm, s.composition.reduced_formula,
                         len(atoms), float(s.volume)))

    write(str(out / "candidates.xyz"), atoms_list)

    import csv
    with open(out / "candidates.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate_id", "template", "tm", "composition",
                    "n_atoms", "volume_A3"])
        for (template, tm, comp, n, vol), a in zip(rows, atoms_list):
            w.writerow([a.info["candidate_id"], template, tm, comp, n, f"{vol:.2f}"])

    print(f"generated {len(atoms_list)} candidates -> {out}/candidates.xyz")
    print(f"templates: layered Li(TM)O2 + spinel Li(TM)2O4 x {len(TM_LIST)} TMs")
    print("TM:", " ".join(TM_LIST))


if __name__ == "__main__":
    main()
