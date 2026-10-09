# -*- coding: utf-8 -*-
"""VASP result parsing: read energies/forces back from vasprun.xml / OUTCAR for cbe.eval.

Uses ASE's built-in vasp-xml / vasp-out parsers (well tested); the Atoms they return carry a
calculator whose results hold the energy/forces/stress.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read


def read_vasp_atoms(dir_or_file, index=-1) -> Atoms:
    """Read VASP output (vasprun.xml preferred, OUTCAR as fallback), returning Atoms with a calculator.

    index: which ionic step to take; -1 = the last step (converged / single point).
    """
    p = Path(dir_or_file)
    if p.is_dir():
        for name, fmt in [("vasprun.xml", "vasp-xml"), ("OUTCAR", "vasp-out")]:
            if (p / name).exists():
                return read(str(p / name), format=fmt, index=index)
        raise FileNotFoundError(f"no vasprun.xml / OUTCAR in {p}")
    fmt = "vasp-xml" if p.name.endswith(".xml") else "vasp-out"
    return read(str(p), format=fmt, index=index)


def read_vasp_results(dir_or_file, index=-1):
    """Return (atoms, energy, forces).

    energy: total energy (eV); forces: (N, 3) atomic forces (eV/Å).
    """
    atoms = read_vasp_atoms(dir_or_file, index=index)
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces())
    return atoms, energy, forces


def collect_vasp_results(dirs):
    """Walk a batch of VASP directories and aggregate into (structures, energies, forces).

    The returned structures are the structure of each calculation (so an MLIP can be compared on
    the same structures); energies is an (n,) array (total energy in eV); forces is a
    list[np.ndarray] (forces in eV/Å), each of shape (n_atoms, 3) - structures have different
    atom counts, so they cannot be stacked with np.asarray.
    """
    structs, es, fs = [], [], []
    for d in sorted(dirs):
        try:
            a, e, f = read_vasp_results(d)
        except Exception as ex:  # skip directories that are unfinished/corrupt
            print(f"[skip] {d}: {ex}")
            continue
        structs.append(a)
        es.append(e)
        fs.append(f)
    return structs, np.asarray(es), fs


def write_reference_xyz(structures, energies, forces, path):
    """Write (structures + VASP energies/forces) to xyz for later use by CBE.

    The energy (a scalar) goes to info['REF_energy']; the forces (per atom) go to
    arrays['REF_forces'].
    """
    from ase.io import write
    outs = []
    for a, e, f in zip(structures, energies, forces):
        aa = a.copy()
        aa.info["REF_energy"] = float(e)
        aa.arrays["REF_forces"] = np.asarray(f, dtype=float)
        outs.append(aa)
    write(path, outs)
    print(f"wrote {len(outs)} reference structures to {path}")
    return outs
