# -*- coding: utf-8 -*-
"""Data layer: structure loading, element filtering, subset sampling.

Feeds real data to the CBE benchmark (MPtrj / OMat24 / your own), or reads structures from an
ASE trajectory.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.io import read


def read_structures(path, index=":", fmt=None):
    """Read structures from xyz/traj/any format ASE supports. Returns list[Atoms]."""
    atoms_list = read(path, index=index, format=fmt)
    if isinstance(atoms_list, Atoms):
        return [atoms_list]
    return list(atoms_list)


def filter_elements(structures, elements):
    """Keep only the structures whose atoms all lie in the elements set."""
    keep = set(elements)
    out = []
    for a in structures:
        if set(a.get_chemical_symbols()) <= keep:
            out.append(a)
    return out


def sample_subset(structures, n, seed=0):
    """Randomly (and reproducibly) sample n structures; returns all if n exceeds the total."""
    rng = np.random.default_rng(seed)
    n = min(n, len(structures))
    idx = rng.choice(len(structures), size=n, replace=False)
    return [structures[i] for i in idx]


def subsample_by_stride(structures, stride):
    """Subsample by a stride (keeps the time order, suitable for trajectory data)."""
    return list(structures[::stride])


def load_mptrj_subset(path_or_parquet, elements=None, n=None, seed=0):
    """Read a subset from the MPtrj parquet file (needs pandas + pyarrow).

    Note: the full MPtrj is ~10 GB, so download a subset or filter columns first.
    Returns list[Atoms] (with energy/force/stress info).
    """
    import pandas as pd
    df = pd.read_parquet(path_or_parquet)
    if elements is not None:
        # filter by material composition (coarse: keep only materials with the target elements)
        raise NotImplementedError("element filtering needs the material composition parsed; "
                                  "filter by materials_project first")
    atoms_list = []
    for _, row in df.iterrows():
        atoms_list.append(_row_to_atoms(row))
    if n is not None:
        atoms_list = sample_subset(atoms_list, n, seed=seed)
    return atoms_list


def _row_to_atoms(row):
    from ase import Atoms
    return Atoms(positions=row["structure"].cart_coords,
                 symbols=row["structure"].species,
                 cell=row["structure"].lattice.matrix,
                 pbc=True,
                 info={"energy": row.get("energy"),
                       "forces": row.get("forces")})
