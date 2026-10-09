# -*- coding: utf-8 -*-
"""Evaluation layer: compute "model vs reference" errors for the benchmark and calibration.

Both the model and the reference are plugged in as ASE calculators (MACE-MP-0 / CHGNet / any
MLIP).
"""
from __future__ import annotations

import numpy as np
from ase import Atoms


def energy(calc, atoms: Atoms) -> float:
    a = atoms.copy()
    a.calc = calc
    return float(a.get_potential_energy())


def forces(calc, atoms: Atoms) -> np.ndarray:
    a = atoms.copy()
    a.calc = calc
    return np.asarray(a.get_forces())


def energy_error(calc, ref, atoms: Atoms, per_atom: bool = True) -> float:
    """Energy error; divided by the number of atoms when per_atom=True (comparable across
    systems)."""
    e = abs(energy(calc, atoms) - energy(ref, atoms))
    return e / len(atoms) if per_atom else e


def force_error(calc, ref, atoms: Atoms) -> float:
    """Force error = mean over atoms of |F_model - F_ref|."""
    df = forces(calc, atoms) - forces(ref, atoms)
    return float(np.mean(np.linalg.norm(df, axis=1)))


def evaluate(calc, ref, atoms_list):
    """For a batch of structures, return (energy_errors, force_errors) as arrays."""
    e = np.array([energy_error(calc, ref, a) for a in atoms_list])
    f = np.array([force_error(calc, ref, a) for a in atoms_list])
    return e, f


def load_calculator(name, device="cpu", **kwargs):
    """Load a common MLIP calculator by name.

    name: 'mace-mp-0-small' | 'mace-mp-0-medium' | 'mace-mp-0-large' | 'chgnet' | 'emt'
    device: 'cpu' or 'cuda' (note: coverage descriptors always use the MACE-MP-0-small latent
            features with one shared default_dtype, while energies/forces use each model's own
            calculator)
    """
    if name == "emt":
        from ase.calculators.emt import EMT
        return EMT()
    if name.startswith("mace"):
        from mace.calculators import mace_mp
        size = {"mace-mp-0-small": "small",
                "mace-mp-0-medium": "medium",
                "mace-mp-0-large": "large"}.get(name, "small")
        return mace_mp(model=size, device=device, default_dtype="float64", **kwargs)
    if name == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator()
    raise ValueError(f"unknown model: {name}")
