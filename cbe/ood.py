# -*- coding: utf-8 -*-
"""Controlled OOD perturbation protocol: split "extrapolation" into four gradable classes.

The four classes (the four sources of extrapolation):
  - volume      volume / pressure (isotropic scaling of the cell)
  - thermal     temperature (random displacements, a zeroth-order model of thermal disorder)
  - composition composition (element substitution / doping)
  - defects     defects (vacancies / interstitials)

Each class is graded from "mild" to "severe", so that one can quantify when coverage fails.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase import Atom


# ---- single-step perturbations ----
def perturb_volume(atoms: Atoms, scale: float) -> Atoms:
    """Isotropically scale the cell. scale<1 compresses, >1 expands (fractional coords kept)."""
    a = atoms.copy()
    a.set_cell(a.get_cell() * scale, scale_atoms=True)
    return a


def perturb_thermal(atoms: Atoms, std: float, seed: int = None) -> Atoms:
    """Random displacements (a zeroth-order approximation of thermal disorder)."""
    rng = np.random.default_rng(seed)
    a = atoms.copy()
    a.positions += rng.normal(0.0, std, a.positions.shape)
    return a


def substitute(atoms: Atoms, species_map, n_subs: int = 1, seed: int = None) -> Atoms:
    """Element substitution: randomly pick n_subs atoms and replace them with a new element.

    species_map: old element -> new element (or just the new element name, which replaces
    every atom).
    """
    rng = np.random.default_rng(seed)
    a = atoms.copy()
    symbols = np.array(a.get_chemical_symbols())
    cand = np.arange(len(a))
    if isinstance(species_map, str):
        species_map = {s: species_map for s in np.unique(symbols)}
    eligible = [i for i in cand if symbols[i] in species_map]
    if not eligible:
        return a
    idx = rng.choice(eligible, size=min(n_subs, len(eligible)), replace=False)
    for i in idx:
        a.symbols[i] = species_map[symbols[i]]
    return a


def remove_atom(atoms: Atoms, index: int = 0) -> Atoms:
    """Remove one atom (a vacancy defect)."""
    a = atoms.copy()
    del a[index]
    return a


def add_interstitial(atoms: Atoms, species: str, position=None,
                     seed: int = None) -> Atoms:
    """Add one interstitial atom (at a random position in the cell by default)."""
    rng = np.random.default_rng(seed)
    a = atoms.copy()
    if position is None:
        position = rng.random(3) @ a.get_cell()
    a.append(Atom(species, position=position))
    return a


# ---- graded OOD generation ----
def make_ood_volume(base_list, scales):
    """Volume extrapolation: scale base by several scales. Returns [(label, atoms)]."""
    out = []
    for s in scales:
        for a in base_list:
            out.append((f"volume_s{s}", perturb_volume(a, s)))
    return out


def make_ood_thermal(base_list, stds, seed=0):
    """Thermal-disorder extrapolation: jitter base with several displacement magnitudes."""
    out = []
    for std in stds:
        for a in base_list:
            out.append((f"thermal_s{std}", perturb_thermal(a, std, seed=seed)))
    return out


def make_ood_composition(base_list, species_maps, n_subs=1, seed=0):
    """Composition extrapolation: element substitution. species_maps may hold several
    dicts, one per group."""
    out = []
    for i, sm in enumerate(species_maps):
        for a in base_list:
            out.append((f"composition_{i}", substitute(a, sm, n_subs=n_subs, seed=seed)))
    return out


def make_ood_defects(base_list, mode="vacancy", indices=None, species=None, seed=0):
    """Defect extrapolation: vacancy or interstitial."""
    out = []
    for j, a in enumerate(base_list):
        if mode == "vacancy":
            idx = (indices[j] if indices is not None else 0)
            out.append((f"vacancy", remove_atom(a, idx)))
        elif mode == "interstitial":
            out.append((f"interstitial", add_interstitial(a, species, seed=seed)))
        else:
            raise ValueError(f"unknown defect mode: {mode}")
    return out


def generate_ood_protocol(base_list, *, volume_scales=(0.85, 0.90, 1.10, 1.15),
                          thermal_stds=(0.1, 0.3, 0.6),
                          species_maps=None, defect_modes=("vacancy",)):
    """Generate all four OOD classes in one call (each class graded).

    Returns a dict {group: [(label, atoms), ...]}; score each group with a CoverageModel to see
    how coverage changes with severity.
    """
    protocol = {
        "volume": make_ood_volume(base_list, volume_scales),
        "thermal": make_ood_thermal(base_list, thermal_stds),
    }
    if species_maps is not None:
        protocol["composition"] = make_ood_composition(base_list, species_maps)
    for mode in defect_modes:
        protocol[f"defect_{mode}"] = make_ood_defects(base_list, mode=mode,
                                                      species="H")
    return protocol
