# -*- coding: utf-8 -*-
"""Descriptor layer: map ASE Atoms to feature vectors.

Convention (important):
  - a per_atom descriptor returns (n_atoms, dim) -- each row is one atom's local environment.
  - a structure descriptor returns (dim,) -- the global feature of the whole structure.
CoverageModel uses this convention to decide whether the index is built on the atom level or
on the structure level.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from ase import Atoms


class Descriptor(ABC):
    """Abstract base class for descriptors."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Feature dimension."""

    @property
    @abstractmethod
    def level(self) -> str:
        """'per_atom' or 'structure'."""

    @abstractmethod
    def describe(self, atoms: Atoms) -> np.ndarray:
        """Return the features."""


class SOAPDescriptor(Descriptor):
    """SOAP (Smooth Overlap of Atomic Positions) descriptor backed by dscribe, rotation
    invariant.

    Parameters:
        r_cut: cutoff radius (Å)
        nmax: number of radial basis functions
        lmax: maximum angular momentum
        species: list of elements; inferred from the atoms when None
        average: True -> structure level (dim,); False -> per-atom (n_atoms, dim)
        periodic: whether to use periodic boundary conditions
    """

    def __init__(self, r_cut=5.0, nmax=6, lmax=4, sigma=1.0, species=None,
                 average=False, periodic=True):
        self.r_cut, self.nmax, self.lmax, self.sigma = r_cut, nmax, lmax, sigma
        self.species = species
        self.average = average
        self.periodic = periodic
        self._soap = None  # built lazily (needs species)

    def _build(self, species):
        from dscribe.descriptors import SOAP
        self._soap = SOAP(
            species=species,
            r_cut=self.r_cut,
            n_max=self.nmax,
            l_max=self.lmax,
            sigma=self.sigma,
            periodic=self.periodic,
            average="inner" if self.average else "off",
            sparse=False,
        )
        self._species = species

    def _ensure(self, atoms: Atoms):
        species = self.species or sorted(set(atoms.get_chemical_symbols()))
        if self._soap is None or list(self._species) != list(species):
            self._build(species)

    @property
    def dim(self) -> int:
        if self._soap is None:
            if not self.species:
                raise ValueError("the SOAP dimension needs species; pass species=[...] or "
                                 "call describe() once first")
            self._build(sorted(self.species))
        return self._soap.get_number_of_features()

    @property
    def level(self) -> str:
        return "structure" if self.average else "per_atom"

    def describe(self, atoms: Atoms) -> np.ndarray:
        self._ensure(atoms)
        return np.asarray(self._soap.create(atoms), dtype=np.float32)


class BehlerDescriptor(Descriptor):
    """Behler–Parrinello symmetry-function descriptor (pure numpy, no external dependency).

    A lightweight alternative as "model input" or as a "coverage descriptor". It contains:
      - G2 radial functions (several (eta, Rs) combinations)
      - G4 angular functions (several (eta, zeta, lambda) combinations)
    Returns per-atom features (n_atoms, dim).
    """

    def __init__(self, r_cut=5.0, radial=(0.5, 1.0, 2.0),
                 angular=((1.0, 1.0, 1.0), (1.0, 1.0, -1.0)),
                 structure_average=False):
        self.r_cut = r_cut
        self.radial = list(radial)
        self.angular = list(angular)
        self.structure_average = structure_average

    @property
    def dim(self) -> int:
        return len(self.radial) + len(self.angular)

    @property
    def level(self) -> str:
        return "structure" if self.structure_average else "per_atom"

    def _fc(self, r):
        rc = self.r_cut
        out = np.zeros_like(r)
        m = r < rc
        out[m] = 0.5 * (1.0 + np.cos(np.pi * r[m] / rc))
        return out

    def describe(self, atoms: Atoms) -> np.ndarray:
        n = len(atoms)
        feats = np.zeros((n, self.dim), dtype=np.float32)

        # minimum-image distance matrix (ASE's C implementation: fast, handles periodic images)
        D = atoms.get_all_distances(mic=True)
        np.fill_diagonal(D, np.inf)
        fc = self._fc(D)

        col = 0
        for eta in self.radial:
            feats[:, col] = np.sum(np.exp(-eta * D**2) * fc, axis=1)
            col += 1
        if self.angular:
            # angular terms: need the azimuth; computed from the ASE neighbour list
            # (off in the default demo for speed)
            pos = atoms.get_positions()
            for (eta, zeta, lam) in self.angular:
                acc = np.zeros(n)
                nl = _neighbor_pairs(atoms, self.r_cut)
                for (i, j, k) in nl:
                    vj = pos[j] - pos[i]
                    vk = pos[k] - pos[i]
                    rj = np.linalg.norm(vj)
                    rk = np.linalg.norm(vk)
                    if rj < 1e-8 or rk < 1e-8:
                        continue
                    cos = np.dot(vj, vk) / (rj * rk)
                    acc[i] += (1.0 + lam * cos)**zeta * np.exp(
                        -eta * (rj**2 + rk**2))
                feats[:, col] = 2.0**(1 - zeta) * acc
                col += 1

        if self.structure_average:
            return feats.mean(axis=0)
        return feats


def _neighbor_pairs(atoms: Atoms, r_cut: float):
    """Return (i, j, k) triplets: both j and k lie within r_cut of i."""
    from ase.neighborlist import neighbor_list
    i_idx, j_idx, _ = neighbor_list("ijD", atoms, r_cut)
    pairs = []
    for a in range(len(atoms)):
        js = j_idx[i_idx == a]
        if len(js) < 2:
            continue
        for u in range(len(js)):
            for v in range(u + 1, len(js)):
                pairs.append((a, js[u], js[v]))
    return pairs


class LatentDescriptor(Descriptor):
    """Use the MLIP's own latent/node features as the descriptor (the model-aligned route).

    Usage: pass a callable fn(atoms) -> np.ndarray; the caller is responsible for tapping the
    model's intermediate features and reducing them to rotation invariants. For MACE/NequIP
    node irreps one can take the degree-0 (scalar) channels or apply an invariant pooling to
    them (see the equivariant-to-invariant reduction of approach A, open item #4).
    """

    def __init__(self, fn, dim: int, level: str = "per_atom"):
        self.fn = fn
        self._dim = dim
        self._level = level

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def level(self) -> str:
        return self._level

    def describe(self, atoms: Atoms) -> np.ndarray:
        return np.asarray(self.fn(atoms), dtype=np.float32)


class ACEDescriptor(Descriptor):
    """Lightweight ACE (Atomic Cluster Expansion) invariant descriptor, pure numpy.

    Implements the 1-body and 2-body terms (standard truncated linear ACE):
      1-body : radial moments of sum_i R_n(r_ij) (blocked by element pair)
      2-body : rotation invariants of sum_{ij} R_n(r_ij) Y_lm(r̂_ij)
               B_{n1 n2 l} = sum_m A_{n1 l m} A_{n2 l m}* (same element pair)
    Plus the element count (one-hot) of each centre atom, so that different chemical
    environments stay separable.

    Needs only a neighbour list and no external dependency; invariant under rotation,
    translation and permutation, and its size is independent of the number of species
    (blocked by element pair using a fixed species list). It is an "engineering-ready" ACE
    variant, not the full PACE.

    Parameters:
        r_cut: cutoff radius (Å)
        n_max: number of radial basis functions (polynomial basis)
        l_max: maximum angular momentum
        species: fixed list of elements (keeps the dimension consistent); inferred when None
        level: 'per_atom' (default) or 'structure'
        periodic: periodic boundary conditions
    """

    def __init__(self, r_cut=5.0, n_max=5, l_max=3, species=None,
                 level="per_atom", periodic=True):
        self.r_cut, self.n_max, self.l_max = r_cut, n_max, l_max
        self.species = list(species) if species else None
        self._level = level
        self.periodic = periodic

    # ---------- basis functions ----------
    def _radial(self, r):
        """Polynomial radial basis R_n(r) = (r_cut - r)^(n+1) / r_cut^(n+1), zero at r_cut."""
        x = np.clip(1.0 - r / self.r_cut, 0.0, None)
        return np.stack([x ** (n + 1) for n in range(self.n_max)], axis=-1)

    @staticmethod
    def _ylm_unit(u, l_max):
        """Real spherical harmonics (unit vectors); returns a list of [(2l+1), ...] arrays."""
        x, y, z = u[..., 0], u[..., 1], u[..., 2]
        out = [np.ones_like(x)[..., None]]
        if l_max >= 1:
            c = np.sqrt(3.0 / (4.0 * np.pi))
            out.append(np.stack([c * y, c * z, c * x], axis=-1))
        if l_max >= 2:
            c1 = np.sqrt(15.0 / (4.0 * np.pi))
            c2 = np.sqrt(5.0 / (16.0 * np.pi))
            r2 = np.stack([c1 * x * y, c1 * y * z,
                           0.5 * c2 * (-x * x - y * y + 2 * z * z),
                           c1 * x * z, 0.5 * c1 * (x * x - y * y)], axis=-1)
            out.append(r2)
        if l_max >= 3:
            c1 = np.sqrt(35.0 / (32.0 * np.pi))
            c2 = np.sqrt(105.0 / (4.0 * np.pi))
            c3 = np.sqrt(21.0 / (32.0 * np.pi))
            r3 = np.stack([
                c1 * y * (3 * x * x - y * y), c2 * x * y * z,
                c1 * y * (-x * x - y * y + 4 * z * z),
                c3 * z * (-3 * x * x - 3 * y * y + 2 * z * z),
                c1 * x * (-x * x - y * y + 4 * z * z),
                c2 * z * (x * x - y * y), c1 * x * (x * x - 3 * y * y)], axis=-1)
            out.append(r3)
        return out

    def _describe_one(self, atoms, zs, pairs, fixed_species):
        Z = atoms.get_atomic_numbers()
        n_species = len(fixed_species)
        idx_of = {int(z): i for i, z in enumerate(fixed_species)}
        # dimension: element counts + 1-body (n_max*Z) + 2-body (n_max^2*l_max*Z^2)
        dim = n_species + self.n_max * n_species \
            + self.n_max * self.n_max * self.l_max * n_species * n_species
        feats = np.zeros((len(atoms), dim))
        i, j, D, Dv = pairs
        for a in range(len(atoms)):
            m = i == a
            if not m.any():
                continue
            jj, dd, dv = j[m], D[m], Dv[m]
            r = np.linalg.norm(dv, axis=1)
            R = self._radial(r)                       # (nb, n_max)
            u = dv / r[:, None]
            Y = self._ylm_unit(u, self.l_max)         # list of (nb, 2l+1)
            zi = idx_of.get(int(Z[a]), 0)
            acc = []
            # element counts
            cnt = np.zeros(n_species)
            for zj in Z[jj]:
                cnt[idx_of.get(int(zj), 0)] += 1
            acc.append(cnt)
            # 1-body: blocked by neighbour element
            one = np.zeros((n_species, self.n_max))
            zj_idx = np.array([idx_of.get(int(z), 0) for z in Z[jj]])
            for s in range(n_species):
                if (zj_idx == s).any():
                    one[s] = R[zj_idx == s].sum(axis=0)
            acc.append(one.ravel())
            # 2-body: A_{nlm} = sum_j R_n Y_lm, then contract over m
            two = np.zeros((n_species, n_species, self.n_max, self.n_max, self.l_max))
            for l in range(self.l_max):
                Yl = Y[l]                              # (nb, 2l+1)
                for s in range(n_species):
                    mask = zj_idx == s
                    if not mask.any():
                        continue
                    As = np.einsum("bn,bk->nk", R[mask], Yl[mask])
                    for s2 in range(s, n_species):
                        mask2 = zj_idx == s2
                        if not mask2.any():
                            continue
                        A2 = np.einsum("bn,bk->nk", R[mask2], Yl[mask2])
                        B = As @ A2.T                  # (n_max, n_max)
                        two[s, s2, :, :, l] = B
                        if s != s2:
                            two[s2, s, :, :, l] = B
            acc.append(two.ravel())
            feats[a] = np.concatenate(acc)
        return feats

    # ---------- interface ----------
    def describe(self, atoms: Atoms) -> np.ndarray:
        from ase.neighborlist import neighbor_list
        if self.species is None:
            self.species = sorted(set(atoms.get_chemical_symbols()))
            # represent species by atomic number
            from ase.data import atomic_numbers
            self.species = sorted(int(atomic_numbers[s]) for s in self.species)
        fixed = list(self.species)
        i, j, D = neighbor_list("ijD", atoms, self.r_cut)
        m = i != j
        pairs = (i[m], j[m], D[m], D[m])
        f = self._describe_one(atoms, None, pairs, fixed)
        return f if self._level == "per_atom" else f.mean(axis=0)

    @property
    def dim(self) -> int:
        n = len(self.species) if self.species else 1
        return n + self.n_max * n + self.n_max ** 2 * self.l_max * n ** 2

    @property
    def level(self) -> str:
        return self._level


# --------------------------------------------------------------------------
# User-facing descriptor factory: SOAP and ACE only
#   (LatentDescriptor is kept, but it is the model-latent descriptor used in the paper,
#    needs mace-torch and is an advanced option, not in the default list.)
# --------------------------------------------------------------------------
def make_descriptor(name: str = "soap", **kwargs) -> Descriptor:
    """Construct a descriptor by name.

    name:
        "soap"    -> SOAPDescriptor (dscribe; smooth overlap of atomic positions,
                     rotation invariant)
        "behler"  -> BehlerDescriptor (pure-numpy Behler–Parrinello G2/G4 symmetry functions)
        "latent"  -> LatentDescriptor (**the one used in the paper**: rotation-invariant node
                     features of MACE-MP-0 small, 256 dims -> PCA 16; needs a tap function,
                     built in examples/index/build_index.py and
                     examples/eval/compare_scorings.py)
        "ace"     -> ACEDescriptor (the lightweight linear-ACE invariants shipped with this
                     library, pure numpy)

    Common kwargs:
        soap  : r_cut=5.0, nmax=6, lmax=4, species=None, average=False
        behler: r_cut=5.0, radial=(...), angular=(...)
        latent: fn=<callable>, dim=256, level="per_atom"
        ace   : r_cut=5.0, n_max=5, l_max=3, species=None, level="per_atom"
    """
    key = name.lower()
    if key == "soap":
        return SOAPDescriptor(**kwargs)
    if key == "behler":
        return BehlerDescriptor(**kwargs)
    if key == "latent":
        return LatentDescriptor(**kwargs)
    if key == "ace":
        return ACEDescriptor(**kwargs)
    raise ValueError(f"unknown descriptor {name!r}; available: "
                     f"{', '.join(AVAILABLE_DESCRIPTORS)}")


def mace_latent_descriptor(model: str = "small", device: str = "cpu",
                           dtype: str = "float64", dim: int = 256,
                           level: str = "per_atom") -> Descriptor:
    """The descriptor used in the paper: rotation-invariant node features of MACE-MP-0 small
    (256 dims).

    Requires `pip install mace-torch`; the model weights are downloaded automatically on first
    use.
    """
    from mace.calculators import mace_mp
    calc = mace_mp(model=model, device=device, default_dtype=dtype)
    return LatentDescriptor(
        lambda atoms: calc.get_descriptors(atoms, invariants_only=True),
        dim=dim, level=level)


AVAILABLE_DESCRIPTORS = ("soap", "behler", "latent", "ace")
