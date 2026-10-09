# -*- coding: utf-8 -*-
"""Descriptor tests: factory, shapes, rotation/translation/permutation invariance."""
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from cbe import make_descriptor, AVAILABLE_DESCRIPTORS


def _kws(name):
    if name == "soap":
        return {"nmax": 4, "lmax": 2, "species": [29]}
    if name == "ace":
        return {"n_max": 4, "l_max": 2, "species": [29]}
    if name == "behler":
        return {}
    return {}


def _latent():
    """MACE latent descriptor (the main one used in the paper); skip if mace-torch is absent."""
    pytest.importorskip("mace")
    from cbe import mace_latent_descriptor
    return mace_latent_descriptor()


@pytest.mark.parametrize("name", AVAILABLE_DESCRIPTORS)
def test_factory_and_shapes(name):
    a = bulk("Cu", "fcc", a=3.6).repeat((2, 2, 2))
    d = _latent() if name == "latent" else make_descriptor(name, r_cut=4.0, **_kws(name))
    f = d.describe(a)
    assert f.shape == (len(a), d.dim)
    assert np.isfinite(f).all()


@pytest.mark.parametrize("name", AVAILABLE_DESCRIPTORS)
def test_rotation_translation_invariance(name):
    a = bulk("Cu", "fcc", a=3.6).repeat((2, 2, 2))
    d = _latent() if name == "latent" else make_descriptor(name, r_cut=4.0, **_kws(name))
    f0 = d.describe(a)
    b = a.copy()
    b.rotate(37, "z", rotate_cell=True)
    b.positions += 1.7
    f1 = d.describe(b)
    assert np.abs(f0 - f1).max() < 1e-8


@pytest.mark.parametrize("name", AVAILABLE_DESCRIPTORS)
def test_permutation_equivariance(name):
    a = bulk("Cu", "fcc", a=3.6).repeat((2, 2, 2))
    d = _latent() if name == "latent" else make_descriptor(name, r_cut=4.0, **_kws(name))
    f0 = d.describe(a)
    perm = np.random.default_rng(0).permutation(len(a))
    f1 = d.describe(a[perm])
    assert np.abs(np.sort(f0, axis=0) - np.sort(f1, axis=0)).max() < 1e-8


def test_ace_multi_element():
    rng = np.random.default_rng(1)
    a = Atoms("Cu2O3", positions=rng.random((5, 3)) * 4,
              cell=np.eye(3) * 8, pbc=True)
    d = make_descriptor("ace", r_cut=4.0, n_max=3, l_max=2, species=[8, 29])
    f = d.describe(a)
    assert f.shape == (5, d.dim)
    assert np.isfinite(f).all()


def test_unknown_descriptor():
    with pytest.raises(ValueError):
        make_descriptor("nope")
