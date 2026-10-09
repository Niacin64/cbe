# -*- coding: utf-8 -*-
"""Corpus-matched coverage: MPtrj-only models use the MPtrj index, Alexandria models use both.

Key point: a model's coverage must be measured against **its own training corpus**.
  MPtrj-only              ρ = ρ_MPtrj(h)
  MPtrj + Alexandria family  ρ = max(ρ_MPtrj(h), ρ_Alexandria(h))
Both indexes share the same kernel width h (otherwise the union is polluted by differing widths; this
file defaults to h=0.2184, the Silverman bandwidth of the MPtrj index in the paper).
(OMat24 has no index yet, so OAM-family coverage is only approximate as MPtrj + Alexandria.)

Usage:
  from coverage_matched import load_dual, matched_coverage, CORPUS_INDEXES
  dual = load_dual("dist_test.npz")
  cov = matched_coverage(dual, corpus="MPtrj + Alex", h=0.2184)   # (n_structures,)
"""
from pathlib import Path

import numpy as np

H_DEFAULT = 0.2184          # Silverman bandwidth of the MPtrj index (backward-compatible default)

# The three corpus indexes **share** the same kernel width h=0.2184.
# Rationale (measured; per-structure min-ρ over the 623 test structures):
#   uniform h=0.2184   : in-dist 0.87/0.83/0.69 (mp/alex/omat), random 0.00/0.00/0.01  <- best separation
#   per-index Silverman: omat's 1.17 is clearly too large (in-dist 0.99 / random 0.82, only 1.2x apart)
#   raw NN distances are comparable across indexes (in-dist 0.058/0.068/0.107, random 0.478/0.362/0.299),
#   so a shared width is reasonable and comparable; per-index Silverman values are reference only.
INDEX_BANDWIDTH = {"mp": 0.2184, "alex": 0.2184, "omat": 0.2184}

# corpus -> which indexes to use ("mp"=MPtrj, "alex"=Alexandria)
CORPUS_INDEXES = {
    "MPtrj": ("mp",),
    "MPtrj + sAlex": ("mp", "alex"),
    "MPtrj + Alex": ("mp", "alex"),
    "MPtrj + Alex + OMat24": ("mp", "alex", "omat"),
    "MPtrj + OMat24 + sAlex": ("mp", "alex", "omat"),
    "MPtrj + MDR-MP PBE ω_q": ("mp",),
}

# index alias -> key in the distance cache / local directory name
INDEX_ALIAS = {"mp": "mptrj", "alex": "alexandria", "omat": "omat24"}


def indexes_for(corpus: str):
    if corpus in CORPUS_INDEXES:
        return CORPUS_INDEXES[corpus]
    # fallback: if the corpus name contains Alex/sAlex, include the Alexandria index
    return ("mp", "alex") if ("lex" in corpus) else ("mp",)


def load_dual(npz_path, keep=None):
    """Load a distance cache (any number of indexes: D_mp / D_alex / D_omat ...).

    keep: optional structure-level bool mask (len = n_cached) to pick the physically filtered subset.
    """
    d = np.load(npz_path)
    bnd = d["bnd"]
    tags = [k[2:] for k in d.files if k.startswith("D_")]
    if keep is None:
        return {"bnd": bnd, **{f"D_{t}": d[f"D_{t}"] for t in tags}}
    keep = np.asarray(keep)
    cnt = np.diff(bnd)
    atom_mask = np.repeat(keep, cnt)
    return {"bnd": np.concatenate([[0], np.cumsum(cnt[keep])]),
            **{f"D_{t}": d[f"D_{t}"][atom_mask] for t in tags}}


def slice_dual(dual, mask):
    """Slice an already-loaded dual by a **structure-level** mask (len = current structure count)."""
    mask = np.asarray(mask)
    cnt = np.diff(dual["bnd"])
    atom_mask = np.repeat(mask, cnt)
    out = {"bnd": np.concatenate([[0], np.cumsum(cnt[mask])])}
    for key in dual:
        if key.startswith("D_"):
            out[key] = dual[key][atom_mask]
    return out


def per_index_coverage(dual, tag, h=None, k=None):
    """Structure-level coverage (min over the per-atom KDE values).

    h=None (default) uses that index's own Silverman bandwidth; passing h explicitly overrides it.
    k=None uses all cached neighbours.
    """
    if h is None:
        h = INDEX_BANDWIDTH.get(tag, H_DEFAULT)   # all three indexes share 0.2184 (see the note above)
    bnd = dual["bnd"]
    if f"D_{tag}" not in dual:
        raise KeyError(f"no D_{tag} in the cache (available: {[k[2:] for k in dual if k.startswith('D_')]})")
    D = dual[f"D_{tag}"]
    Dk = D if k is None else D[:, :k]
    pa = np.mean(np.exp(-(Dk ** 2) / (2 * h ** 2)), axis=1)
    return np.array([pa[a:b].min() for a, b in zip(bnd[:-1], bnd[1:])])


def matched_coverage(dual, corpus, h=None, k=None):
    """Structure-level coverage for a corpus (max over corpora per structure)."""
    covs = [per_index_coverage(dual, tag, h=h, k=k) for tag in indexes_for(corpus)]
    return np.max(np.stack(covs, 0), axis=0) if len(covs) > 1 else covs[0]
