# -*- coding: utf-8 -*-
"""Shared configuration for the three main-text figures: the 8 representative models +
corpus groups + dropped cathode points."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# The 8 representative models: (key, display name, corpus group)
MODELS8 = [
    ("mace-mp-0-medium", "MACE-MP-0 medium", "MPtrj"),
    ("sevennet-0",       "SevenNet-0",       "MPtrj"),
    ("orb-v2-mptrj",     "ORB v2 MPtrj",     "MPtrj"),
    ("mace-mpa-0",       "MACE-MPA-0",       "MPtrj + Alexandria"),
    ("orb-v2",           "ORB v2 MPA",       "MPtrj + Alexandria"),
    ("sevennet-mf-ompa", "SevenNet-MF-ompa", "MPtrj + Alexandria + OMat24"),
    ("mattersim-v1",     "MatterSim-v1",     "MPtrj + Alexandria + OMat24"),
    ("grace-2l-oam",     "GRACE-2L-OAM",     "MPtrj + Alexandria + OMat24"),
]

GROUP_ORDER = ["MPtrj", "MPtrj + Alexandria", "MPtrj + Alexandria + OMat24"]
GROUP_COLOR = {"MPtrj": "#1a4d8f", "MPtrj + Alexandria": "#c0392b",
               "MPtrj + Alexandria + OMat24": "#2e7d32"}

# Cathode: drop layered LiFeO2 (idx 8) -- this layered phase does not exist experimentally,
# it is an artefact of the protocol
CATHODE_DROP = [8]

TCOLORS = {"layered": "#1a4d8f", "spinel": "#c0392b", "olivine": "#2e7d32"}
GCOLORS = {"in-dist": "#1a4d8f", "perturbed": "#e08a00", "random": "#2e7d32"}


def corpus_group(corpus: str) -> str:
    """Map matbench's fine-grained corpus names onto the three groups."""
    if corpus == "MPtrj":
        return "MPtrj"
    if "OMat" in corpus:
        return "MPtrj + Alexandria + OMat24"
    return "MPtrj + Alexandria"

# ---------------------------------------------------------------------------
# Data file resolution: data lives in <repo>/DATA/, but scripts may still just
# give the file name. The old layout (files at the repository root) is supported
# too: look in DATA/ first and fall back to the root if nothing is found.
# ---------------------------------------------------------------------------
DATA = ROOT / "DATA"


def data(name):
    """Return the data file path: DATA/<name> preferred, <repo root>/<name> as fallback."""
    p = DATA / name
    return p if p.exists() else ROOT / name


def data_glob(pattern):
    """Glob inside DATA/; fall back to the repository root when there are no hits."""
    hits = sorted(DATA.glob(pattern))
    return hits if hits else sorted(ROOT.glob(pattern))
