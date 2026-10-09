# -*- coding: utf-8 -*-
"""Out of the box: load a prebuilt coverage index + each model's calibrated threshold θ.

Directory layout (two subdirectories under `pretrained/`):
  pretrained/indexes/<corpus>/meta.json    index metadata (which files, how many environments, bandwidth)
  pretrained/indexes/<corpus>/*.faiss     index binaries (large, not tracked by git; see README)
  pretrained/thetas/thetas.json            per-model θ_f (+ calibration protocol and statistics)
  pretrained/thetas/thetas.csv             the same content as a table

Search order: $CBE_PRETRAINED -> <repo>/CBE/pretrained -> <repo>/pretrained -> ~/.cbe/pretrained

Typical usage:
    from cbe import pretrained
    cov = pretrained.load_index("mptrj")               # coverage model (with PCA)
    theta = pretrained.theta("mace-mpa-0")             # that model's calibrated threshold
    rho = cov.coverage_many(structures)
    flagged = [r < theta for r in rho]

    # or in one step
    pretrained.screen("mace-mpa-0", structures)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_BANDWIDTH = 0.2184          # Silverman bandwidth shared by the two indexes in the paper
DEFAULT_K = 30

_SEARCH = ("CBE_PRETRAINED",)


def _candidates() -> list[Path]:
    out = []
    env = os.environ.get(_SEARCH[0])
    if env:
        out.append(Path(env).expanduser())
    here = Path(__file__).resolve()
    for up in (here.parent.parent / "pretrained",      # <repo>/CBE/pretrained
               here.parent.parent.parent / "pretrained"):  # <repo>/pretrained
        out.append(up)
    out.append(Path.home() / ".cbe" / "pretrained")
    return out


def pretrained_dir(required: bool = True) -> Path | None:
    """Return the first pretrained directory that exists (containing indexes/ or thetas/)."""
    for d in _candidates():
        if (d / "thetas" / "thetas.json").exists() or (d / "indexes").is_dir():
            return d
    if required:
        raise FileNotFoundError(
            "pretrained/ directory not found. Set CBE_PRETRAINED to point at it, "
            "or place it in ~/.cbe/pretrained/ (see cbe/pretrained/README.md)")
    return None


# --------------------------------------------------------------------------
# θ registry
# --------------------------------------------------------------------------
def load_thetas() -> dict:
    """The whole θ registry (including the calibration protocol description)."""
    p = pretrained_dir() / "thetas" / "thetas.json"
    return json.loads(p.read_text())


def list_models() -> list[str]:
    return [m["model_key"] for m in load_thetas()["models"]]


def model_info(model: str) -> dict:
    """Full entry for one model (corpus, θ_f, held-out statistics)."""
    for m in load_thetas()["models"]:
        if m["model_key"] == model or m["model_name"] == model:
            return m
    raise KeyError(f"θ registry has no entry for {model!r}; available: {', '.join(list_models())}")


def theta(model: str) -> float:
    """That model's calibrated θ_f (a fixed value, used directly for flagging)."""
    return float(model_info(model)["theta_f"])


# --------------------------------------------------------------------------
# index registry
# --------------------------------------------------------------------------
def list_indexes() -> list[str]:
    d = pretrained_dir() / "indexes"
    return sorted(p.name for p in d.iterdir() if (p / "meta.json").exists())


_ALIAS = {"mp": "mptrj", "alex": "alexandria", "omat": "omat24"}


def resolve_index(name: str) -> str:
    """Map an internal short name (mp/alex/omat) to the registry name."""
    return _ALIAS.get(name, name)


def index_meta(name: str) -> dict:
    name = resolve_index(name)
    p = pretrained_dir() / "indexes" / name / "meta.json"
    if not p.exists():
        raise KeyError(f"no index {name!r}; available: {', '.join(list_indexes())}")
    return json.loads(p.read_text())


# Known local index locations (relative to the repository root), so users need not move files
_KNOWN_DIRS = {
    "mptrj": ["CBE/examples/mptrj_index_all", "mptrj_index_all"],
    "alexandria": ["alexandria_index_full", "CBE/alexandria_index_full"],
    "omat24": ["omat24_index", "CBE/omat24_index"],
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def index_dir(name: str, required_files: bool = True) -> Path:
    """Index directory; with required_files=True the binaries must already be in place.

    Search order: $CBE_INDEX_DIR/<name> -> pretrained/indexes/<name> -> known local locations.
    """
    name = resolve_index(name)
    cands = []
    env = os.environ.get("CBE_INDEX_DIR")
    if env:
        cands.append(Path(env).expanduser() / name)
        cands.append(Path(env).expanduser())
    cands.append(pretrained_dir() / "indexes" / name)
    cands += [_repo_root() / p for p in _KNOWN_DIRS.get(name, [])]
    d = next((c for c in cands if c.is_dir() and (list(c.glob("*.faiss"))
                                                  or (c / "index.faiss").exists())), None)
    if d is None:
        d = cands[0] if cands else pretrained_dir() / "indexes" / name
    if not d.exists():
        raise FileNotFoundError(f"index directory does not exist: {d}")
    if required_files and not (list(d.glob("*.faiss")) or (d / "index.faiss").exists()):
        raise FileNotFoundError(
            f"no *.faiss files in {d}. The index binaries are not tracked by git - see "
            f"cbe/pretrained/README.md (download or build them, then put them in this directory)")
    return d


def load_index(name: str, descriptor=None, k: int = DEFAULT_K, bandwidth: float | None = None,
               lazy: bool = True):
    """Load a prebuilt index -> CoverageModel.

    With descriptor=None the paper's MACE-MP-0 small latent descriptor is used by default
    (requires mace-torch); you may also pass make_descriptor("soap"/"ace"/"behler"), but it must
    match the one used to build the index, otherwise the feature spaces do not line up
    (meta.json records that mace-latent was used).
    """
    from .coverage import CoverageModel

    meta = index_meta(name)
    d = index_dir(name)
    if descriptor is None:
        from .descriptors import LatentDescriptor
        try:
            from mace.calculators import mace_mp
        except Exception as e:      # pragma: no cover
            raise ImportError("the default descriptor is the MACE-MP-0 small latent one, which "
                              "needs `pip install mace-torch`; or pass descriptor= yourself") from e
        calc = mace_mp(model="small", device="cpu", default_dtype="float64")
        descriptor = LatentDescriptor(
            lambda atoms: calc.get_descriptors(atoms, invariants_only=True),
            dim=256, level="per_atom")
    bw = bandwidth if bandwidth is not None else meta.get("bandwidth_used", DEFAULT_BANDWIDTH)
    if meta.get("layout") == "sharded":
        return CoverageModel.from_shards(d, d / "pca.npz", descriptor, k=k,
                                         scoring="kde", coverage_agg="min",
                                         bandwidth=bw, lazy=lazy)
    return CoverageModel.from_index(d / "index.faiss", d / "pca.npz", descriptor, k=k,
                                    scoring="kde", coverage_agg="min", bandwidth=bw)


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------
def screen(model: str, structures, index: str | None = None, cov=None):
    """Screen structures: return (coverage array, boolean array for below that model's θ_f, θ).

    structures: a single Atoms object or a list of them.
    With index=None the index is chosen from the model's training corpus (a single corpus uses
    that index; several corpora use the per-structure max).
    """
    import numpy as np

    info = model_info(model)
    single = hasattr(structures, "get_positions")
    structs = [structures] if single else list(structures)
    names = index and [index] or info["indexes"]
    covs = []
    for name in names:
        c = cov if (cov is not None and len(names) == 1) else load_index(name)
        covs.append(np.asarray(c.coverage_many(structs), dtype=float))
    rho = np.max(np.stack(covs, 0), axis=0) if len(covs) > 1 else covs[0]
    th = float(info["theta_f"])
    return (float(rho[0]) if single else rho), (rho < th), th
