# -*- coding: utf-8 -*-
"""Unified roster evaluation: any model on the {test, cathode} sets.

Roster (keys match matbench-discovery's model_key; corpora taken from its models/*.yml):
  MACE family / CHGNet / AlphaNet -> grace environment
  SevenNet / ORB / MatterSim / GRACE(OAM) -> mlip7 environment

Usage:
  # test set (623 structures after physical filtering)
  PYTHONPATH=CBE python eval_roster.py \
      --set test --models mace-mp-0-small mace-mp-0-medium ... --out eval_roster_test_grace.pkl
  # cathode set (75 candidates, DFT single-point reference)
  PYTHONPATH=CBE python eval_roster.py \
      --set cathode --models ... --out eval_roster_cathode_grace.pkl
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent.parent

# key -> (display name, training corpus, backend)
ROSTER = {
    # ---- MPtrj only ----
    "mace-mp-0-small":   ("MACE-MP-0 small",   "MPtrj", "mace", "small"),
    "mace-mp-0-medium":  ("MACE-MP-0 medium",  "MPtrj", "mace", "medium"),
    "mace-mp-0-large":   ("MACE-MP-0 large",   "MPtrj", "mace", "large"),
    "mace-mp-0b2-medium": ("MACE-MP-0b2 medium", "MPtrj", "mace", "medium-0b2"),
    "mace-mp-0b3-medium": ("MACE-MP-0b3 medium", "MPtrj", "mace", "medium-0b3"),
    "chgnet":            ("CHGNet",            "MPtrj", "chgnet", None),
    "sevennet-0":        ("SevenNet-0",        "MPtrj", "sevennet", "7net-0"),
    "sevennet-l3i5":     ("SevenNet-l3i5",     "MPtrj", "sevennet", "7net-l3i5"),
    "orb-v2-mptrj":      ("ORB v2 MPtrj",      "MPtrj", "orb", "orb_mptraj_only_v2"),
    "grace-2l-mptrj":    ("GRACE-2L-MPtrj",    "MPtrj", "grace-local", "MP_GRACE_2L_r6_11Nov2024"),
    "alphanet-v1-mptrj": ("AlphaNet-v1-MPtrj", "MPtrj", "alphanet", "alphanet-mptrj.model"),
    "esen-30m-mp":       ("eSEN-30M-MP",       "MPtrj", "esen", "esen_30m_mp"),
    # ---- MPtrj + Alexandria (sAlex / Alex / OAM) ----
    "mace-mpa-0":        ("MACE-MPA-0",        "MPtrj + sAlex", "mace", "medium-mpa-0"),
    "sevennet-mf-ompa":  ("SevenNet-MF-ompa",  "MPtrj + OMat24 + sAlex", "sevennet", ("7net-mf-ompa", "mpa")),
    "orb-v2":            ("ORB v2 MPA",        "MPtrj + Alex", "orb", "orb_v2"),
    "orb-v3":            ("ORB v3",            "MPtrj + Alex + OMat24", "orb", "orb_v3_conservative_20_mpa"),
    "mattersim-v1":      ("MatterSim-v1",      "MPtrj + Alex + OMat24", "mattersim", None),
    "grace-1l-oam":      ("GRACE-1L-OAM",      "MPtrj + OMat24 + sAlex", "grace", "GRACE-1L-OAM"),
    "grace-2l-oam":      ("GRACE-2L-OAM",      "MPtrj + OMat24 + sAlex", "grace", "GRACE-2L-OAM"),
    "grace-fs-oam":      ("GRACE-FS-OAM",      "MPtrj + OMat24 + sAlex", "grace", "GRACE-FS-OAM"),
    "alphanet-v1-oam":   ("AlphaNet-v1-OAM",   "MPtrj + OMat24 + sAlex", "alphanet", "alphanet-oam.model"),
    "esen-30m-oam":      ("eSEN-30M-OAM",      "MPtrj + OMat24 + sAlex", "esen", "esen_30m_oam"),
}


def load_model(key, device="cpu"):
    name, corpus, backend, arg = ROSTER[key]
    if backend == "mace":
        from mace.calculators import mace_mp
        local = ROOT / ".cache/mace_foundations" / f"mace-{arg}.model"
        return mace_mp(model=str(local) if local.exists() else arg,
                       device=device, default_dtype="float64")
    if backend == "alphanet":
        from mace.calculators import MACECalculator
        p = ROOT / ".cache/alphanet" / arg
        if not p.exists():
            raise FileNotFoundError(f"missing AlphaNet checkpoint: {p}")
        return MACECalculator(model_paths=[str(p)], device=device, default_dtype="float64")
    if backend == "chgnet":
        from cbe.eval import load_calculator
        return load_calculator("chgnet", device=device)
    if backend == "sevennet":
        from sevenn.calculator import SevenNetCalculator
        if isinstance(arg, tuple):      # multi-modal checkpoint: (model, modal)
            return SevenNetCalculator(model=arg[0], modal=arg[1], device=device)
        return SevenNetCalculator(model=arg, device=device)
    if backend == "orb":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        return ORBCalculator(model=getattr(pretrained, arg)(device=device), device=device)
    if backend == "mattersim":
        from mattersim.forcefield import MatterSimCalculator
        ck = ROOT / ".cache/mattersim/mattersim-v1.0.0-1M.pth"
        return MatterSimCalculator(device=device, load_path=str(ck))
    if backend == "grace":
        from tensorpotential.calculator import grace_fm
        return grace_fm(arg)
    if backend == "grace-local":
        from tensorpotential.calculator import TPCalculator
        return TPCalculator(model=str(ROOT / ".cache/grace" / arg))
    if backend == "esen":
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        predictor = pretrained_mlip.get_predict_unit(arg, device=device)
        return FAIRChemCalculator(predictor, task_name="omat")
    raise ValueError(backend)


def load_set(which):
    """Return (structures, list of ref_forces or None, tag)."""
    from ase.io import read
    if which == "test":
        refs = read(str(data("refs_full.xyz")), index=":") + read(str(data("refs_ood.xyz")), index=":")
        keep = np.load(data("test_keep.npy"))
        refs = [a for a, k in zip(refs, keep) if k]
        return refs, [np.asarray(a.arrays["REF_forces"], float) for a in refs], "test(623)"
    if which == "calib":
        refs = read(str(data("refs_calib.xyz")), index=":")
        return refs, [np.asarray(a.arrays["REF_forces"], float) for a in refs], "calib(540)"
    if which == "cathode":
        refs = read(str(ROOT / "case_cathode/relaxed_all.xyz"), index=":")[:75]
        ref_f = []
        for i in range(len(refs)):
            d = ROOT / "project" / f"{i:04d}"
            if not (d / "vasprun.xml").exists():
                raise FileNotFoundError(f"missing DFT results: {d}")
            from cbe.vasp import read_vasp_results
            _a, _e, f = read_vasp_results(d)
            ref_f.append(np.asarray(f, float))
        return refs, ref_f, "cathode(75)"
    raise ValueError(which)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["test", "cathode", "calib"])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--roster-out", default=str(data("roster_meta.pkl")))
    args = ap.parse_args()

    refs, ref_forces, tag = load_set(args.set)
    print(f"{tag}: {len(refs)} structures", flush=True)

    out = {}
    for key in args.models:
        if key not in ROSTER:
            print(f"[skip] unknown key: {key}", flush=True)
            continue
        name, corpus, backend, _ = ROSTER[key]
        try:
            calc = load_model(key, args.device)
        except Exception as e:
            print(f"[skip] {name}: {type(e).__name__}: {str(e)[:90]}", flush=True)
            continue
        errs = []
        for i, a in enumerate(refs):
            am = a.copy()
            am.calc = calc
            f = np.asarray(am.get_forces(), float)
            errs.append(float(np.mean(np.linalg.norm(f - ref_forces[i], axis=1))))
            if (i + 1) % 200 == 0:
                print(f"  {name}: {i+1}/{len(refs)}", flush=True)
        out[key] = np.array(errs)
        print(f"  {name:20s} [{corpus:24s}] median force error {np.median(errs)*1000:8.1f} meV/A", flush=True)

    with open(args.out, "wb") as fh:
        pickle.dump(out, fh)
    # also save the roster metadata (display name / corpus) for plotting
    meta = {k: {"name": ROSTER[k][0], "corpus": ROSTER[k][1]} for k in ROSTER}
    with open(args.roster_out, "wb") as fh:
        pickle.dump(meta, fh)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
