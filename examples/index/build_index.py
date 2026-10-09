# -*- coding: utf-8 -*-
"""Build a full coverage index for MACE-MP-0 (relative to the MPtrj training set).

Full pipeline (recommended on a cluster GPU):
  - stream through the MPtrj JSON (never load the whole file into memory);
  - per-atom descriptor for each structure (MACE-MP-0 latent invariant descriptor, 256-dim; or SOAP);
  - (optional) fit PCA on a sample (16 dims by default, as in Willimetz, saving 16x storage);
  - incrementally add to a Faiss index (exact FlatL2, avoiding HNSW approximate-search recall issues);
  - save index + PCA + metadata (including the KDE bandwidth) for CoverageModel to load.

Usage:
  # take the most relaxed structure per material (~146k structures, light)
  python build_index.py --json MPtrj.json --mode relaxed --out mptrj_index/

  # all ionic steps (~1.58M structures, closest to the MACE-MP-0 training distribution, heavy)
  python build_index.py --json MPtrj.json --mode all --out mptrj_index/ --device cuda

  # quick self-check on a small sample
  python build_index.py --json MPtrj.json --max-structures 500 --out /tmp/idx/
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import ijson

# so that `import cbe` finds the cbe package one level up (the parent of examples/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---- structure parsing ----
def atoms_from_task(task):
    from ase import Atoms
    st = task["structure"]
    matrix = np.array(st["lattice"]["matrix"])
    sites = st["sites"]
    symbols = [s["species"][0]["element"] for s in sites]
    if all("xyz" in s for s in sites):
        return Atoms(symbols, positions=[s["xyz"] for s in sites],
                     cell=matrix, pbc=True)
    return Atoms(symbols, scaled_positions=[s["abc"] for s in sites],
                 cell=matrix, pbc=True)


def pick_relaxed(tasks):
    best_t, best_e = None, np.inf
    for tid, t in tasks.items():
        e = t.get("energy_per_atom", t.get("e_per_atom_relaxed", np.inf))
        if e is None:
            e = np.inf
        if e < best_e:
            best_e, best_t = e, t
    return best_t


def iter_structures(json_path, mode="all", max_structures=None):
    n = 0
    with open(json_path, "rb") as f:
        for mid, tasks in ijson.kvitems(f, ""):
            if mode == "relaxed":
                try:
                    yield atoms_from_task(pick_relaxed(tasks))
                    n += 1
                except Exception:
                    continue
            else:
                for tid, t in tasks.items():
                    try:
                        yield atoms_from_task(t)
                        n += 1
                    except Exception:
                        continue
            if max_structures and n >= max_structures:
                return
            if n % 20000 == 0 and n > 0:
                print(f"  processed {n} structures", flush=True)


# ---- descriptors ----
def get_descriptor_fn(name, device):
    if name == "mace-latent":
        from mace.calculators import mace_mp
        calc = mace_mp(model="small", device=device, default_dtype="float64")

        def fn(atoms):
            return np.asarray(calc.get_descriptors(atoms, invariants_only=True),
                              dtype=np.float32)
        return fn, 256
    raise ValueError(f"unknown descriptor: {name} (mace-latent is currently recommended)")


def build_index(json_path, mode, out_dir, descriptor="mace-latent",
                pca_dim=16, max_structures=None, pca_sample=200000, device="cpu",
                kde_k=100, bw_sample=200000):
    import faiss

    desc_fn, dim = get_descriptor_fn(descriptor, device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- pass 1: sample and fit PCA ----
    pca = None
    if pca_dim and pca_dim < dim:
        from cbe.preprocess import PCA
        sample = []
        n_sample = 0
        print("pass 1: sampling to fit PCA ...", flush=True)
        for atoms in iter_structures(json_path, mode, max_structures):
            f = desc_fn(atoms)
            sample.append(f)
            n_sample += len(f)
            if n_sample >= pca_sample:
                break
        X = np.concatenate(sample, axis=0)
        pca = PCA(pca_dim).fit(X)
        print(f"  PCA fitted: {dim} -> {pca_dim} ({n_sample} atoms sampled)", flush=True)

    # ---- pass 2: compute descriptors + build the index ----
    d = pca_dim if pca else dim
    # Exact FlatL2: 16 dims, 46M atoms = ~2.9GB, batched GPU/CPU queries take
    # seconds; HNSW misses exact self-matches and reports false "misplacements".
    index = faiss.IndexFlatL2(d)
    n_atoms = 0
    print(f"pass 2: building the exact Flat index ...", flush=True)
    for atoms in iter_structures(json_path, mode, max_structures):
        f = desc_fn(atoms)
        if pca:
            f = pca.transform(f)
        index.add(f.astype(np.float32))
        n_atoms += len(f)
        if n_atoms % 2000000 < 100:
            print(f"    indexed {n_atoms} atoms", flush=True)

    # ---- estimate the KDE bandwidth h (on the training set alone, independent of queries) ----
    # h = std of k-NN distances among training points. Sample bw_sample training
    # descriptors (index.reconstruct returns PCA'd vectors), take std of kde_k+1 NNs minus self.
    rng = np.random.default_rng(0)
    n_bw = min(bw_sample, int(index.ntotal))
    idxs = rng.choice(int(index.ntotal), n_bw, replace=False)
    Xb = np.vstack([np.asarray(index.reconstruct(int(i)), dtype=np.float32)
                    for i in idxs])
    if hasattr(index, "hnsw"):   # raise efSearch to >= k before HNSW queries (otherwise low recall)
        index.hnsw.efSearch = max(index.hnsw.efSearch, kde_k + 1, 64)
    D2, _ = index.search(Xb, kde_k + 1)
    Db = np.sqrt(np.maximum(D2[:, 1:], 0.0))   # drop self + sqrt (faiss returns squared distances)
    h = float(np.std(Db))
    print(f"  KDE bandwidth h_std = {h:.4f} (k={kde_k}, {n_bw} training points sampled)", flush=True)

    # ---- save ----
    faiss.write_index(index, str(out / "index.faiss"))
    meta = {"dim": d, "n_atoms": n_atoms, "descriptor": descriptor,
            "pca_dim": pca_dim, "mode": mode,
            "bandwidth": h, "bandwidth_k": kde_k, "bandwidth_mode": "std"}
    if pca:
        np.savez(out / "pca.npz", mean=pca.mean_, components=pca.components_)
    with open(out / "meta.json", "w") as fh:
        json.dump(meta, fh)
    print(f"\nindex complete: {n_atoms} atoms -> {out}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--mode", choices=["all", "relaxed"], default="relaxed")
    ap.add_argument("--out", default="mptrj_index")
    ap.add_argument("--descriptor", default="mace-latent")
    ap.add_argument("--pca-dim", type=int, default=16)
    ap.add_argument("--max-structures", type=int, default=None)
    ap.add_argument("--pca-sample", type=int, default=200000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kde-k", type=int, default=100, help="neighbours k for the KDE bandwidth")
    ap.add_argument("--bw-sample", type=int, default=200000, help="training points sampled for the bandwidth")
    args = ap.parse_args()
    build_index(args.json, args.mode, args.out, descriptor=args.descriptor,
                pca_dim=args.pca_dim, max_structures=args.max_structures,
                pca_sample=args.pca_sample, device=args.device,
                kde_k=args.kde_k, bw_sample=args.bw_sample)


if __name__ == "__main__":
    main()
