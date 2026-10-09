# -*- coding: utf-8 -*-
"""Local: build a faiss index from GB10 features (features_chunk_*.npy or a single features.npy).

Two modes are supported:
  - single index: all features go into one index.faiss (use when they fit in local memory)
  - sharded: --shard-size sets the atoms per shard (default 40M ~ 2.5GB), written as shard_{i}.faiss
    queries go shard by shard, merging and re-sorting the per-shard kNNs (index too big for local RAM)

Usage (local grace environment):
  PYTHONPATH=CBE python make_index_from_npy.py \
      --features alexandria_index_full/ --out alexandria_index_full/ \
      --shard-size 40000000
"""
import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cbe.index import FaissIndex
from cbe.kde import estimate_bandwidth_sampled


def find_chunks(features_path):
    p = Path(features_path)
    if p.is_dir():
        chunks = sorted(p.glob("features_chunk_*.npy"))
        if chunks:
            return chunks
        if (p / "features.npy").exists():
            return [p / "features.npy"]
        raise FileNotFoundError(f"{p} contains no features.npy / features_chunk_*.npy")
    return [p]


def build_shards(chunk_files, out, shard_size, dim):
    shards = []
    buf = []
    n = 0
    sid = 0
    total = 0
    for cf in chunk_files:
        X = np.load(cf).astype(np.float32)
        buf.append(X)
        n += len(X)
        while n >= shard_size:
            Xs = np.concatenate(buf, axis=0)
            keep, rest = Xs[:shard_size], Xs[shard_size:]
            idx = faiss.IndexFlatL2(dim)
            idx.add(keep)
            faiss.write_index(idx, str(out / f"shard_{sid:03d}.faiss"))
            shards.append((sid, shard_size))
            print(f"  shard_{sid:03d}.faiss written ({shard_size} atoms)", flush=True)
            total += shard_size
            sid += 1
            buf = [rest] if len(rest) else []
            n = len(rest)
    if buf and n > 0:
        Xs = np.concatenate(buf, axis=0)
        idx = faiss.IndexFlatL2(dim)
        idx.add(Xs)
        faiss.write_index(idx, str(out / f"shard_{sid:03d}.faiss"))
        shards.append((sid, int(len(Xs))))
        print(f"  shard_{sid:03d}.faiss written ({len(Xs)} atoms)", flush=True)
        total += len(Xs)
    return shards, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="features.npy, or the directory containing it (with chunks)")
    ap.add_argument("--pca", default=None, help="pca.npz (optional, only copied to out)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--kde-k", type=int, default=30)
    ap.add_argument("--bw-sample", type=int, default=20000)
    ap.add_argument("--shard-size", type=int, default=None,
                    help="atoms per shard (default None = single index; 40M ~ 2.5GB per shard)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    chunks = find_chunks(args.features)
    dim = np.load(chunks[0], mmap_mode="r").shape[1]
    total_atoms = sum(np.load(c, mmap_mode="r").shape[0] for c in chunks)
    print(f"{len(chunks)} chunks, {total_atoms} atoms x {dim} dims", flush=True)

    if args.shard_size:
        shards, total = build_shards(chunks, out, args.shard_size, dim)
        index_type = "flat_l2_sharded"
    else:
        index = faiss.IndexFlatL2(dim)
        for cf in chunks:
            X = np.load(cf).astype(np.float32)
            index.add(X)
        faiss.write_index(index, str(out / "index.faiss"))
        shards = [(0, int(index.ntotal))]
        total = int(index.ntotal)
        index_type = "flat_l2_exact"
        print(f"index.faiss written ({total} vectors)", flush=True)

    # bandwidth: estimated from samples across the shard indexes (search every shard, merge distances)
    wrapped = FaissIndex()
    rng = np.random.default_rng(0)
    n_sample = min(args.bw_sample, total)
    # sample points are spread evenly over the shards
    h_std, h_mean = 0.0, 0.0
    sample_per_shard = max(1, n_sample // len(shards))
    dists = []
    for sid, n_shard in shards:
        idx = faiss.read_index(str(out / f"shard_{sid:03d}.faiss" if args.shard_size else out / "index.faiss"))
        w = FaissIndex()
        w._index = idx
        w._n = int(idx.ntotal)
        ids = rng.choice(w.n, size=min(sample_per_shard, w.n), replace=False)
        Xs = np.vstack([np.asarray(w.reconstruct(int(i)), dtype=np.float32) for i in ids])
        D, _ = idx.search(Xs, min(args.kde_k + 1, w.n))
        if D.shape[1] > 1:
            D = D[:, 1:]  # drop self
        dists.append(D)
    if dists:
        all_d = np.concatenate(dists, axis=0)
        h_std = float(np.std(all_d))
        h_mean = float(np.mean(all_d))
    print(f"bandwidth h_std={h_std:.4f} h_mean={h_mean:.4f} (k={args.kde_k})", flush=True)

    meta = {"dim": dim, "n_atoms": total, "descriptor": "mace-latent",
            "pca_dim": dim, "mode": "alexandria", "index_type": index_type,
            "bandwidth": h_std, "bandwidth_mean": h_mean, "bandwidth_k": args.kde_k,
            "bandwidth_mode": "std", "n_shards": len(shards),
            "shard_atoms": [n for _, n in shards]}
    with open(out / "meta.json", "w") as fh:
        json.dump(meta, fh)
    if args.pca and Path(args.pca).exists():
        src = Path(args.pca).resolve()
        dst = (out / "pca.npz").resolve()
        if src != dst:
            import shutil
            shutil.copy(src, dst)
    print(f"done: {total} atoms in {len(shards)} shards -> {out}/", flush=True)


if __name__ == "__main__":
    main()
