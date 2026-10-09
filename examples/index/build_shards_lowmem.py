# -*- coding: utf-8 -*-
"""Low-memory sharded index build (for the 159M-atom Alexandria set).

Unlike make_index_from_npy.py, this does **not** concatenate the 20 chunks into one big array
(that would cost an extra 2.5 GB peak and get OOM-killed on a 15 GB machine);
instead each chunk is added to the faiss index in turn, peaking at ~one shard (2.56 GB) + one chunk (128 MB).

Supports resuming: --start-atom skips the atoms already covered by existing shards.

Usage:
  PYTHONPATH=CBE python build_shards_lowmem.py \
      --features alexandria_index_full/ --out alexandria_index_full/ \
      --shard-size 40000000 --start-atom 40000000
"""
import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-size", type=int, default=40_000_000)
    ap.add_argument("--start-atom", type=int, default=0,
                    help="skip the first N atoms (already covered by built shards); resume from atom N")
    ap.add_argument("--first-shard-id", type=int, default=None,
                    help="starting shard id (default: inferred from start-atom/shard-size)")
    ap.add_argument("--kde-k", type=int, default=30)
    ap.add_argument("--bw-sample", type=int, default=20000)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cbe.kde import estimate_bandwidth_sampled

    feat = Path(args.features)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chunks = sorted(feat.glob("features_chunk_*.npy"))
    assert chunks, f"no features_chunk_*.npy found under {feat}"

    dim = int(np.load(chunks[0], mmap_mode="r").shape[1])
    total = sum(int(np.load(c, mmap_mode="r").shape[0]) for c in chunks)
    sid = args.first_shard_id if args.first_shard_id is not None else args.start_atom // args.shard_size
    print(f"{len(chunks)} chunks / {total:,} atoms x {dim} dims; "
          f"starting at atom {args.start_atom:,}, next shard id {sid}", flush=True)

    idx = faiss.IndexFlatL2(dim)
    filled = 0          # atoms currently packed into the shard
    seen = 0            # atoms skipped so far

    def flush(next_sid):
        nonlocal idx, filled
        if idx.ntotal == 0:
            return next_sid
        p = out / f"shard_{next_sid:03d}.faiss"
        faiss.write_index(idx, str(p))
        print(f"  {p.name} written ({idx.ntotal:,} atoms, {p.stat().st_size/1e9:.2f} GB)", flush=True)
        idx = faiss.IndexFlatL2(dim)
        filled = 0
        return next_sid + 1

    for c in chunks:
        X = np.load(c).astype(np.float32)
        if seen + len(X) <= args.start_atom:      # whole chunk belongs to already-built shards
            seen += len(X)
            continue
        if seen < args.start_atom:                # partial skip
            X = X[args.start_atom - seen:]
            seen = args.start_atom
        pos = 0
        while pos < len(X):
            take = min(args.shard_size - filled, len(X) - pos)
            idx.add(np.ascontiguousarray(X[pos:pos + take]))
            pos += take
            filled += take
            seen += take
            if filled == args.shard_size:
                sid = flush(sid)
    sid = flush(sid)

    shards = sorted(out.glob("shard_*.faiss"))
    n_shard = sum(faiss.read_index(str(p)).ntotal for p in shards)
    print(f"\n{len(shards)} shards in total, {n_shard:,} atoms (features total {total:,})")
    assert n_shard == total, "shard atom count does not match the features"

    # meta.json: fill in the bandwidth and shard info (used by CoverageModel.from_shards)
    metas = sorted(out.glob("meta*.json"))
    meta = json.loads(metas[0].read_text()) if metas else {}
    if "bandwidth" not in meta:
        # sample the first shard to estimate the Silverman bandwidth (as in make_index_from_npy.py)
        # Note: estimate_bandwidth_sampled takes the cbe Index wrapper (has .n), not a raw faiss index
        from cbe.index import FaissIndex
        tmp = FaissIndex().load(str(shards[0]))
        meta["bandwidth"] = float(estimate_bandwidth_sampled(
            tmp, k=args.kde_k, n_sample=args.bw_sample))
        del tmp
        print(f"Silverman bandwidth h = {meta['bandwidth']:.4f}")
    meta.update({"shards": [p.name for p in shards], "shard_size": args.shard_size,
                 "n_atoms_indexed": n_shard, "dim": dim})
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    print(f"meta.json updated -> {out/'meta.json'}")


if __name__ == "__main__":
    main()
