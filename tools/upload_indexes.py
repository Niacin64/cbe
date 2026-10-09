#!/usr/bin/env python
"""Upload the ten CBE index binaries to the Hugging Face dataset.

Run this on a machine with a fast, direct connection to huggingface.co. It is fully
resumable: files already present with the right size are skipped, so re-running after
an interruption continues where it stopped.

    pip install "huggingface_hub[hf_transfer]"
    export HF_HUB_ENABLE_HF_TRANSFER=1
    hf auth login                     # or: export HF_TOKEN=hf_xxx
    python upload_indexes.py --base /data/cbe-indexes

Layout on the Hub (the download script in the repository expects exactly this):

    omat24/shard_00{0..4}.faiss
    mptrj/index.faiss
    alexandria/shard_00{0..3}.faiss

The pca.npz and meta.json files are NOT uploaded: they live in the git repository.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO = "Niacin64/cbe-indexes"

# (local relative path under --base, path on the Hub)
FILES = [
    ("omat24_index/shard_000.faiss",                    "omat24/shard_000.faiss"),
    ("omat24_index/shard_001.faiss",                    "omat24/shard_001.faiss"),
    ("omat24_index/shard_002.faiss",                    "omat24/shard_002.faiss"),
    ("omat24_index/shard_003.faiss",                    "omat24/shard_003.faiss"),
    ("omat24_index/shard_004.faiss",                    "omat24/shard_004.faiss"),
    ("CBE/examples/mptrj_index_all/index.faiss",        "mptrj/index.faiss"),
    ("alexandria_index_full/shard_000.faiss",           "alexandria/shard_000.faiss"),
    ("alexandria_index_full/shard_001.faiss",           "alexandria/shard_001.faiss"),
    ("alexandria_index_full/shard_002.faiss",           "alexandria/shard_002.faiss"),
    ("alexandria_index_full/shard_003.faiss",           "alexandria/shard_003.faiss"),
]


def remote_sizes(api: HfApi, repo: str) -> dict:
    return {f.rfilename: f.size
            for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True)
            if hasattr(f, "size") and f.size}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, type=Path,
                    help="directory holding omat24_index/, alexandria_index_full/ and CBE/")
    ap.add_argument("--repo", default=REPO, help=f"Hugging Face dataset repo (default {REPO})")
    ap.add_argument("--retries", type=int, default=20, help="attempts per file")
    ap.add_argument("--only", default=None,
                    help="upload only files whose Hub path contains this substring")
    args = ap.parse_args()

    api = HfApi()
    try:
        print("logged in as:", api.whoami().get("name"))
    except Exception as exc:
        print(f"not logged in ({str(exc)[:80]}). Run `hf auth login` first.", file=sys.stderr)
        return 2

    api.create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True)
    remote = remote_sizes(api, args.repo)

    todo = [(s, d) for s, d in FILES if args.only is None or args.only in d]
    upload = 0
    for src_rel, dst in todo:
        src = args.base / src_rel
        if not src.is_file():
            print(f"skip (not found): {src}")
            continue
        size = src.stat().st_size
        if remote.get(dst) == size:
            print(f"ok (already on the Hub): {dst}")
            continue
        for attempt in range(1, args.retries + 1):
            t0 = time.time()
            try:
                print(f"uploading [{attempt}/{args.retries}] {dst}  ({size / 2**30:.2f} GiB)",
                      flush=True)
                api.upload_file(path_or_fileobj=str(src), path_in_repo=dst,
                                repo_id=args.repo, repo_type="dataset")
                break
            except Exception as exc:
                print(f"  failed: {str(exc)[:160]}", flush=True)
                if attempt == args.retries:
                    print(f"giving up on {dst}", file=sys.stderr)
                    return 1
                time.sleep(min(30 * attempt, 300))
        got = remote_sizes(api, args.repo).get(dst)
        if got != size:
            print(f"size mismatch after upload: {dst} ({got} vs {size})", file=sys.stderr)
            return 1
        rate = size / max(time.time() - t0, 1e-9) / 2**20
        print(f"done: {dst}  ({time.time() - t0:.0f}s, {rate:.2f} MiB/s)", flush=True)
        upload += 1

    print(f"\nuploaded {upload} file(s)")
    print(f"repository: https://huggingface.co/datasets/{args.repo}")
    print("\nNext: verify the bytes round trip from a machine with a fast connection:")
    print("    python verify_upload.py --base <same base>     # sha256 of every file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
