#!/usr/bin/env python
"""Verify that the indexes on the Hub are byte-for-byte the ones on disk.

Downloads each file once (through the Hub cache) and compares its sha256 with the local
copy. Run this after upload_indexes.py, ideally from a different machine than the one
that uploaded: it is the only check that proves the 17.5 GiB arrived intact.

    python verify_upload.py --base /data/cbe-indexes
    python verify_upload.py --base /data/cbe-indexes --sha-only   # skip downloads, compare
                                                                  # against CHECKSUMS.json

Exit code 0 means every file matches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "Niacin64/cbe-indexes"
FILES = [
    ("omat24_index/shard_000.faiss",              "omat24/shard_000.faiss"),
    ("omat24_index/shard_001.faiss",              "omat24/shard_001.faiss"),
    ("omat24_index/shard_002.faiss",              "omat24/shard_002.faiss"),
    ("omat24_index/shard_003.faiss",              "omat24/shard_003.faiss"),
    ("omat24_index/shard_004.faiss",              "omat24/shard_004.faiss"),
    ("CBE/examples/mptrj_index_all/index.faiss",  "mptrj/index.faiss"),
    ("alexandria_index_full/shard_000.faiss",     "alexandria/shard_000.faiss"),
    ("alexandria_index_full/shard_001.faiss",     "alexandria/shard_001.faiss"),
    ("alexandria_index_full/shard_002.faiss",     "alexandria/shard_002.faiss"),
    ("alexandria_index_full/shard_003.faiss",     "alexandria/shard_003.faiss"),
]


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--sha-only", action="store_true",
                    help="do not download; compare local files against CHECKSUMS.json")
    args = ap.parse_args()

    api = HfApi()
    remote = {f.rfilename: f.size
              for f in api.list_repo_tree(args.repo, repo_type="dataset", recursive=True)
              if hasattr(f, "size") and f.size}

    manifest = {}
    checksums = args.base / "CHECKSUMS.json"
    if checksums.is_file():
        doc = json.loads(checksums.read_text())
        for corpus, entry in doc.get("indexes", {}).items():
            for item in entry.get("binaries", []):
                manifest[f"{corpus}/{item['file']}"] = item.get("sha256")

    ok = bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for src_rel, dst in FILES:
            src = args.base / src_rel
            if not src.is_file():
                print(f"missing locally: {src}")
                continue
            size = src.stat().st_size
            if remote.get(dst) != size:
                print(f"MISMATCH on the Hub: {dst} ({remote.get(dst)} vs {size} bytes)")
                bad += 1
                continue
            local_hash = sha256(src)
            if manifest.get(dst) not in (None, "TO_FILL", local_hash):
                print(f"local file does not match CHECKSUMS.json: {dst}")
                bad += 1
                continue
            if args.sha_only:
                print(f"ok (local sha256 matches the manifest): {dst}")
                ok += 1
                continue
            t0 = time.time()
            got = hf_hub_download(repo_id=args.repo, filename=dst, repo_type="dataset",
                                  local_dir=tmp)
            same = sha256(Path(got)) == local_hash
            print(f"{'ok  ' if same else 'FAIL'} {dst:28s} {size / 2**30:6.3f} GiB "
                  f"({time.time() - t0:.0f}s)")
            ok += same
            bad += (not same)
            os.remove(got)

    print(f"\n{ok} verified, {bad} problematic (of {len(FILES)})")
    print("CHECKSUMS.json can be used as published" if bad == 0
          else "fix the files listed above before announcing the dataset")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
