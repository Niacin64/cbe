#!/usr/bin/env python
"""Download a pretrained CBE coverage index.

The index binaries (6-10 GB each) are too large for the git repository, so they are
distributed separately. This script fetches one corpus at a time, resumes interrupted
transfers, verifies the sha256 of every file against ``pretrained/CHECKSUMS.json`` and
then runs a short functional check on the index it just downloaded.

    python examples/index/download_pretrained.py --corpus mptrj --dest ~/cbe_data
    python examples/index/download_pretrained.py --corpus omat24 --dest ~/cbe_data
    python examples/index/download_pretrained.py --corpus alexandria --dest ~/cbe_data

You only need the corpus (or corpora) your models were trained on: the index is matched
to the training set, so an MPtrj-trained model uses the MPtrj index. Downloading all
three is only necessary to reproduce the corpus comparison of the paper.

Environment
-----------
``HF_REPO``   Hugging Face dataset repository holding the index binaries, e.g.
              ``yourname/cbe-indexes``. Defaults to the value in this file.
              Requires ``pip install huggingface_hub`` (or the ``hf`` CLI on PATH).

Notes
-----
* Downloads go to ``<dest>/<corpus>/`` and are resumable: re-running the script after an
  interruption continues where it stopped instead of starting over.
* ``--skip-verify`` removes the hashing pass (17.5 GB of sha256 takes a few minutes).
* The FP32 float16-free exact indexes are reproducible byte-for-byte only if the corpus
  snapshot, the descriptor checkpoint and the faiss version match, so treat the checksums
  as the authority for *this* distribution rather than for any rebuild.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HF_REPO = os.environ.get("HF_REPO", "<account>/cbe-indexes")
HERE = Path(__file__).resolve().parent
PRETRAINED = HERE.parent.parent / "pretrained"
CHUNK = 1 << 22


def sha256(path: Path, quiet: bool = False) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with path.open("rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            h.update(block)
            done += len(block)
            if not quiet and total > (1 << 30):
                pct = 100.0 * done / total
                print(f"\r    hashing {path.name}: {pct:5.1f}%", end="", flush=True)
    if not quiet and total > (1 << 30):
        print()
    return h.hexdigest()


def manifest() -> dict:
    path = PRETRAINED / "CHECKSUMS.json"
    if not path.exists():
        sys.exit(f"error: {path} not found; run this script from a clone of the repository")
    return json.loads(path.read_text())


def hf_download(repo: str, remote: str, local: Path) -> None:
    """Fetch one file from a Hugging Face dataset repository."""
    if local.exists() and local.stat().st_size > 0:
        print(f"    {local.name}: already present "
              f"({local.stat().st_size / 2**30:.2f} GiB), skipping download")
        return
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        cmd = ["hf", "download", repo, remote, "--repo-type", "dataset",
               "--local-dir", str(local.parent)]
        print(f"    $ {' '.join(cmd)}")
        rc = subprocess.call(cmd)
        if rc != 0:
            sys.exit("error: both huggingface_hub and the `hf` CLI failed; "
                     "install one with `pip install huggingface_hub`")
        return
    print(f"    fetching {remote} from {repo} ...")
    hf_hub_download(repo_id=repo, filename=remote, repo_type="dataset",
                    local_dir=str(local.parent))


def functional_check(corpus: str, dest: Path, pca_dim: int) -> int:
    """Load what was downloaded and confirm it behaves like a coverage index."""
    import numpy as np

    meta = json.loads((PRETRAINED / "indexes" / corpus / "meta.json").read_text())
    pca = np.load(PRETRAINED / "indexes" / corpus / "pca.npz")
    mean, comps = pca["mean"], pca["components"]
    assert mean.shape == (256,) and comps.shape == (pca_dim, 256), "pca.npz has the wrong shape"

    files = sorted(dest.glob("*.faiss"))
    if not files:
        sys.exit(f"error: no .faiss files under {dest}")
    print(f"    {len(files)} index file(s) found")

    try:
        import faiss  # type: ignore
    except ImportError:
        print("    faiss not installed: skipping the index load check "
              "(`pip install faiss-cpu`)")
        return 0

    total = 0
    for path in files:
        index = faiss.read_index(str(path))
        if index.d != pca_dim:
            sys.exit(f"error: {path.name} has dimension {index.d}, expected {pca_dim}")
        total += index.ntotal
    expected = meta["n_environments"]
    print(f"    {total:,} environments indexed (meta.json says {expected:,})")
    if total != expected:
        sys.exit("error: environment count does not match meta.json; the download is "
                 "incomplete or the wrong corpus snapshot")
    print("    OK: dimensions and environment count match meta.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, choices=["mptrj", "alexandria", "omat24"])
    ap.add_argument("--dest", required=True, type=Path,
                    help="directory that will hold <corpus>/ (created if missing)")
    ap.add_argument("--repo", default=HF_REPO,
                    help=f"Hugging Face dataset repo (default: {HF_REPO})")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the sha256 pass (faster, but does not prove integrity)")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the functional check on the downloaded index")
    args = ap.parse_args()

    man = manifest()["indexes"].get(args.corpus)
    if man is None:
        sys.exit(f"error: no manifest entry for corpus '{args.corpus}'")
    meta = json.loads((PRETRAINED / "indexes" / args.corpus / "meta.json").read_text())
    target = args.dest / args.corpus
    target.mkdir(parents=True, exist_ok=True)

    print(f"[{args.corpus}] {meta['note']}")
    print(f"[{args.corpus}] destination {target}")

    # the pca matrix ships with the repository and must be next to the index
    pca_src = PRETRAINED / "indexes" / args.corpus / "pca.npz"
    (target / "pca.npz").write_bytes(pca_src.read_bytes())
    (target / "meta.json").write_bytes((PRETRAINED / "indexes" / args.corpus
                                        / "meta.json").read_bytes())

    total = sum(b["bytes"] for b in man["binaries"] if b.get("bytes"))
    print(f"[{args.corpus}] {len(man['binaries'])} binary file(s), "
          f"{total / 2**30:.2f} GiB to download")

    for entry in man["binaries"]:
        name = entry["file"]
        hf_download(args.repo, f"{args.corpus}/{name}", target / name)
        got = (target / name).stat().st_size
        if entry.get("bytes") and got != entry["bytes"]:
            sys.exit(f"error: {name} is {got} bytes, expected {entry['bytes']}; "
                     "delete it and re-run")

    if not args.skip_verify:
        print(f"[{args.corpus}] verifying sha256 ...")
        for entry in man["binaries"]:
            name, want = entry["file"], entry.get("sha256", "TO_FILL")
            if want in (None, "", "TO_FILL"):
                print(f"    {name}: no checksum published yet, skipping")
                continue
            got = sha256(target / name)
            if got != want:
                sys.exit(f"error: {name} failed the checksum:\n"
                         f"    expected {want}\n    got      {got}\n"
                         "Delete the file and re-run to download it again.")
            print(f"    {name}: OK")

    if not args.no_check:
        print(f"[{args.corpus}] functional check ...")
        functional_check(args.corpus, target, meta.get("pca_dim", 16))

    print(f"\n[{args.corpus}] ready: {target}")
    print("Verify it, then point the coverage model at it:")
    print(f"    python examples/index/verify_index.py --index {target}")
    print(f"    cov = pretrained.load_index('{args.corpus}')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
