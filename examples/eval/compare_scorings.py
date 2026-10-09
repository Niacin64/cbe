# -*- coding: utf-8 -*-
"""Compare the distributions of several scorings on the same three groups (in-dist / perturbed / random).

Panels (2x4):
  row 1 (kNN / kernel density family): KDE coverage | exp coverage | distance-k1 | distance-k30
  row 2 (classical density / OOD family, from broader ML):
      Mahalanobis | GMM density | Local Outlier Factor | One-Class SVM

Semantics: row 1 (KDE/exp) is coverage ρ∈[0,1] (larger = covered); others are risk (larger = extrapolated).
Row-2 scorers are fitted on a subsample of the index (PCA 16-d space), scoring per atom, max per structure.

Usage (local grace environment):
  PYTHONPATH=CBE python compare_scorings.py \
      --index-dir mptrj_index_all/ \
      --in-dist test_set/test_in_dist.xyz \
      --perturbed refs_perturbed.xyz \
      --random refs_random_new.xyz \
      --model mace-mp-0-small --out compare_scorings_8.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cbe.descriptors import LatentDescriptor
from cbe.coverage import CoverageModel
from cbe.data import read_structures
from cbe.eval import load_calculator
from cbe.index import FaissIndex
from cbe.kde import estimate_bandwidth_sampled
from cbe.preprocess import PCA
from cbe.scorings import make_scorer


def load_bandwidth(index_dir, k, mode):
    """Prefer meta.json (Silverman > h_std / h_mean); otherwise None."""
    meta = Path(index_dir) / "meta.json"
    if not meta.exists():
        return None
    m = json.loads(meta.read_text())
    if mode == "std" and m.get("bandwidth_k") == k:
        for key in ("bandwidth_silverman", "bandwidth"):
            if m.get(key) is not None:
                return float(m[key])
    if mode == "mean" and m.get("bandwidth_k") == k:
        if m.get("bandwidth_mean") is not None:
            return float(m["bandwidth_mean"])
    return None


def sample_from_index(index, n, seed=0):
    """Uniformly sample n indexed vectors from the faiss index (PCA space)."""
    rng = np.random.default_rng(seed)
    n = min(int(n), int(index.n))
    idxs = rng.choice(int(index.n), size=n, replace=False)
    return np.vstack([np.asarray(index.reconstruct(int(i)), dtype=np.float32)
                      for i in idxs])


def pca_features(pca, descriptor, structs):
    """Per-atom PCA features for each structure [(n_atoms, 16), ...]."""
    return [pca.transform(np.asarray(descriptor.describe(a), dtype=np.float64))
            for a in structs]


def density_scores(scorer, feats_list, agg="max"):
    """Per-atom risk -> structure level (max = weakest atom = most extrapolative)."""
    out = []
    for F in feats_list:
        r = np.asarray(scorer.score(F))
        out.append(float(np.max(r)) if agg == "max" else float(np.mean(r)))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--in-dist", required=True)
    ap.add_argument("--perturbed", required=True)
    ap.add_argument("--random", required=True)
    ap.add_argument("--model", default="mace-mp-0-small")
    ap.add_argument("--out", default="compare_scorings_8.png")
    ap.add_argument("--bandwidth", type=float, default=None,
                    help="KDE bandwidth h; if None, use the Silverman value from meta (recommended)")
    ap.add_argument("--n-train", type=int, default=20000,
                    help="number of training points sampled for the row-2 scorers (drawn from the 49M index)")
    args = ap.parse_args()

    model = load_calculator(args.model)
    descriptor = LatentDescriptor(
        lambda a: model.get_descriptors(a, invariants_only=True), dim=256,
        level="per_atom")

    # load the index + PCA only once (avoids reloading the 3GB index for every panel)
    index = FaissIndex().load(str(Path(args.index_dir) / "index.faiss"))
    pca = PCA.load(str(Path(args.index_dir) / "pca.npz"))

    groups = {
        "in-dist": read_structures(args.in_dist),
        "perturbed": read_structures(args.perturbed),
        "random": read_structures(args.random),
    }

    # row-2 scorers: sample a training subset and fit once
    print(f"sampling {args.n_train} points from the index and fitting four density scorers ...", flush=True)
    X_train = sample_from_index(index, args.n_train)
    scorers = {
        "Mahalanobis": make_scorer("mahalanobis").fit(X_train),
        "GMM density": make_scorer("gmm", n_components=16).fit(X_train),
        "LOF": make_scorer("lof", n_neighbors=20).fit(X_train),
        "One-Class SVM": make_scorer(
            "oneclasssvm").fit(X_train[: min(5000, len(X_train))]),
    }

    # per-structure PCA features (used by row 2)
    feats = {g: pca_features(pca, descriptor, structs)
             for g, structs in groups.items()}

    # panel configs: (title, kind, scoring function structs->scores)
    # kind: "coverage" -> linear on x in [0,1]; "risk" -> linear (range from percentiles)
    def cov_cfg(k, scoring, agg, coverage_agg, bandwidth=None, bw_mode="std"):
        cov = CoverageModel(descriptor, index=index, k=k, scoring=scoring,
                            agg=agg, coverage_agg=coverage_agg,
                            bandwidth=bandwidth, bandwidth_mode=bw_mode)
        cov._pca = pca
        cov._per_atom = True
        return cov

    # KDE / exp bandwidths
    h_kde = args.bandwidth or load_bandwidth(args.index_dir, 100, "std")
    h_exp = load_bandwidth(args.index_dir, 100, "mean")
    if h_kde is None:
        h_kde = estimate_bandwidth_sampled(index, k=100, n_sample=5000, mode="std")
    if h_exp is None:
        h_exp = estimate_bandwidth_sampled(index, k=100, n_sample=5000, mode="mean")

    kde = cov_cfg(100, "kde", "max", "min", h_kde)
    exp = cov_cfg(100, "exp", "max", "min", h_exp)
    k1 = cov_cfg(1, "distance", "max", "min")
    k30 = cov_cfg(30, "distance", "max", "min")

    # per-group scores (precomputed, avoiding lambda closure ambiguity)
    results = {
        f"KDE ρ (h={h_kde:.3f})": {g: kde.coverage_many(s) for g, s in groups.items()},
        f"exp ρ (s={h_exp:.3f})": {g: exp.coverage_many(s) for g, s in groups.items()},
        "distance-k1": {g: k1.score_many(s) for g, s in groups.items()},
        "distance-k30": {g: k30.score_many(s) for g, s in groups.items()},
    }
    # (display title, scorer name) pairs, keeping the results and configs keys in sync
    density_panels = [
        ("Mahalanobis", "Mahalanobis"),
        ("GMM density (−log p)", "GMM density"),
        ("Local Outlier Factor", "LOF"),
        ("One-Class SVM (−decision)", "One-Class SVM"),
    ]
    for disp, key in density_panels:
        results[disp] = {g: density_scores(scorers[key], feats[g]) for g in groups}

    configs = [
        (f"KDE ρ (h={h_kde:.3f})", "coverage"),
        (f"exp ρ (s={h_exp:.3f})", "coverage"),
        ("distance-k1", "risk"),
        ("distance-k30", "risk"),
    ] + [(disp, "risk") for disp, _ in density_panels]

    colors = {"in-dist": "#1a4d8f", "perturbed": "#e08a00", "random": "#2e7d32"}
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    for ax, (title, kind) in zip(axes.ravel(), configs):
        scores = results[title]
        if kind == "coverage":
            lo, hi = 0.0, 1.0
            for g in groups:
                ax.hist(scores[g], bins=30, range=(lo, hi), density=True,
                        alpha=0.45, color=colors[g],
                        label=f"{g} (mean={scores[g].mean():.3f})")
            ax.set_xlabel("coverage ρ (1=covered)")
        else:
            pooled = np.concatenate([scores[g] for g in groups])
            lo = float(np.percentile(pooled, 0.5))
            hi = float(np.percentile(pooled, 99.5))
            for g in groups:
                ax.hist(scores[g], bins=30, range=(lo, hi), density=True,
                        alpha=0.45, color=colors[g],
                        label=f"{g} (mean={scores[g].mean():.3g})")
            ax.set_xlabel("risk (higher = more extrapolative; range = 0.5-99.5 percentile)")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25, axis="y")

    fig.suptitle(f"Distributions of eight scorings on the same three groups | {index.n/1e6:.0f}M MPtrj index"
                 f" (exact FlatL2); row-2 scorers are fitted on a {args.n_train}-point training subset")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}\n")

    print(f"{'scoring':22s} {'in-dist':>11s} {'perturbed':>11s} {'random':>11s}")
    for title, scores in results.items():
        print(f"{title:22s} {scores['in-dist'].mean():>11.4g} "
              f"{scores['perturbed'].mean():>11.4g} {scores['random'].mean():>11.4g}")
    print("\nNote: KDE/exp are coverage (higher = covered); the rest is risk (higher = extrapolative).")


if __name__ == "__main__":
    main()
