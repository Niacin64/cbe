# -*- coding: utf-8 -*-
"""Paper flow diagram (Fig 1): corpus -> descriptor -> PCA -> index; candidate structure ->
one query -> ρ; calibration (corpus samples + difficult structures) -> conformal θ ->
warn/trust.

Layout: two pipelines stacked on top plus calibration/decision at the bottom, canvas
17.0 x 6.2 cm, sized for a full-width \\textwidth column.
Outputs fig_pipeline.pdf / .png.
Usage:
  python fig_pipeline.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.size": 15})

ROOT = Path(__file__).resolve().parent.parent.parent
BLUE, RED, GREEN, GREY = "#1a4d8f", "#c0392b", "#2e7d32", "#6b7280"


def box(ax, x, y, w, h, text, fc, ec=None, fs=15, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.06",
                                linewidth=1.6, facecolor=fc, edgecolor=ec or fc, alpha=0.97))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(ax, x0, y0, x1, y1, color=GREY, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=17,
                                 linewidth=1.7, color=color, shrinkA=1, shrinkB=1,
                                 linestyle=ls))


def main():
    W, H = 17.0, 6.2
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    bh = 1.30                                    # box height

    # ---------------- top row: built once ----------------
    y1 = 4.30
    ax.text(0.15, y1 + bh + 0.16, "build once", fontsize=16, color=BLUE, weight="bold")
    box(ax, 0.15, y1, 3.5, bh, "training corpus\nMPtrj / Alexandria / OMat24", "#dbe7f5", BLUE, fs=14)
    arrow(ax, 3.72, y1 + bh / 2, 4.28, y1 + bh / 2)
    box(ax, 4.35, y1, 3.2, bh, "descriptor\nMACE-MP-0 latent,\n256 components", "#dbe7f5", BLUE, fs=14)
    arrow(ax, 7.62, y1 + bh / 2, 8.18, y1 + bh / 2)
    box(ax, 8.25, y1, 2.0, bh, "PCA\n16 components", "#dbe7f5", BLUE, fs=14)
    arrow(ax, 10.32, y1 + bh / 2, 10.88, y1 + bh / 2)
    box(ax, 10.95, y1, 5.9, bh,
        "coverage index\n$k$-NN, exact $L_2$ (sharded if > RAM)", "#dbe7f5", BLUE, fs=14)

    # ---------------- bottom row: screening one candidate structure ----------------
    y2 = 2.35
    ax.text(0.15, y2 + bh + 0.16, "screen any candidate (one query per structure)",
            fontsize=16, color=GREEN, weight="bold")
    box(ax, 0.15, y2, 3.5, bh, "candidate structure\nany composition, any cell", "#dcefdc", GREEN, fs=14)
    arrow(ax, 3.72, y2 + bh / 2, 4.28, y2 + bh / 2, GREEN)
    box(ax, 4.35, y2, 3.2, bh, "same descriptor\n+ PCA projection", "#dcefdc", GREEN, fs=14)
    arrow(ax, 7.62, y2 + bh / 2, 8.18, y2 + bh / 2, GREEN)
    box(ax, 8.25, y2, 2.0, bh, "nearest-\nneighbour query", "#dcefdc", GREEN, fs=13)
    arrow(ax, 10.32, y2 + bh / 2, 10.88, y2 + bh / 2, GREEN)
    box(ax, 10.95, y2, 5.9, bh,
        "coverage $\\rho$ per atom\nstructure score $c=\\min_a\\rho_a$", "#dcefdc", GREEN, fs=14)

    # the index is reused (top row -> bottom row, dashed)
    arrow(ax, 13.9, y1 - 0.03, 13.9, y2 + bh + 0.03, BLUE, ls=(0, (4, 3)))
    ax.text(14.15, (y1 + y2 + bh) / 2 - 0.10, "reused for every model", fontsize=12, color=BLUE)

    # ---------------- bottom: calibration and decision ----------------
    y3 = 0.35
    box(ax, 4.35, y3, 4.7, 1.26,
        "calibration sample: corpus + difficult structures\n"
        "$\\Rightarrow$ conformal, $\\alpha=0.1$, $\\varepsilon_f=0.5$ eV/Å",
        "#f7dfdc", RED, fs=13.5)
    arrow(ax, 9.12, y3 + 0.63, 10.28, y3 + 0.63, RED)
    box(ax, 10.35, y3, 6.5, 1.26,
        "one threshold $\\theta$ per model\n$\\rho<\\theta$: warn    |    $\\rho\\geq\\theta$: trust",
        "#f2f2f2", GREY, fs=13.5)

    fig.subplots_adjust(left=0.004, right=0.996, top=0.995, bottom=0.005)
    fig.savefig(ROOT / "DOC/figures/fig_pipeline.pdf")
    fig.savefig(ROOT / "DOC/figures/fig_pipeline.png", dpi=200)
    print(f"saved -> {ROOT}/DOC/figures/fig_pipeline.pdf (+ .png)")


if __name__ == "__main__":
    main()
