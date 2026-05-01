"""ECE heatmap — models × datasets, one panel per modality.

Produces a 1x3 figure: (a) text, (b) image, (c) image_text.
Rows = models, cols = datasets. Cell = raw micro-ECE, annotated inside.
Shared colorbar. Defaults to verbalized source.

Usage:
    python ece_heatmap.py                 # verbalized
    python ece_heatmap.py --source consistency

Writes: calibration/results/ece_heatmap_<source>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from calibration_metrics import ece
from loader import CALIBRATION_LABELS, load_consistency, load_self_reported

MODELS = ["Qwen 4B", "Qwen 8B", "Gemma 3 4B", "GPT-4.1-mini"]
DATASETS = ["English", "Translated", "Ukrainian"]
MODALITIES = ["text", "image", "image_text"]
N_BINS = 10


def get_cell(source, model, dataset, modality):
    if source == "verbalized":
        return load_self_reported(model, dataset, modality)
    return load_consistency(model, dataset)


def pooled_ece(cell):
    c = np.concatenate([cell.class_conf[l] for l in CALIBRATION_LABELS])
    k = np.concatenate([cell.class_corr[l] for l in CALIBRATION_LABELS])
    return ece(c, k, n_bins=N_BINS)


def build_grid(source, modality):
    g = np.full((len(MODELS), len(DATASETS)), np.nan)
    for i, m in enumerate(MODELS):
        for j, d in enumerate(DATASETS):
            cell = get_cell(source, m, d, modality)
            if cell is None:
                continue
            g[i, j] = pooled_ece(cell)
    return g


def draw_panel(ax, grid, title, tag, vmin, vmax):
    im = ax.imshow(grid, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS, fontsize=10)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=10)
    ax.set_title(title, fontsize=11)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isnan(v):
                txt = "—"
                color = "gray"
            else:
                txt = f"{v:.3f}"
                color = "white" if v > (vmin + vmax) / 2 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color=color, fontweight="bold")
    ax.text(0.5, -0.14, f"({tag}) modality: {title}",
            transform=ax.transAxes, ha="center", fontsize=10, fontweight="bold")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["verbalized", "consistency"], default="verbalized")
    args = ap.parse_args()

    modalities = MODALITIES if args.source == "verbalized" else ["image_text"]
    grids = [(md, build_grid(args.source, md)) for md in modalities]

    all_vals = np.concatenate([g.ravel() for _, g in grids])
    finite = all_vals[np.isfinite(all_vals)]
    if finite.size == 0:
        raise RuntimeError("no finite ECE values — caches missing?")
    vmin, vmax = 0.0, float(np.nanmax(finite))

    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2))
    if n == 1:
        axes = [axes]
    tags = ["a", "b", "c"]
    last_im = None
    for ax, (md, grid), tag in zip(axes, grids, tags):
        last_im = draw_panel(ax, grid, md, tag, vmin, vmax)

    fig.suptitle(f"Raw micro-ECE — {args.source}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 0.94, 1])
    cbar_ax = fig.add_axes([0.955, 0.15, 0.012, 0.7])
    fig.colorbar(last_im, cax=cbar_ax, label="ECE")

    out = HERE / "results" / f"ece_heatmap_{args.source}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
