"""Confidence distribution — verbalized, image_text.

3 subplots = 3 datasets. Per subplot: grouped bars per confidence bin
(width 0.1, [0,1]), one bar per model. Y-axis = % of all per-label
confidence values in that bin (multi-label: 22 labels × n samples).

Usage:
    python confidence_distribution.py

Writes: calibration/results/confidence_distribution_image_text.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from loader import CALIBRATION_LABELS, load_self_reported

MODELS = ["Qwen 4B", "Qwen 8B", "Gemma 3 4B", "GPT-4.1-mini"]
DATASETS = ["English", "Translated", "Ukrainian"]
MODALITY = "image_text"
N_BINS = 10
PALETTE = ["#4C72B0", "#DD8452", "#55A467", "#C44E52"]


def pooled_conf(cell):
    return np.concatenate([cell.class_conf[l] for l in CALIBRATION_LABELS])


def hist_pct(conf):
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    counts, _ = np.histogram(conf, bins=edges)
    if counts.sum() == 0:
        return counts.astype(float)
    return counts / counts.sum() * 100.0


def bin_labels():
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    return [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(N_BINS)]


def draw_subplot(ax, dataset):
    x = np.arange(N_BINS)
    n_models = len(MODELS)
    width = 0.8 / n_models
    n_per_model = []
    for i, m in enumerate(MODELS):
        cell = load_self_reported(m, dataset, MODALITY)
        if cell is None:
            n_per_model.append(0)
            continue
        conf = pooled_conf(cell)
        pct = hist_pct(conf)
        offs = (i - (n_models - 1) / 2) * width
        ax.bar(x + offs, pct, width * 0.95, color=PALETTE[i],
               label=f"{m} (n={cell.n})", edgecolor="white", linewidth=0.4)
        n_per_model.append(cell.n)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels(), fontsize=8, rotation=35, ha="right")
    ax.set_xlabel("Confidence bin")
    ax.set_ylabel("% of per-label confidences")
    ax.set_title(dataset, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper center", fontsize=8, framealpha=0.9)


def main():
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6.2 * len(DATASETS), 5.0), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        draw_subplot(ax, ds)
    fig.tight_layout()
    out = HERE / "results" / f"confidence_distribution_{MODALITY}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
