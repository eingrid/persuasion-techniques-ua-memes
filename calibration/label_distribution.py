"""Ground-truth label distribution across the three test sets.

Reads test.jsonl for each dataset, counts per-label sample frequency
(multi-label: one sample can contribute to several labels). NO_PROPAGANDA
is an implicit label — samples with an empty `labels` list.

Outputs:
    calibration/results/label_distribution.csv
    calibration/results/label_distribution.png  (horizontal bars, 3 subplots)

Usage:
    python label_distribution.py
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from loader import UNIQUE_LABELS

DATASETS = {
    "English":    ROOT / "datasets" / "propaganda_950" / "annotations" / "test.jsonl",
    "Translated": ROOT / "datasets" / "translated"     / "annotations" / "test.jsonl",
    "Ukrainian":  ROOT / "datasets" / "ukrainian"      / "annotations" / "test.jsonl",
}


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_labels(rows):
    counts = {l: 0 for l in UNIQUE_LABELS}
    for r in rows:
        labels = r.get("labels") or []
        if not labels:
            counts["NO_PROPAGANDA"] += 1
            continue
        for l in labels:
            if l in counts:
                counts[l] += 1
    return counts


def is_propaganda(labels):
    if not labels:
        return False
    return any(l != "NO_PROPAGANDA" for l in labels)


def _plot(df, totals, labels, pct_col_suffix, out_png):
    order_key = df.loc[labels][[f"{n}{pct_col_suffix}" for n in DATASETS]].max(axis=1)
    order = order_key.sort_values(ascending=True).index.tolist()
    df_plot = df.loc[order]
    y = np.arange(len(order))

    dataset_names = list(DATASETS.keys())
    palette = {"English": "#4C72B0", "Translated": "#DD8452", "Ukrainian": "#55A467"}
    bar_h = 0.8 / len(dataset_names)

    fig, ax = plt.subplots(figsize=(9, 10))
    for i, name in enumerate(dataset_names):
        col = f"{name}{pct_col_suffix}"
        vals = df_plot[col].values
        offs = (i - (len(dataset_names) - 1) / 2) * bar_h
        ax.barh(y + offs, vals, height=bar_h * 0.92, color=palette[name],
                label=f"{name} (n={totals[name]})",
                edgecolor="white", linewidth=0.4)
        xmax = df_plot[col].max() if df_plot[col].max() > 0 else 1.0
        for yi, v in zip(y + offs, vals):
            if v > 0:
                ax.text(v + xmax * 0.01, yi, f"{v:.1f}", va="center", fontsize=7)

    ax.set_yticks(y)
    wrapped = [textwrap.fill(l, width=22, break_long_words=False) for l in order]
    ax.set_yticklabels(wrapped, fontsize=9)
    ax.set_ylim(-0.5, len(order) - 0.5)
    ax.set_xlabel("% of dataset")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main():
    stats = {}
    totals = {}
    prop_totals = {}  # samples with ≥1 propaganda label
    for name, path in DATASETS.items():
        rows = load_jsonl(path)
        stats[name] = count_labels(rows)
        totals[name] = len(rows)
        prop_totals[name] = sum(1 for r in rows if is_propaganda(r.get("labels") or []))

    df = pd.DataFrame(stats).reindex(UNIQUE_LABELS)
    for name in DATASETS:
        df[f"{name}_pct"]      = df[name] / totals[name]      * 100
        df[f"{name}_pct_prop"] = df[name] / prop_totals[name] * 100

    out_dir = HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "label_distribution.csv")
    print(f"Wrote {out_dir}/label_distribution.csv")
    print(f"n      : English={totals['English']}  Translated={totals['Translated']}  Ukrainian={totals['Ukrainian']}")
    print(f"n_prop : English={prop_totals['English']}  Translated={prop_totals['Translated']}  Ukrainian={prop_totals['Ukrainian']}")

    _plot(df, totals, UNIQUE_LABELS, "_pct",
          out_dir / "label_distribution.png")

    prop_labels = [l for l in UNIQUE_LABELS if l != "NO_PROPAGANDA"]
    _plot(df, prop_totals, prop_labels, "_pct_prop",
          out_dir / "label_distribution_no_np.png")


if __name__ == "__main__":
    main()
