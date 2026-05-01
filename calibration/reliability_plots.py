"""Reliability diagrams — grouped-bar comparison style.

1x3 figure with grouped bars per confidence bin. X-axis = 10 equal-width bins
on [0,1]. Y-axis = accuracy (%). Dashed line = perfect calibration (bin mid).
Legend: `<variant> (ECE: X.XX)`. Caption per subplot: `(a/b/c) Weighted ECE: X.XXX`
(mean of the variants' micro-ECEs).

Layouts:
  models_by_dataset  — subplots = datasets,   bars = 4 models      (source, modality)
  methods            — subplots = datasets,   bars = Raw/Platt/Temp (source, model, modality)
  sources            — subplots = datasets,   bars = Verbalized/Consistency (model, modality)
  all                — produce the full set for (verbalized, image_text)

Usage:
    python reliability_plots.py models_by_dataset verbalized image_text
    python reliability_plots.py methods verbalized "Qwen 8B" image_text
    python reliability_plots.py sources "Qwen 8B" image_text
    python reliability_plots.py all

Writes: calibration/results/reliability_<slug>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from calibration_metrics import apply_temperature, ece, find_optimal_temperature
from loader import CALIBRATION_LABELS, load_consistency, load_self_reported
from run_evaluation import N_BINS, build_pred_sets, ece_pred, platt_oof

BIN_LO, BIN_HI = 0.0, 1.0
N_VIS_BINS = N_BINS
PALETTE = ["#4C72B0", "#DD8452", "#55A467", "#C44E52", "#8172B3", "#937860"]
MODELS_DEFAULT = ["Qwen 4B", "Qwen 8B", "Gemma 3 4B", "GPT-4.1-mini"]
DATASETS_DEFAULT = ["English", "Translated", "Ukrainian"]
DATASET_DISPLAY = {"English": "Propaganda 950", "Translated": "PTM-UA", "Ukrainian": "UkrMeme"}
MODALITIES_DEFAULT = ["text", "image", "image_text"]

def get_cell(source, model, dataset, modality):
    if source == "verbalized":
        return load_self_reported(model, dataset, modality)
    if source == "consistency":
        return load_consistency(model, dataset)
    raise ValueError(f"source must be verbalized|consistency, got {source}")

def pool(cell):
    c = np.concatenate([cell.class_conf[l] for l in CALIBRATION_LABELS])
    k = np.concatenate([cell.class_corr[l] for l in CALIBRATION_LABELS])
    return c, k

def platt_pooled(cell):
    out_c = [platt_oof(cell.class_conf[l], cell.class_corr[l])[0] for l in CALIBRATION_LABELS]
    out_k = [cell.class_corr[l] for l in CALIBRATION_LABELS]
    return np.concatenate(out_c), np.concatenate(out_k)

def temperature_pooled(cell):
    c, k = pool(cell)
    T = find_optimal_temperature(c, k)
    c_t = np.concatenate([apply_temperature(cell.class_conf[l], T) for l in CALIBRATION_LABELS])
    return c_t, k, T

def _edges():
    return np.linspace(BIN_LO, BIN_HI, N_VIS_BINS + 1)

def per_bin_accuracy(conf, corr):
    edges = _edges()
    accs = np.zeros(N_VIS_BINS)
    counts = np.zeros(N_VIS_BINS, dtype=int)
    for i in range(N_VIS_BINS):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < N_VIS_BINS - 1 else (conf >= lo) & (conf <= hi)
        counts[i] = mask.sum()
        if counts[i] > 0:
            accs[i] = corr[mask].mean()
    return accs, counts

def _bin_labels():
    edges = _edges()
    return [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in range(N_VIS_BINS)]

def draw_subplot(ax, variants, title):
    n_var = len(variants)
    x = np.arange(N_VIS_BINS)
    width = 0.8 / max(n_var, 1)

    for i, v in enumerate(variants):
        accs, counts = per_bin_accuracy(v["conf"], v["corr"])
        e_pred, _, _ = ece_pred(v["cc_map"], v["ck_map"], v["pred_sets"])
        offs = (i - (n_var - 1) / 2) * width
        color = v.get("color", PALETTE[i % len(PALETTE)])
        bars = ax.bar(x + offs, accs * 100.0, width * 0.95, color=color,
                      label=f"{v['name']} (ECE_pred: {e_pred:.2f})",
                      edgecolor="white", linewidth=0.6)
        for b, c in zip(bars, counts):
            if c == 0:
                b.set_alpha(0.12)

    edges = _edges()
    mids_pct = (edges[:-1] + edges[1:]) / 2 * 100.0
    ax.step(x, mids_pct, where="mid", linestyle="--", color="black",
            linewidth=1.2, label="Calibration Line")

    ax.set_xticks(x)
    ax.set_xticklabels(_bin_labels(), fontsize=12, rotation=35, ha="right")
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Confidence Interval", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

def render_figure(subplots, suptitle, out_path, figsize=None, dpi=130, layout="row"):
    n = len(subplots)
    if layout == "two_up_one_down" and n == 3:
        if figsize is None:
            figsize = (14.5, 10.5)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 4)
        axes = [
            fig.add_subplot(gs[0, 0:2]),
            fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[1, 1:3]),
        ]
    else:
        if figsize is None:
            figsize = (6.2 * n, 5.2)
        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = [axes]
    for ax, (title, variants) in zip(axes, subplots):
        draw_subplot(ax, variants, title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

def _from_cell(name, cell, color):
    c, k = pool(cell)
    pred_sets = build_pred_sets(cell.records, cell.class_conf)
    return {
        "name": name, "conf": c, "corr": k, "color": color,
        "cc_map": cell.class_conf, "ck_map": cell.class_corr, "pred_sets": pred_sets,
    }

def layout_models_by_dataset(source, modality):
    subplots = []
    for ds in DATASETS_DEFAULT:
        variants = []
        for i, m in enumerate(MODELS_DEFAULT):
            cell = get_cell(source, m, ds, modality)
            if cell is None:
                continue
            variants.append(_from_cell(m, cell, PALETTE[i]))
        subplots.append((DATASET_DISPLAY[ds], variants))
    return subplots, f"Reliability — {source} | modality: {modality}", f"models_by_dataset_{source}_{modality}"

def layout_methods(source, model, modality):
    subplots = []
    for ds in DATASETS_DEFAULT:
        cell = get_cell(source, model, ds, modality)
        if cell is None:
            subplots.append((DATASET_DISPLAY[ds], []))
            continue
        r_c, r_k = pool(cell)
        p_c, p_k = platt_pooled(cell)
        t_c, t_k, T = temperature_pooled(cell)
        pred_sets = build_pred_sets(cell.records, cell.class_conf)
        cc = cell.class_conf
        ck = cell.class_corr
        # build per-label Platt and Temperature maps
        cc_p = {l: platt_oof(cell.class_conf[l], cell.class_corr[l])[0] for l in CALIBRATION_LABELS}
        T_opt = find_optimal_temperature(r_c, r_k)
        cc_t = {l: apply_temperature(cell.class_conf[l], T_opt) for l in CALIBRATION_LABELS}
        variants = [
            {"name": "Raw",                "conf": r_c, "corr": r_k, "color": PALETTE[0],
             "cc_map": cc,   "ck_map": ck, "pred_sets": pred_sets},
            {"name": "Platt (OOF)",        "conf": p_c, "corr": p_k, "color": PALETTE[1],
             "cc_map": cc_p, "ck_map": ck, "pred_sets": pred_sets},
            {"name": f"Temperature T={T:.2f}", "conf": t_c, "corr": t_k, "color": PALETTE[2],
             "cc_map": cc_t, "ck_map": ck, "pred_sets": pred_sets},
        ]
        subplots.append((DATASET_DISPLAY[ds], variants))
    m_slug = model.replace(" ", "").replace("-", "").lower()
    return subplots, f"Recalibration effect — {source} | {model} | modality: {modality}", f"methods_{source}_{m_slug}_{modality}"

def layout_sources(model, modality):
    subplots = []
    for ds in DATASETS_DEFAULT:
        variants = []
        v_cell = get_cell("verbalized", model, ds, modality)
        if v_cell is not None:
            variants.append(_from_cell("Verbalized", v_cell, PALETTE[0]))
        c_cell = get_cell("consistency", model, ds, modality)
        if c_cell is not None:
            variants.append(_from_cell("Consistency", c_cell, PALETTE[1]))
        subplots.append((DATASET_DISPLAY[ds], variants))
    m_slug = model.replace(" ", "").replace("-", "").lower()
    return subplots, f"Verbalized vs Consistency — {model} | modality: {modality}", f"sources_{m_slug}_{modality}"

def run_all():
    out_dir = HERE / "results"
    layouts = []
    layouts.append(layout_models_by_dataset("verbalized", "image_text"))
    for m in MODELS_DEFAULT:
        layouts.append(layout_methods("verbalized", m, "image_text"))
    for m in MODELS_DEFAULT:
        layouts.append(layout_sources(m, "image_text"))
    for subplots, suptitle, slug in layouts:
        render_figure(subplots, suptitle, out_dir / f"reliability_{slug}.png")

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="layout", required=True)

    a = sub.add_parser("models_by_dataset")
    a.add_argument("source", choices=["verbalized", "consistency"])
    a.add_argument("modality", choices=MODALITIES_DEFAULT)

    b = sub.add_parser("methods")
    b.add_argument("source", choices=["verbalized", "consistency"])
    b.add_argument("model")
    b.add_argument("modality", choices=MODALITIES_DEFAULT)

    c = sub.add_parser("sources")
    c.add_argument("model")
    c.add_argument("modality", choices=MODALITIES_DEFAULT)

    sub.add_parser("all")

    args = p.parse_args()
    out_dir = HERE / "results"

    if args.layout == "all":
        run_all()
        return

    if args.layout == "models_by_dataset":
        subplots, suptitle, slug = layout_models_by_dataset(args.source, args.modality)
    elif args.layout == "methods":
        subplots, suptitle, slug = layout_methods(args.source, args.model, args.modality)
    else:
        subplots, suptitle, slug = layout_sources(args.model, args.modality)

    layout = "two_up_one_down" if args.layout == "models_by_dataset" else "row"
    figsize = (14.5, 10.5) if args.layout == "models_by_dataset" else None
    dpi = 160 if args.layout == "models_by_dataset" else 130
    render_figure(subplots, suptitle, out_dir / f"reliability_{slug}.png", figsize=figsize, dpi=dpi, layout=layout)

if __name__ == "__main__":
    main()
