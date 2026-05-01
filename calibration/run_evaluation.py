"""Evaluate verbalized + consistency calibration across the full grid.

Usage:
    python run_evaluation.py

Writes calibration.{txt,csv,json} to results/. Reliability plots: reliability_plots.py.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from calibration_metrics import (
    apply_temperature,
    ece as ece_fn,
    find_optimal_temperature,
    multilabel_calibration_report,
)
from src.hierarchical_f1 import hierarchical_f1
from loader import (
    CALIBRATION_LABELS, Cell,
    load_consistency, load_self_reported,
)

MODELS = ["Qwen 4B", "Qwen 8B", "Gemma 3 4B", "GPT-4.1-mini"]
DATASETS = ["English", "Translated", "Ukrainian"]
MODALITIES = ["text", "image", "image_text"]
N_BINS = 10
N_FOLDS = 5
SEED = 42
THR = 0.5
EPS = 1e-10


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class PlattScaler:
    A: float
    B: float

    def predict(self, conf):
        return 1.0 / (1.0 + np.exp(-(self.A * _logit(conf) + self.B)))


def fit_platt(conf, corr):
    y = corr.astype(int)
    if len(np.unique(y)) < 2:
        return None
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(_logit(conf).reshape(-1, 1), y)
    return PlattScaler(A=float(lr.coef_[0, 0]), B=float(lr.intercept_[0]))


def platt_oof(conf, corr):
    oof = conf.copy()
    y = corr.astype(int)
    if len(np.unique(y)) < 2:
        return oof, []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scalers = []
    for tr, val in skf.split(conf, y):
        if len(np.unique(y[tr])) < 2:
            continue
        s = fit_platt(conf[tr], corr[tr])
        if s is None:
            continue
        oof[val] = s.predict(conf[val])
        scalers.append(s)
    return oof, scalers


def gold_for_eval(labels):
    s = set(labels) if labels else set()
    return set() if "NO_PROPAGANDA" in s else s


def hf1_from_pred(records, pred_sets):
    gold = [gold_for_eval(r.get("gold_labels", [])) for r in records]
    return hierarchical_f1(gold, pred_sets)


def thresholded_preds(conf_map, n, thr=THR):
    return [
        {l for l in CALIBRATION_LABELS if conf_map[l][i] > thr}
        for i in range(n)
    ]


def split_indices(records):
    """Indices of (NO_PROPAGANDA-only samples, propaganda samples)."""
    np_idx, prop_idx = [], []
    for i, r in enumerate(records):
        g = set(r.get("gold_labels") or [])
        if "NO_PROPAGANDA" in g or len(g) == 0:
            np_idx.append(i)
        else:
            prop_idx.append(i)
    return np.asarray(np_idx, dtype=int), np.asarray(prop_idx, dtype=int)


def subset_maps(cc_map, ck_map, idx):
    return {l: cc_map[l][idx] for l in CALIBRATION_LABELS}, {l: ck_map[l][idx] for l in CALIBRATION_LABELS}


def subset_metrics(cc_map, ck_map, idx):
    if idx.size == 0:
        return None
    sc, sk = subset_maps(cc_map, ck_map, idx)
    rep = multilabel_calibration_report(sc, sk, n_bins=N_BINS)
    pos_mi, pos_ma, pos_n = ece_pos(sc, sk)
    return {
        "n":     int(idx.size),
        "ece":     {"micro": rep.micro_ece, "macro": rep.macro_ece},
        "ece_pos": {"micro": pos_mi, "macro": pos_ma, "n_slots": pos_n},
    }


def build_pred_sets(records, cc_map):
    """Build per-sample predicted-label set.

    Verbalized: model wrote labels (records[i]['pred_labels']).
    Consistency: labels picked by ANY of the n_runs (raw conf > 0 ⇔ ≥1 run).
                 Detected by presence of 'consistency_confidence' field.
    """
    n = len(records)
    if records and "consistency_confidence" in records[0]:
        return [{l for l in CALIBRATION_LABELS if cc_map[l][i] > 0.0} for i in range(n)]
    return [set(r.get("pred_labels") or []) for r in records]


def ece_pred(cc_map, ck_map, pred_sets):
    """ECE on label-slots where label ∈ pred_sets[i].

    Threshold-free. Stable under recalibration: prediction set is fixed by raw
    model output, only the confidence per slot changes.
    """
    micro_c, micro_k = [], []
    per_label = []
    n = len(pred_sets)
    for l in CALIBRATION_LABELS:
        mask = np.array([l in pred_sets[i] for i in range(n)], dtype=bool)
        if mask.sum() == 0:
            continue
        c = cc_map[l][mask]
        k = ck_map[l][mask]
        micro_c.append(c)
        micro_k.append(k)
        per_label.append(ece_fn(c, k, n_bins=N_BINS))
    if not micro_c:
        return 0.0, 0.0, 0
    pc = np.concatenate(micro_c)
    pk = np.concatenate(micro_k)
    return float(ece_fn(pc, pk, n_bins=N_BINS)), float(np.mean(per_label)), int(pc.size)


def ece_pos(cc_map, ck_map, thr=THR):
    """Pooled ECE restricted to label-slots where conf > thr.

    Returns (micro_ece, macro_ece, n_pos). Macro = mean per-label ECE among
    labels with at least one positive prediction.
    """
    micro_c, micro_k = [], []
    per_label = []
    for l in CALIBRATION_LABELS:
        c = cc_map[l]
        k = ck_map[l]
        mask = c > thr
        if mask.sum() == 0:
            continue
        micro_c.append(c[mask])
        micro_k.append(k[mask])
        per_label.append(ece_fn(c[mask], k[mask], n_bins=N_BINS))
    if not micro_c:
        return 0.0, 0.0, 0
    pc = np.concatenate(micro_c)
    pk = np.concatenate(micro_k)
    return float(ece_fn(pc, pk, n_bins=N_BINS)), float(np.mean(per_label)), int(pc.size)


def _hf_summary(r):
    m, ma = r["micro"], r["macro_per_label"]
    return {
        "micro_f1": m["f1"], "micro_p": m["precision"], "micro_r": m["recall"],
        "macro_f1": ma["f1"], "macro_p": ma["precision"], "macro_r": ma["recall"],
    }


def eval_cell(cell: Cell):
    cc = {l: cell.class_conf[l] for l in CALIBRATION_LABELS}
    ck = {l: cell.class_corr[l] for l in CALIBRATION_LABELS}

    pooled_c = np.concatenate([cc[l] for l in CALIBRATION_LABELS])
    pooled_k = np.concatenate([ck[l] for l in CALIBRATION_LABELS])
    T = find_optimal_temperature(pooled_c, pooled_k)
    cc_t = {l: apply_temperature(cc[l], T) for l in CALIBRATION_LABELS}

    cc_p, platt_scalers = {}, {}
    for l in CALIBRATION_LABELS:
        oof, scalers = platt_oof(cc[l], ck[l])
        cc_p[l] = oof
        platt_scalers[l] = scalers

    rep_raw = multilabel_calibration_report(cc, ck, n_bins=N_BINS)
    rep_t = multilabel_calibration_report(cc_t, ck, n_bins=N_BINS)
    rep_p = multilabel_calibration_report(cc_p, ck, n_bins=N_BINS)

    raw_pos_mi, raw_pos_ma, raw_pos_n = ece_pos(cc,   ck)
    plt_pos_mi, plt_pos_ma, plt_pos_n = ece_pos(cc_p, ck)
    tmp_pos_mi, tmp_pos_ma, tmp_pos_n = ece_pos(cc_t, ck)

    pred_sets = build_pred_sets(cell.records, cc)  # use raw conf for set
    raw_pr_mi, raw_pr_ma, raw_pr_n = ece_pred(cc,   ck, pred_sets)
    plt_pr_mi, plt_pr_ma, plt_pr_n = ece_pred(cc_p, ck, pred_sets)
    tmp_pr_mi, tmp_pr_ma, tmp_pr_n = ece_pred(cc_t, ck, pred_sets)

    np_idx, prop_idx = split_indices(cell.records)
    splits = {
        "no_propaganda": {
            "raw":   subset_metrics(cc,   ck, np_idx),
            "platt": subset_metrics(cc_p, ck, np_idx),
            "temperature": subset_metrics(cc_t, ck, np_idx),
        },
        "propaganda": {
            "raw":   subset_metrics(cc,   ck, prop_idx),
            "platt": subset_metrics(cc_p, ck, prop_idx),
            "temperature": subset_metrics(cc_t, ck, prop_idx),
        },
    }

    pred_raw = [set(r.get("pred_labels") or []) - {"NO_PROPAGANDA"} for r in cell.records]
    hf_raw = hf1_from_pred(cell.records, pred_raw)
    hf_t = hf1_from_pred(cell.records, thresholded_preds(cc_t, cell.n))
    hf_p = hf1_from_pred(cell.records, thresholded_preds(cc_p, cell.n))

    a_vals, b_vals, platt_params = [], [], {}
    for l, scalers in platt_scalers.items():
        if not scalers:
            continue
        a = float(np.mean([s.A for s in scalers]))
        b = float(np.mean([s.B for s in scalers]))
        a_vals.append(a)
        b_vals.append(b)
        platt_params[l] = {"A": a, "B": b, "n_folds": len(scalers)}

    def _stat(vals):
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    a_mean, a_std = _stat(a_vals)
    b_mean, b_std = _stat(b_vals)

    return {
        "n": cell.n,
        "ece": {
            "raw":         {"micro": rep_raw.micro_ece, "macro": rep_raw.macro_ece},
            "platt":       {"micro": rep_p.micro_ece,   "macro": rep_p.macro_ece},
            "temperature": {"micro": rep_t.micro_ece,   "macro": rep_t.macro_ece},
        },
        "ece_pos": {
            "raw":         {"micro": raw_pos_mi, "macro": raw_pos_ma, "n_slots": raw_pos_n},
            "platt":       {"micro": plt_pos_mi, "macro": plt_pos_ma, "n_slots": plt_pos_n},
            "temperature": {"micro": tmp_pos_mi, "macro": tmp_pos_ma, "n_slots": tmp_pos_n},
        },
        "ece_pred": {
            "raw":         {"micro": raw_pr_mi, "macro": raw_pr_ma, "n_slots": raw_pr_n},
            "platt":       {"micro": plt_pr_mi, "macro": plt_pr_ma, "n_slots": plt_pr_n},
            "temperature": {"micro": tmp_pr_mi, "macro": tmp_pr_ma, "n_slots": tmp_pr_n},
        },
        "split": splits,
        "hf1": {"raw": _hf_summary(hf_raw), "platt": _hf_summary(hf_p), "temperature": _hf_summary(hf_t)},
        "platt_mean": {"a": a_mean, "a_std": a_std, "b": b_mean, "b_std": b_std},
        "platt_per_label": platt_params,
        "temperature_T": float(T),
    }


def fit_platt_full(cc, ck):
    return {l: fit_platt(cc[l], ck[l]) for l in CALIBRATION_LABELS}


def apply_scalers(scalers, cc):
    return {
        l: (scalers[l].predict(cc[l]) if scalers.get(l) is not None else cc[l])
        for l in CALIBRATION_LABELS
    }


GLOBAL_HEADER = [
    "# ============================================================",
    "# Calibration evaluation — global settings",
    "# ============================================================",
    "# Binning            : 10 equal-width bins on [0, 1]",
    "# ECE variants       : Micro (pooled) + Macro (per-label average)",
    "# ECE_pos            : ECE restricted to label-slots with conf > 0.5",
    "#                      (drops abstention mass, exposes calibration of asserted labels)",
    "# ECE_pred           : ECE restricted to label-slots where label is in the model's prediction set",
    "#                      Verbalized: pred set = labels model wrote in JSON output (one inference)",
    "#                      Consistency: pred set = union of labels picked by ANY of n_runs",
    "#                      (threshold-free, stable under recalibration)",
    "# Split              : per-cell split by gold — NO_PROPAGANDA-only vs propaganda samples",
    "#                      (printed only when both subsets have ≥10 samples)",
    "# Train/test split   : Platt = 5-fold StratifiedKFold OOF",
    "#                      Temperature = single T on pooled labels",
    "# Confidence scale   : float in [0, 1]",
    "# NO_PROPAGANDA      : excluded from Platt / Temperature / hF1 thresholding",
    "#                      (sentinel, not a technique; near-random on UA)",
    "# hF1 threshold      : 0.5 on calibrated per-label prob",
    "",
]


def _split_lines(r, min_n=10):
    np_ = r["split"]["no_propaganda"]["raw"]
    pr_ = r["split"]["propaganda"]["raw"]
    if not np_ or not pr_ or np_["n"] < min_n or pr_["n"] < min_n:
        return []
    lines = ["", "  Subset metrics:"]
    for tag, key in (("NP-only ", "no_propaganda"), ("Propag.  ", "propaganda")):
        s = r["split"][key]
        raw = s["raw"]
        plt_ = s["platt"]
        tmp = s["temperature"]
        lines.append(
            f"    [{tag}n={raw['n']:3d}] "
            f"ECE raw mi={raw['ece']['micro']:.4f} ma={raw['ece']['macro']:.4f} | "
            f"Platt mi={plt_['ece']['micro']:.4f} ma={plt_['ece']['macro']:.4f} | "
            f"Temp mi={tmp['ece']['micro']:.4f} ma={tmp['ece']['macro']:.4f}"
        )
        lines.append(
            f"    [{tag}      ] "
            f"ECE_pos raw mi={raw['ece_pos']['micro']:.4f} n={raw['ece_pos']['n_slots']:4d} | "
            f"Platt mi={plt_['ece_pos']['micro']:.4f} n={plt_['ece_pos']['n_slots']:4d} | "
            f"Temp mi={tmp['ece_pos']['micro']:.4f} n={tmp['ece_pos']['n_slots']:4d}"
        )
    return lines


def cell_block(header, r):
    base = [
        "############################################################",
        f"# {header}",
        "############################################################",
        "",
        f"  ECE (raw)          : micro={r['ece']['raw']['micro']:.4f}  macro={r['ece']['raw']['macro']:.4f}",
        f"  ECE (Platt)        : micro={r['ece']['platt']['micro']:.4f}  macro={r['ece']['platt']['macro']:.4f}",
        f"  ECE (Temperature)  : micro={r['ece']['temperature']['micro']:.4f}  macro={r['ece']['temperature']['macro']:.4f}",
        "",
        f"  ECE_pos (raw)      : micro={r['ece_pos']['raw']['micro']:.4f}  macro={r['ece_pos']['raw']['macro']:.4f}  n_pos={r['ece_pos']['raw']['n_slots']}",
        f"  ECE_pos (Platt)    : micro={r['ece_pos']['platt']['micro']:.4f}  macro={r['ece_pos']['platt']['macro']:.4f}  n_pos={r['ece_pos']['platt']['n_slots']}",
        f"  ECE_pos (Temp)     : micro={r['ece_pos']['temperature']['micro']:.4f}  macro={r['ece_pos']['temperature']['macro']:.4f}  n_pos={r['ece_pos']['temperature']['n_slots']}",
        "",
        f"  ECE_pred (raw)     : micro={r['ece_pred']['raw']['micro']:.4f}  macro={r['ece_pred']['raw']['macro']:.4f}  n_pred={r['ece_pred']['raw']['n_slots']}",
        f"  ECE_pred (Platt)   : micro={r['ece_pred']['platt']['micro']:.4f}  macro={r['ece_pred']['platt']['macro']:.4f}  n_pred={r['ece_pred']['platt']['n_slots']}",
        f"  ECE_pred (Temp)    : micro={r['ece_pred']['temperature']['micro']:.4f}  macro={r['ece_pred']['temperature']['macro']:.4f}  n_pred={r['ece_pred']['temperature']['n_slots']}",
    ] + _split_lines(r) + [
        "",
        f"  hF1 raw            : Mi={r['hf1']['raw']['micro_f1']:.4f}  Ma={r['hf1']['raw']['macro_f1']:.4f}",
        f"  hF1 Platt          : Mi={r['hf1']['platt']['micro_f1']:.4f}  Ma={r['hf1']['platt']['macro_f1']:.4f}",
        f"  hF1 Temperature    : Mi={r['hf1']['temperature']['micro_f1']:.4f}  Ma={r['hf1']['temperature']['macro_f1']:.4f}",
        "",
        "  Fitted params:",
        f"    Platt a (mean±sd): {r['platt_mean']['a']:.3f} ± {r['platt_mean']['a_std']:.3f}",
        f"    Platt b (mean±sd): {r['platt_mean']['b']:.3f} ± {r['platt_mean']['b_std']:.3f}",
        f"    Temperature T    : {r['temperature_T']:.3f}",
        "",
        f"  n (samples)        : {r['n']}",
        "",
    ]
    return "\n".join(base)


def cell_slug(source, model, ds, md):
    m = model.replace(" ", "").replace("-", "").lower()
    d = ds.lower()
    return f"{source}_{m}_{d}_{md}"


def _csv_row(source, model, ds, md, r):
    return {
        "source": source, "model": model, "dataset": ds, "modality": md, "n": r["n"],
        "ece_raw_micro": r["ece"]["raw"]["micro"],
        "ece_raw_macro": r["ece"]["raw"]["macro"],
        "ece_platt_micro": r["ece"]["platt"]["micro"],
        "ece_platt_macro": r["ece"]["platt"]["macro"],
        "ece_temp_micro": r["ece"]["temperature"]["micro"],
        "ece_temp_macro": r["ece"]["temperature"]["macro"],
        "ece_pos_raw_micro":   r["ece_pos"]["raw"]["micro"],
        "ece_pos_raw_macro":   r["ece_pos"]["raw"]["macro"],
        "ece_pos_platt_micro": r["ece_pos"]["platt"]["micro"],
        "ece_pos_platt_macro": r["ece_pos"]["platt"]["macro"],
        "ece_pos_temp_micro":  r["ece_pos"]["temperature"]["micro"],
        "ece_pos_temp_macro":  r["ece_pos"]["temperature"]["macro"],
        "ece_pos_n_slots_raw": r["ece_pos"]["raw"]["n_slots"],
        "ece_pred_raw_micro":   r["ece_pred"]["raw"]["micro"],
        "ece_pred_raw_macro":   r["ece_pred"]["raw"]["macro"],
        "ece_pred_platt_micro": r["ece_pred"]["platt"]["micro"],
        "ece_pred_platt_macro": r["ece_pred"]["platt"]["macro"],
        "ece_pred_temp_micro":  r["ece_pred"]["temperature"]["micro"],
        "ece_pred_temp_macro":  r["ece_pred"]["temperature"]["macro"],
        "ece_pred_n_slots_raw": r["ece_pred"]["raw"]["n_slots"],
        "hf1_raw_micro": r["hf1"]["raw"]["micro_f1"],
        "hf1_raw_macro": r["hf1"]["raw"]["macro_f1"],
        "hf1_platt_micro": r["hf1"]["platt"]["micro_f1"],
        "hf1_platt_macro": r["hf1"]["platt"]["macro_f1"],
        "hf1_temp_micro": r["hf1"]["temperature"]["micro_f1"],
        "hf1_temp_macro": r["hf1"]["temperature"]["macro_f1"],
        "platt_a_mean": r["platt_mean"]["a"],
        "platt_a_std": r["platt_mean"]["a_std"],
        "platt_b_mean": r["platt_mean"]["b"],
        "platt_b_std": r["platt_mean"]["b_std"],
        "temperature_T": r["temperature_T"],
    }


def _json_cell(r):
    return {k: r[k] for k in ("n", "ece", "ece_pos", "ece_pred", "split", "hf1", "platt_mean", "platt_per_label", "temperature_T")}


def main():
    out_dir = HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    txt = list(GLOBAL_HEADER)
    csv_rows = []
    json_out = {"cells": {}, "transfer": []}

    def _process(source, model, ds, md, cell):
        res = eval_cell(cell)
        slug = cell_slug(source, model, ds, md)
        header = f"{source.upper()}: {slug}"
        txt.append(cell_block(header, res))
        csv_rows.append(_csv_row(source, model, ds, md, res))
        json_out["cells"][slug] = _json_cell(res)

    for model in MODELS:
        for ds in DATASETS:
            for md in MODALITIES:
                cell = load_self_reported(model, ds, md)
                if cell is not None:
                    _process("verbalized", model, ds, md, cell)

    for model in MODELS:
        for ds in DATASETS:
            cell = load_consistency(model, ds)
            if cell is not None:
                _process("consistency", model, ds, "image_text", cell)

    # Q5: English (Prop950) → Translated / Ukrainian, verbalized only
    txt += [
        "############################################################",
        "# Q5 — CROSS-DATASET TRANSFER  (fit on English, apply raw)",
        "############################################################",
        "",
    ]
    for model in MODELS:
        for md in MODALITIES:
            src = load_self_reported(model, "English", md)
            if src is None:
                continue
            src_cc = {l: src.class_conf[l] for l in CALIBRATION_LABELS}
            src_ck = {l: src.class_corr[l] for l in CALIBRATION_LABELS}
            scalers = fit_platt_full(src_cc, src_ck)
            for tgt_ds in ("Translated", "Ukrainian"):
                tgt = load_self_reported(model, tgt_ds, md)
                if tgt is None:
                    continue
                tgt_cc = {l: tgt.class_conf[l] for l in CALIBRATION_LABELS}
                tgt_ck = {l: tgt.class_corr[l] for l in CALIBRATION_LABELS}
                raw_rep = multilabel_calibration_report(tgt_cc, tgt_ck, n_bins=N_BINS)
                transferred = apply_scalers(scalers, tgt_cc)
                xfer_rep = multilabel_calibration_report(transferred, tgt_ck, n_bins=N_BINS)

                pred_sets = build_pred_sets(tgt.records, tgt_cc)
                raw_pr_mi, raw_pr_ma, raw_pr_n = ece_pred(tgt_cc,     tgt_ck, pred_sets)
                xf_pr_mi,  xf_pr_ma,  _        = ece_pred(transferred, tgt_ck, pred_sets)

                line = (f"  {model:14s} | {md:10s} | English → {tgt_ds:10s}  "
                        f"all   mi: {raw_rep.micro_ece:.4f} → {xfer_rep.micro_ece:.4f}  "
                        f"ma: {raw_rep.macro_ece:.4f} → {xfer_rep.macro_ece:.4f}  | "
                        f"pred mi: {raw_pr_mi:.4f} → {xf_pr_mi:.4f}  "
                        f"ma: {raw_pr_ma:.4f} → {xf_pr_ma:.4f}  (n_pred={raw_pr_n})")
                txt.append(line)
                json_out["transfer"].append({
                    "model": model, "modality": md, "source": "English", "target": tgt_ds,
                    "raw_micro_ece":      raw_rep.micro_ece,  "raw_macro_ece":      raw_rep.macro_ece,
                    "transfer_micro_ece": xfer_rep.micro_ece, "transfer_macro_ece": xfer_rep.macro_ece,
                    "raw_micro_ece_pred":      raw_pr_mi, "raw_macro_ece_pred":      raw_pr_ma,
                    "transfer_micro_ece_pred": xf_pr_mi,  "transfer_macro_ece_pred": xf_pr_ma,
                    "n_pred": raw_pr_n,
                })
    txt.append("")

    (out_dir / "calibration.txt").write_text("\n".join(txt))
    pd.DataFrame(csv_rows).to_csv(out_dir / "calibration.csv", index=False)
    (out_dir / "calibration.json").write_text(json.dumps(json_out, indent=2))
    print(f"Wrote {out_dir}/calibration.{{txt,csv,json}}")


if __name__ == "__main__":
    main()
