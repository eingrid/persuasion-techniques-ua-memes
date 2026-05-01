"""Loader for calibration cache. Returns Cell(records, class_conf, class_corr) per (source, model, dataset, modality)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.resolve()

UNIQUE_LABELS = [
    "Appeal to (Strong) Emotions", "Appeal to authority", "Appeal to fear/prejudice",
    "Bandwagon", "Black-and-white Fallacy/Dictatorship", "Causal Oversimplification",
    "Doubt", "Exaggeration/Minimisation", "Flag-waving",
    "Glittering generalities (Virtue)", "Loaded Language",
    "Misrepresentation of Someone's Position (Straw Man)", "Name calling/Labeling",
    "Obfuscation, Intentional vagueness, Confusion",
    "Presenting Irrelevant Data (Red Herring)", "Reductio ad hitlerum",
    "Repetition", "Slogans", "Smears", "Thought-terminating cliché", "Transfer",
    "Whataboutism", "NO_PROPAGANDA",
]
CALIBRATION_LABELS = [l for l in UNIQUE_LABELS if l != "NO_PROPAGANDA"]

DATASET_DIR_SUFFIX = {
    "English": "propaganda950",
    "Translated": "translated",
    "Ukrainian": "ukrainian",
}

SELF_REPORTED_DIR = {
    "Qwen 4B":      "cache_calibration_{ds}",
    "Qwen 8B":      "cache_calibration_qwen8b_{ds}",
    "Gemma 3 4B":   "cache_calibration_gemma3_4b_{ds}",
    "GPT-4.1-mini": "cache_calibration_gpt41mini_{ds}",
}
CONSISTENCY_DIR = {
    "Qwen 4B":      "4b_{ds}",
    "Qwen 8B":      "8b_{ds}",
    "Gemma 3 4B":   "gemma3_4b_{ds}",
    "GPT-4.1-mini": "gpt41mini_{ds}",
}


@dataclass
class Cell:
    records: list
    class_conf: dict
    class_corr: dict
    n: int


def _safe_load(p):
    try:
        obj = json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def _build_arrays(records, conf_field):
    per_conf = {l: [] for l in UNIQUE_LABELS}
    per_corr = {l: [] for l in UNIQUE_LABELS}
    for r in records:
        gold = set(r.get("gold_labels") or [])
        if "NO_PROPAGANDA" in gold:
            gold = {"NO_PROPAGANDA"}
        conf_dict = r.get(conf_field) or {}
        for label in UNIQUE_LABELS:
            v = conf_dict.get(label, 0.0)
            try:
                v = float(v)
                if not 0.0 <= v <= 1.0:
                    v = 0.0
            except (TypeError, ValueError):
                v = 0.0
            per_conf[label].append(v)
            per_corr[label].append(1.0 if label in gold else 0.0)
    return (
        {l: np.asarray(per_conf[l]) for l in UNIQUE_LABELS},
        {l: np.asarray(per_corr[l]) for l in UNIQUE_LABELS},
    )


def load_self_reported(model, dataset, modality):
    ds = DATASET_DIR_SUFFIX[dataset]
    dir_name = SELF_REPORTED_DIR[model].format(ds=ds)
    mode_dir = ROOT / "calibration" / "cache_self_reported" / dir_name / modality.replace("+", "_")
    if not mode_dir.exists():
        return None
    files = sorted(
        f for f in mode_dir.glob("*.json")
        if f.name not in ("manifest.json", "failures.json")
        and not f.name.startswith("low_hf1")
    )
    records = [r for r in (_safe_load(f) for f in files) if r]
    if not records:
        return None
    cc, ck = _build_arrays(records, "confidence")
    return Cell(records, cc, ck, len(records))


def load_consistency(model, dataset):
    ds = DATASET_DIR_SUFFIX[dataset]
    dir_name = CONSISTENCY_DIR[model].format(ds=ds)
    run_dir = ROOT / "calibration" / "consistency_calibration" / dir_name
    if not run_dir.exists():
        return None
    agg_files = sorted(f for f in run_dir.glob("*.json") if "_run" not in f.name)
    records = [r for r in (_safe_load(f) for f in agg_files) if r]
    if not records:
        return None
    cc, ck = _build_arrays(records, "consistency_confidence")
    return Cell(records, cc, ck, len(records))
