"""Hierarchical-F1 over inference caches.

Usage:
    python compute_metrics.py

Sources: explainability/data_*/, calibration/cache_self_reported/, calibration/consistency_calibration/.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from src.hierarchical_f1 import hierarchical_f1  # noqa: E402


def safe_load(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def clean_gold(labels):
    s = set(labels)
    if "NO_PROPAGANDA" in s:
        return set()
    return s


def is_ukrainian(name: str) -> bool:
    n = name.lower()
    return "ukrainian" in n or "_ukr_" in n or n.endswith("_ukr") or "_ukr" in n


def has_no_propaganda(record) -> bool:
    g = record.get("gold_labels") or []
    return "NO_PROPAGANDA" in set(g) or len(g) == 0


def print_metrics(title, gold, pred):
    results = hierarchical_f1(gold, pred)
    print(f"\n{'='*60}\n  {title}\n{'='*60}")
    for key, label in [
        ("micro", "Micro hF1     "),
        ("macro_per_label", "Macro hF1     "),
        ("per_sample", "Per-sample hF1"),
    ]:
        d = results[key]
        print(f"  {label}: P={d['precision']:.4f}  R={d['recall']:.4f}  F1={d['f1']:.4f}")


def eval_modes(cache_root: Path, header: str):
    if not cache_root.exists():
        return
    print(f"\n\n{'#'*60}\n# {header}\n{'#'*60}")
    ukr = is_ukrainian(cache_root.name) or is_ukrainian(header)
    for mode_dir in sorted(cache_root.iterdir()):
        if not mode_dir.is_dir():
            continue
        sample_files = sorted(
            f for f in mode_dir.glob("*.json")
            if f.name not in ("manifest.json", "failures.json")
            and not f.name.startswith("low_hf1")
        )
        if not sample_files:
            continue
        records = [r for r in (safe_load(f) for f in sample_files) if r]
        gold = [clean_gold(r["gold_labels"]) for r in records]
        pred = [r["pred_labels"] for r in records]
        print_metrics(f"Mode: {mode_dir.name}  (n={len(records)})", gold, pred)

        if ukr:
            filt = [(g, p, r) for g, p, r in zip(gold, pred, records) if not has_no_propaganda(r)]
            if filt:
                fg, fp, _ = zip(*filt)
                print_metrics(
                    f"Mode: {mode_dir.name}  [UA, NO_PROPAGANDA excluded]  (n={len(filt)})",
                    list(fg), list(fp),
                )


def eval_consistency(run_dir: Path):
    agg_files = sorted(f for f in run_dir.glob("*.json") if "_run" not in f.name)
    if not agg_files:
        return
    print(f"\n\n{'#'*60}\n# CONSISTENCY (run0): {run_dir.name}\n{'#'*60}")
    ukr = is_ukrainian(run_dir.name)

    gold, pred, aggs = [], [], []
    missing = 0
    for agg_path in agg_files:
        agg = safe_load(agg_path)
        run0 = safe_load(run_dir / f"{agg_path.stem}_run0.json")
        if agg is None or run0 is None:
            missing += 1
            continue
        gold.append(clean_gold(agg.get("gold_labels", [])))
        pred.append(run0.get("pred_labels", []))
        aggs.append(agg)

    if missing:
        print(f"  (skipped {missing} samples without run0)")
    print_metrics(f"n={len(gold)}", gold, pred)

    if ukr:
        filt = [(g, p) for g, p, r in zip(gold, pred, aggs) if not has_no_propaganda(r)]
        if filt:
            fg, fp = zip(*filt)
            print_metrics(
                f"[UA, NO_PROPAGANDA excluded]  n={len(filt)}",
                list(fg), list(fp),
            )


def main():
    expl_root = ROOT / "explainability"
    for d in sorted(expl_root.glob("data_*")):
        if d.is_dir():
            eval_modes(d / "inference_cache", f"EXPLAINABILITY: {d.name}")

    sr_root = ROOT / "calibration" / "cache_self_reported"
    for d in sorted(sr_root.glob("cache_calibration_*")):
        if d.is_dir():
            eval_modes(d, f"SELF-REPORTED: {d.name}")

    cons_root = ROOT / "calibration" / "consistency_calibration"
    for d in sorted(cons_root.iterdir()):
        if d.is_dir():
            eval_consistency(d)


if __name__ == "__main__":
    main()
