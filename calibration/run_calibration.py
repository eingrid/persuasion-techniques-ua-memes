"""Self-reported per-label confidence calibration. Requires vLLM server.

Usage:
    vllm serve Qwen/Qwen3-VL-4B-Instruct --dtype bfloat16 --limit-mm-per-prompt.video 0
    python run_calibration.py \\
        --model_id Qwen/Qwen3-VL-4B-Instruct \\
        --dataset_path ../datasets/translated/ \\
        --language uk \\
        --output_dir ./cache_calibration \\
        --modes text image image+text \\
        --split test
"""
import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from json_repair import repair_json
from openai import OpenAI
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / ".."))

from calibration_metrics import multilabel_calibration_report, reliability_diagram
from src.utils import encode_image_base64



UNIQUE_LABELS = [
    "Appeal to (Strong) Emotions",
    "Appeal to authority",
    "Appeal to fear/prejudice",
    "Bandwagon",
    "Black-and-white Fallacy/Dictatorship",
    "Causal Oversimplification",
    "Doubt",
    "Exaggeration/Minimisation",
    "Flag-waving",
    "Glittering generalities (Virtue)",
    "Loaded Language",
    "Misrepresentation of Someone's Position (Straw Man)",
    "Name calling/Labeling",
    "Obfuscation, Intentional vagueness, Confusion",
    "Presenting Irrelevant Data (Red Herring)",
    "Reductio ad hitlerum",
    "Repetition",
    "Slogans",
    "Smears",
    "Thought-terminating cliché",
    "Transfer",
    "Whataboutism",
    "NO_PROPAGANDA",
]

LANG_INSTRUCTION = {
    "uk": "All values in referred_words and referred_image_concepts must be in Ukrainian.",
    "en": "All values in referred_words and referred_image_concepts must be in English.",
}
LANG_INSTRUCTION_IMAGE_ONLY = {
    "uk": "All values in referred_image_concepts must be in Ukrainian.",
    "en": "All values in referred_image_concepts must be in English.",
}

BASE_PROMPT = '''You are an expert in analyzing communication strategies in visual media.
Classify the provided content into one or multiple of the following persuasion techniques :
{unique_labels}
As output, provide JSON with four fields: "labels", "referred_words", "referred_image_concepts", and "confidence".
- "labels": list of persuasion techniques used, or "NO_PROPAGANDA" if the content doesn't contain persuasion techniques.
- "referred_words": list of the most important INDIVIDUAL words from the content text supporting your classification. Only single words, not phrases. Only key words most relevant to the classification.
- "referred_image_concepts": up to 5 visual elements most strongly supporting your classification, ranked by importance. No duplicates. Empty list if no image. The list could be empty if the image doesn't provide any clues.
- "confidence": a dict mapping EVERY label from the list above to a float in [0.0, 1.0] — your confidence that this label applies to the content. Include all labels, not only the predicted ones. Use values close to 1.0 for labels you are certain apply, and values close to 0.0 for labels you are certain do not apply.
Visual elements rules:
- Be concrete: name actual people, objects, symbols. Do NOT write abstract scene descriptions like "people in a stressful situation".
- For visual elements: use the shortest identifying phrase — "Obama"/"Обама", "flag"/"прапор". Do not use words on the image themselves.
Language note: {lang_note}
JSON format output example (abbreviated):
{{
    "labels": ["Loaded Language", "Flag-waving"],
    "referred_words": ["word1", "word2"],
    "referred_image_concepts": ["concept1"],
    "confidence": {{"Loaded Language": 0.95, "Flag-waving": 0.80, "Doubt": 0.05, "Bandwagon": 0.10, ...}}
}}
Text from the content: {text}
Output:
'''

IMAGE_ONLY_PROMPT = '''You are an expert in analyzing communication strategies in visual media.
Classify the provided content IMAGE into one or multiple of the following persuasion techniques :
{unique_labels}
As output, provide JSON with three fields: "labels", "referred_image_concepts", and "confidence".
- "labels": list of persuasion techniques used, or "NO_PROPAGANDA" if the content doesn't contain persuasion techniques.
- "referred_image_concepts": up to 5 visual elements most strongly supporting your classification, ranked by importance. No duplicates. The list could be empty if the image doesn't provide any clues.
- "confidence": a dict mapping EVERY label from the list above to a float in [0.0, 1.0] — your confidence that this label applies. Include all labels, not only the predicted ones.
Visual elements rules:
- Be concrete: name actual people, objects, symbols. Do NOT write abstract scene descriptions like "people in a stressful situation".
- For words/text visible in the image: do not include ANY of them to referred_image_concepts, as they belong to the text modality. Only visual elements should be included.
- For visual elements: use the shortest identifying phrase — "Obama"/"Обама", "flag"/"прапор". Do not use words on the image themselves.
Language note: {lang_note}
JSON format output example (abbreviated):
{{
    "labels": ["Loaded Language", "Flag-waving"],
    "referred_image_concepts": ["concept1"],
    "confidence": {{"Loaded Language": 0.95, "Flag-waving": 0.80, "Doubt": 0.05, "Bandwagon": 0.10, ...}}
}}
Output:
'''


def build_prompt_and_content(mode, language, ocr_text="", image_b64=None):
    """Build OpenAI-style message content for the given mode."""
    lang_note = LANG_INSTRUCTION.get(language, "")
    img_lang_note = LANG_INSTRUCTION_IMAGE_ONLY.get(language, "")
    content = []

    if mode == "text":
        prompt = BASE_PROMPT.format(lang_note=lang_note, unique_labels=UNIQUE_LABELS, text=ocr_text)
        content.append({"type": "text", "text": prompt})
    elif mode == "image":
        prompt = IMAGE_ONLY_PROMPT.format(unique_labels=UNIQUE_LABELS, lang_note=img_lang_note)
        if image_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
        content.append({"type": "text", "text": prompt})
    else:  # image+text
        prompt = BASE_PROMPT.format(lang_note=lang_note, unique_labels=UNIQUE_LABELS, text=ocr_text)
        if image_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
        content.append({"type": "text", "text": prompt})

    return content


def parse_confidence(raw):
    if not isinstance(raw, dict):
        return {}
    out = {}
    for label in UNIQUE_LABELS:
        val = raw.get(label)
        try:
            f = float(val)
            if 0.0 <= f <= 1.0:
                out[label] = f
        except (TypeError, ValueError):
            pass
    return out


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_labels(labels):
    s = set(labels) if labels else set()
    return {"NO_PROPAGANDA"} if "NO_PROPAGANDA" in s else s


def run_inference(mode, dataset, img_dir, out_dir, client, cfg):
    mode_dir = out_dir / mode.replace("+", "_")
    mode_dir.mkdir(parents=True, exist_ok=True)
    manifest, failures = [], []

    for item in tqdm(dataset, desc=f"Mode={mode}"):
        sample_id = item["id"]
        cache_file = mode_dir / f"{sample_id}.json"

        if cache_file.exists():
            manifest.append({"id": sample_id, "path": str(cache_file)})
            continue

        try:
            ocr_text = item.get(cfg.text_column, "")
            image_path = img_dir / item.get(cfg.image_column, "")

            image_b64 = None
            if mode in ("image", "image+text"):
                image_b64 = encode_image_base64(image_path)

            content = build_prompt_and_content(mode, cfg.language, ocr_text, image_b64)

            response = client.chat.completions.create(
                model=cfg.model_id,
                messages=[{"role": "user", "content": content}],
                max_tokens=cfg.max_new_tokens,
                temperature=0,
            )

            gen_text = response.choices[0].message.content
            try:
                parsed = json.loads(repair_json(gen_text))
                if not isinstance(parsed, dict):
                    print(f"  WARN {sample_id}: parsed is {type(parsed).__name__}, not dict. Raw: {gen_text[:200]}")
                    parsed = {}
            except Exception:
                print(f"  WARN {sample_id}: JSON parse failed. Raw: {gen_text[:200]}")
                parsed = {}

            record = {
                "id": sample_id,
                "mode": mode,
                "gold_labels": item.get(cfg.label_column, []),
                "pred_labels": parsed.get("labels", []),
                "referred_words": parsed.get("referred_words", []),
                "referred_image_concepts": parsed.get("referred_image_concepts", []),
                "confidence": parse_confidence(parsed.get("confidence")),
                "generated_text": gen_text,
                "parse_ok": bool(parsed.get("labels")),
            }

            cache_file.write_text(json.dumps(record, ensure_ascii=False, indent=2))
            manifest.append({"id": sample_id, "path": str(cache_file)})

        except Exception as e:
            print(f"  FAIL {sample_id}: {e}")
            traceback.print_exc()
            failures.append({"id": sample_id, "error": str(e)})

    (mode_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (mode_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    print(f"Mode '{mode}': {len(manifest)} cached, {len(failures)} failures")
    return manifest


def build_mode_data(mode_records, modes):
    mode_data = {}
    for mode in modes:
        records = mode_records[mode]
        per_label_conf = {lbl: [] for lbl in UNIQUE_LABELS}
        per_label_corr = {lbl: [] for lbl in UNIQUE_LABELS}
        n_missing = 0

        for r in records:
            gold = clean_labels(r["gold_labels"])
            conf_dict = r.get("confidence", {})
            if not conf_dict:
                n_missing += 1
                continue
            for label in UNIQUE_LABELS:
                val = conf_dict.get(label)
                if val is None:
                    continue
                per_label_conf[label].append(float(val))
                per_label_corr[label].append(1.0 if label in gold else 0.0)

        class_conf = {lbl: np.array(v) for lbl, v in per_label_conf.items() if v}
        class_corr = {lbl: np.array(v) for lbl, v in per_label_corr.items() if v}
        mode_data[mode] = {"class_conf": class_conf, "class_corr": class_corr}

        total = sum(len(v) for v in class_conf.values())
        print(f"[{mode}] labels with data: {len(class_conf)}, pairs: {total}, missing conf: {n_missing}")

    return mode_data


def run_calibration(mode_data, modes, n_bins, out_dir):
    reports = {}
    for mode, data in mode_data.items():
        rep = multilabel_calibration_report(data["class_conf"], data["class_corr"], n_bins=n_bins)
        reports[mode] = rep
        print(f"\n{'='*60}\n  Mode: {mode}")
        print(rep)

    n_modes = len(modes)
    fig, axes = plt.subplots(1, n_modes, figsize=(6 * n_modes, 5))
    if n_modes == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        data = mode_data[mode]
        rep = reports[mode]
        all_conf = np.concatenate(list(data["class_conf"].values()))
        all_corr = np.concatenate(list(data["class_corr"].values()))
        reliability_diagram(all_conf, all_corr, n_bins=n_bins,
                            title=f"{mode}  (micro ECE={rep.micro_ece:.3f})", ax=ax)
    fig.suptitle("Reliability Diagrams — Per-Label Self-Reported Confidence", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "reliability_diagrams.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reliability_diagrams.png")

    results_out = {}
    for mode, rep in reports.items():
        results_out[mode] = {
            "macro_ece": rep.macro_ece,
            "macro_ace": rep.macro_ace,
            "macro_brier": rep.macro_brier,
            "micro_ece": rep.micro_ece,
            "micro_brier": rep.micro_brier,
            "per_label": {
                lbl: {
                    "ece": r.ece,
                    "ace": r.ace,
                    "brier": r.brier,
                    "overconfidence": r.overconfidence,
                    "mean_confidence": r.mean_confidence,
                    "accuracy": r.accuracy,
                    "n_samples": r.n_samples,
                }
                for lbl, r in rep.per_class.items()
            },
        }

    out_path = out_dir / "calibration_results.json"
    out_path.write_text(json.dumps(results_out, indent=2))
    print(f"Saved calibration_results.json")

    print(f"\n{'Mode':<12} {'macro ECE':>10} {'micro ECE':>10} {'macro Brier':>12} {'micro Brier':>12}")
    print("-" * 60)
    for mode, d in results_out.items():
        print(f"{mode:<12} {d['macro_ece']:>10.4f} {d['micro_ece']:>10.4f} "
              f"{d['macro_brier']:>12.4f} {d['micro_brier']:>12.4f}")

    return reports


def parse_args():
    p = argparse.ArgumentParser(description="Calibration evaluation with self-reported per-label confidence")
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--language", default="uk", choices=["uk", "en"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--modes", nargs="+", default=["text", "image", "image+text"])
    p.add_argument("--split", default="test")
    p.add_argument("--label_column", default="labels")
    p.add_argument("--text_column", default="text")
    p.add_argument("--image_column", default="image")
    p.add_argument("--max_new_tokens", type=int, default=600)
    p.add_argument("--n_bins", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_only", action="store_true",
                   help="Skip inference, only run calibration on existing cache")
    p.add_argument("--api_base", default="http://localhost:8000/v1")
    return p.parse_args()


def main():
    cfg = parse_args()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "run_config.json").write_text(json.dumps(vars(cfg), indent=2))

    dataset_path = Path(cfg.dataset_path)
    ann_path = dataset_path / "annotations" / f"{cfg.split}.jsonl"
    dataset = load_jsonl(ann_path)
    img_dir = dataset_path / "images"
    print(f"Loaded {len(dataset)} samples from {ann_path}")

    if not cfg.eval_only:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), base_url=cfg.api_base, timeout=3600)
        print(f"Using vLLM at {cfg.api_base}, model={cfg.model_id}")

        for mode in cfg.modes:
            run_inference(mode, dataset, img_dir, out_dir, client, cfg)
        print("\nAll modes done.")

    def _load_json(p):
        text = p.read_text()
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            if "Extra data" in str(e):
                return json.loads(text[:e.pos])
            raise

    mode_records = {}
    for mode in cfg.modes:
        mode_dir = out_dir / mode.replace("+", "_")
        manifest = json.loads((mode_dir / "manifest.json").read_text())
        mode_records[mode] = [_load_json(Path(m["path"])) for m in manifest]
        print(f"[{mode}] loaded {len(mode_records[mode])} records")

    mode_data = build_mode_data(mode_records, cfg.modes)
    run_calibration(mode_data, cfg.modes, cfg.n_bins, out_dir)


if __name__ == "__main__":
    main()
