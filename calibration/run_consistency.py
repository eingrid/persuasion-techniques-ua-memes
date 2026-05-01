"""Consistency-based calibration: sample N times, use label frequency as confidence. Requires vLLM server.

Usage:
    vllm serve Qwen/Qwen3-VL-4B-Instruct --dtype bfloat16 --limit-mm-per-prompt image=1
    python run_consistency.py \\
        --model_id Qwen/Qwen3-VL-4B-Instruct \\
        --dataset_path ../datasets/translated/ \\
        --language uk \\
        --output_dir ./consistency_calibration/4b_translated \\
        --n_runs 10 \\
        --temperature 0.7 \\
        --split test
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from json_repair import repair_json
from openai import OpenAI
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR / ".."))

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

BASE_PROMPT = '''You are an expert in analyzing communication strategies in visual media.
Classify the provided content into one or multiple of the following persuasion techniques :
{unique_labels}
As output, provide JSON with three fields: "labels", "referred_words", and "referred_image_concepts".
- "labels": list of persuasion techniques used, or "NO_PROPAGANDA" if the content doesn't contain persuasion techniques.
- "referred_words": list of the most important INDIVIDUAL words from the content text supporting your classification. Only single words, not phrases. Only key words most relevant to the classification.
- "referred_image_concepts": up to 5 visual elements most strongly supporting your classification, ranked by importance. No duplicates. Empty list if no image. The list could be empty if the image doesn't provide any clues.
Visual elements rules:
- Be concrete: name actual people, objects, symbols. Do NOT write abstract scene descriptions like "people in a stressful situation".
- For visual elements: use the shortest identifying phrase — "Obama"/"Обама", "flag"/"прапор". Do not use words on the image themselves.
Language note: {lang_note}
JSON format output example (abbreviated):
{{
    "labels": ["Loaded Language", "Flag-waving"],
    "referred_words": ["word1", "word2"],
    "referred_image_concepts": ["concept1"]
}}
Text from the content: {text}
Output:
'''


def build_prompt(language, ocr_text=""):
    lang_note = LANG_INSTRUCTION.get(language, "")
    return BASE_PROMPT.format(lang_note=lang_note, unique_labels=UNIQUE_LABELS, text=ocr_text)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_single(client, model_id, prompt, image_b64, cfg):
    """Single sampling pass; returns parsed labels and raw text."""
    content = []
    if image_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
        })
    content.append({"type": "text", "text": prompt})

    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": content}],
        max_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )

    gen_text = response.choices[0].message.content
    try:
        parsed = json.loads(repair_json(gen_text))
    except Exception:
        parsed = {}

    labels = parsed.get("labels", [])
    if not isinstance(labels, list):
        labels = []
    return labels, gen_text


def run_consistency(dataset, img_dir, out_dir, client, cfg):
    """Run N sampling passes per sample; cache each run and aggregate label frequencies."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, failures = [], []

    for item in tqdm(dataset, desc="Consistency sampling"):
        sample_id = item["id"]
        agg_file = out_dir / f"{sample_id}.json"

        if agg_file.exists():
            manifest.append({"id": sample_id, "path": str(agg_file)})
            continue

        try:
            ocr_text = item.get(cfg.text_column, "")
            image_path = img_dir / item.get(cfg.image_column, "")
            prompt = build_prompt(cfg.language, ocr_text)

            image_b64 = encode_image_base64(image_path)

            all_run_labels = []
            all_run_texts = []
            for run_idx in range(cfg.n_runs):
                run_file = out_dir / f"{sample_id}_run{run_idx}.json"
                if run_file.exists():
                    run_data = json.loads(run_file.read_text())
                    all_run_labels.append(run_data["pred_labels"])
                    all_run_texts.append(run_data["generated_text"])
                else:
                    labels, gen_text = run_single(client, cfg.model_id, prompt, image_b64, cfg)
                    run_data = {
                        "id": sample_id,
                        "run": run_idx,
                        "pred_labels": labels,
                        "generated_text": gen_text,
                    }
                    run_file.write_text(json.dumps(run_data, ensure_ascii=False, indent=2))
                    all_run_labels.append(labels)
                    all_run_texts.append(gen_text)

            n_runs = len(all_run_labels)
            label_counts = Counter()
            for run_labels in all_run_labels:
                for lbl in run_labels:
                    if lbl in UNIQUE_LABELS:
                        label_counts[lbl] += 1

            consistency_confidence = {
                lbl: label_counts.get(lbl, 0) / n_runs
                for lbl in UNIQUE_LABELS
            }

            # Majority vote: labels appearing in >50% of runs.
            pred_labels = [lbl for lbl, freq in consistency_confidence.items() if freq > 0.5]
            if not pred_labels:
                pred_labels = ["NO_PROPAGANDA"]

            record = {
                "id": sample_id,
                "gold_labels": item.get(cfg.label_column, []),
                "pred_labels": pred_labels,
                "consistency_confidence": consistency_confidence,
                "label_counts": dict(label_counts),
                "n_runs": n_runs,
                "per_run_labels": all_run_labels,
            }

            agg_file.write_text(json.dumps(record, ensure_ascii=False, indent=2))
            manifest.append({"id": sample_id, "path": str(agg_file)})

        except Exception as e:
            print(f"  FAIL {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            failures.append({"id": sample_id, "error": str(e)})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    print(f"Done: {len(manifest)} cached, {len(failures)} failures")
    return manifest


def parse_args():
    p = argparse.ArgumentParser(description="Consistency-based calibration via repeated sampling (vLLM)")
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--language", default="uk", choices=["uk", "en"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--label_column", default="labels")
    p.add_argument("--text_column", default="text")
    p.add_argument("--image_column", default="image")
    p.add_argument("--max_new_tokens", type=int, default=600)
    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--api_base", default="http://localhost:8000/v1")
    return p.parse_args()


def main():
    cfg = parse_args()

    np.random.seed(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "run_config.json").write_text(json.dumps(vars(cfg), indent=2))

    dataset_path = Path(cfg.dataset_path)
    ann_path = dataset_path / "annotations" / f"{cfg.split}.jsonl"
    dataset = load_jsonl(ann_path)
    img_dir = dataset_path / "images"
    print(f"Loaded {len(dataset)} samples from {ann_path}")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), base_url=cfg.api_base, timeout=3600)
    print(f"Using vLLM at {cfg.api_base}, model={cfg.model_id}")

    run_consistency(dataset, img_dir, out_dir, client, cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
