"""Qwen3-VL inference for all three modes (text, image, image+text).

Usage:
    python run_inference.py \\
        --model_id Qwen/Qwen3-VL-4B-Instruct \\
        --dataset_path ../datasets/translated/ \\
        --language uk \\
        --output_dir ./data_trans_qwen4/inference_cache \\
        --modes text image image+text \\
        --split test
"""
import argparse
import json
import random
import traceback
from pathlib import Path

import numpy as np
import torch
from json_repair import repair_json
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.models.qwen3_vl import modeling_qwen3_vl


PROPAGANDA_LABELS = [
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
]
NO_PROP_CLAUSE = ', or "NO_PROPAGANDA" if the content doesn\'t contain persuasion techniques'

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
As output, provide JSON with three fields: "labels", "referred_words", and "referred_image_concepts".
- "labels": list of persuasion techniques used{no_prop_clause}.
- "referred_words": list of the most important INDIVIDUAL words from the content text supporting your classification. Only single words, not phrases. Only key words most relevant to the classification.
- "referred_image_concepts": up to 5 visual elements most strongly supporting your classification, ranked by importance. No duplicates. Empty list if no image. The list could be empty if the image doesn't provide any clues.
Visual elements rules:
- Be concrete: name actual people, objects, symbols. Do NOT write abstract scene descriptions like "people in a stressful situation".
- For visual elements: use the shortest identifying phrase — "Obama"/"Обама", "flag"/"прапор". Do not use words on the image themselves.
Language note: {lang_note}
JSON format output example:
{{
    "labels": ["label1", "label2"],
    "referred_words": ["word1", "word2"],
    "referred_image_concepts": ["concept1", "concept2"]
}}
Text from the content: {text}
Output:
'''

IMAGE_ONLY_PROMPT = '''You are an expert in analyzing communication strategies in visual media.
Classify the provided content IMAGE into one or multiple of the following persuasion techniques :
{unique_labels}
As output, provide JSON with two fields: "labels" and "referred_image_concepts".
- "labels": list of persuasion techniques used{no_prop_clause}.
- "referred_image_concepts": up to 5 visual elements most strongly supporting your classification, ranked by importance. No duplicates. The list could be empty if the image doesn't provide any clues.
Visual elements rules:
- Be concrete: name actual people, objects, symbols. Do NOT write abstract scene descriptions like "people in a stressful situation".
- For words/text visible in the image: do not include ANY of them to referred_image_concepts, as they belong to the text modality. Only visual elements should be included.
- For visual elements: use the shortest identifying phrase — "Obama"/"Обама", "flag"/"прапор". Do not use words on the image themselves.
Language note: {lang_note}
JSON format output example:
{{
    "labels": ["label1", "label2"],
    "referred_image_concepts": ["concept1", "concept2"]
}}
Output:
'''


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_messages(mode, language, include_no_prop, ocr_text=""):
    labels = PROPAGANDA_LABELS + (["NO_PROPAGANDA"] if include_no_prop else [])
    no_prop_clause = NO_PROP_CLAUSE if include_no_prop else ""

    if mode == "image":
        prompt = IMAGE_ONLY_PROMPT.format(
            unique_labels=labels,
            lang_note=LANG_INSTRUCTION_IMAGE_ONLY.get(language, ""),
            no_prop_clause=no_prop_clause,
        )
    else:
        prompt = BASE_PROMPT.format(
            unique_labels=labels,
            lang_note=LANG_INSTRUCTION.get(language, ""),
            no_prop_clause=no_prop_clause,
            text=ocr_text,
        )

    content = []
    if mode in ("image", "image+text"):
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def run_inference(mode, dataset, img_dir, out_dir, model, processor, cfg):
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

            messages = build_messages(mode, cfg.language, cfg.include_no_propaganda, ocr_text)
            chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if mode in ("image", "image+text"):
                image = Image.open(image_path).convert("RGB").resize(cfg.image_size)
                inputs = processor(text=[chat_text], images=[image], padding=True, return_tensors="pt").to("cuda")
            else:
                inputs = processor(text=[chat_text], padding=True, return_tensors="pt").to("cuda")

            with torch.no_grad():
                gen_kwargs = dict(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=False,
                )
                if mode in ("image", "image+text"):
                    gen_kwargs["pixel_values"] = inputs.pixel_values
                    gen_kwargs["image_grid_thw"] = inputs.image_grid_thw
                output_ids = model.generate(**gen_kwargs)

            input_len = inputs.input_ids.shape[1]
            gen_text = processor.decode(output_ids[0, input_len:], skip_special_tokens=True)
            try:
                parsed = json.loads(repair_json(gen_text))
            except Exception:
                parsed = {}

            record = {
                "id":              sample_id,
                "mode":            mode,
                "gold_labels":     item.get(cfg.label_column, []),
                "pred_labels":     parsed.get("labels", []),
                "referred_words":  parsed.get("referred_words", []),
                "referred_image_concepts": parsed.get("referred_image_concepts", []),
                "generated_text":  gen_text,
                "input_ids":       inputs.input_ids[0].tolist(),
                "output_ids":      output_ids[0].tolist(),
                "input_len":       input_len,
                "image_grid_thw":  inputs.image_grid_thw[0].tolist() if mode in ("image", "image+text") else None,
                "parse_ok":        bool(parsed.get("labels")),
            }
            cache_file.write_text(json.dumps(record, ensure_ascii=False, indent=2))
            manifest.append({"id": sample_id, "path": str(cache_file)})

        except Exception as e:
            print(f"  FAIL {sample_id}: {e}")
            traceback.print_exc()
            failures.append({"id": sample_id, "error": str(e)})
        finally:
            torch.cuda.empty_cache()

    (mode_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (mode_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    print(f"Mode '{mode}': {len(manifest)} cached, {len(failures)} failures")


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-VL self-reported inference grid")
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--language", default="uk", choices=["uk", "en"])
    p.add_argument("--split", default="test")
    p.add_argument("--label_column", default="labels")
    p.add_argument("--text_column", default="text")
    p.add_argument("--image_column", default="image")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--modes", nargs="+", default=["text", "image", "image+text"])
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--image_size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--include_no_propaganda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include NO_PROPAGANDA in label list and prompt clause (use --no-include-no-propaganda to disable).",
    )
    return p.parse_args()


def main():
    cfg = parse_args()
    cfg.image_size = tuple(cfg.image_size)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(vars(cfg), indent=2, default=str))

    dataset_path = Path(cfg.dataset_path)
    ann_path = dataset_path / "annotations" / f"{cfg.split}.jsonl"
    dataset = load_jsonl(ann_path)
    img_dir = dataset_path / "images"
    print(f"Loaded {len(dataset)} samples from {ann_path}")

    model = modeling_qwen3_vl.Qwen3VLForConditionalGeneration.from_pretrained(
        cfg.model_id, device_map="cuda", dtype=torch.bfloat16,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    print(f"Model loaded: {cfg.model_id}")

    for mode in cfg.modes:
        run_inference(mode, dataset, img_dir, out_dir, model, processor, cfg)


if __name__ == "__main__":
    main()
