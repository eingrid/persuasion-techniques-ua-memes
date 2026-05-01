"""Consolidate LRP + verbalized explanations into one row per sample.

Per (model, dataset, mode) reads:
- inference cache (gold/pred labels, OCR text)
- LRP text relevance (run_lrp_text.py output)
- verbalized explanations (run_parse_explanations.py output)

Computes per-sample F1, top-K LRP words via tokenizer offset mapping, GPT word-categorization
(ENTITY / EMOTIONAL / RHETORIC / CONTENT / OTHER), and Jaccard between LRP and LLM word sets.

Writes `<output_dir>/<mode>_processed.{csv,parquet}`.

Usage:
    python run_processing.py \\
        --model_id Qwen/Qwen3-VL-4B-Instruct \\
        --dataset_path ../datasets/translated/ \\
        --language uk \\
        --inference_cache_dir ./data_trans_qwen4/inference_cache \\
        --lrp_results_dir ./data_trans_qwen4/lrp_results \\
        --vlm_explanations_dir ./data_trans_qwen4/vlm_explanations \\
        --output_dir ./data_trans_qwen4/processed \\
        --modes text image+text
"""
import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from json_repair import repair_json
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import AutoProcessor


STOPWORDS_EN = {
    "a", "an", "the", "of", "in", "on", "at", "to", "with", "and", "or",
    "is", "are", "was", "be", "this", "that", "it", "its", "by", "for",
}
STOPWORDS_UK = {
    "я", "ти", "ви", "він", "вона", "вони", "ми", "це", "те", "ті", "мене", "мені", "мій", "моя",
    "не", "ні", "за", "на", "що", "так", "але", "та", "бо", "і", "щоб", "аби", "якщо", "коли", "чи",
    "то", "ще", "вже", "теж", "від", "для", "до", "в", "у", "з", "по", "без", "через", "перед",
    "е", "с", "нам", "наше",
}
STOPWORDS = STOPWORDS_EN | STOPWORDS_UK


LANG_NAME = {"uk": "Ukrainian", "en": "English"}

CATEGORIZE_PROMPT = """\
You are analyzing memes. The meme text is in {lang_name}. Categorize words in context.

Meme text: "{meme_text}"
Predicted persuasion techniques: {pred_labels}

The following words were flagged as important in this meme by automated methods:
{word_list}

For each word, assign ONE category based on its role in THIS specific meme:
- ENTITY: named people, places, organizations, brands
- EMOTIONAL: emotionally loaded, fear/hate/outrage-inducing language
- RHETORIC: calls to action, imperatives, slogans, persuasive framing
- CONTENT: neutral factual nouns, events, dates, statistics
- OTHER: numbers alone, punctuation artifacts, unclear

Return ONLY a JSON object mapping each word to its category.
No markdown, no explanation.
Example: {{"word1": "ENTITY", "word2": "EMOTIONAL", "word3": "CONTENT"}}
"""


def normalize_word(word):
    return re.sub(r"^\W+|\W+$", "", str(word), flags=re.UNICODE).lower().strip()


def safe_set(words, top_k):
    seen, out = set(), []
    for w in (words or []):
        n = normalize_word(w)
        if n and n not in STOPWORDS and n not in seen and len(n) > 1:
            seen.add(n)
            out.append(n)
        if len(out) >= top_k:
            break
    return set(out)


def multilabel_f1(gold_labels, pred_labels):
    gold = set(gold_labels or [])
    pred = set(pred_labels or [])
    denom = len(gold) + len(pred)
    if denom == 0:
        return 1.0
    return 2.0 * len(gold & pred) / denom


def get_lrp_top_words(lrp_record, ocr_text, n_words, tokenizer):
    """Top-`n_words` words by mean abs LRP relevance over tokens overlapping the OCR span."""
    relevance_norm = lrp_record.get("relevance_norm", [])
    input_len = lrp_record.get("input_len", 0)
    token_ids = lrp_record.get("token_ids", [])

    if not ocr_text.strip() or not relevance_norm:
        return set()

    prompt_ids = token_ids[:input_len]
    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    ocr_start_char = prompt_text.rfind(ocr_text)
    if ocr_start_char < 0:
        return set()
    ocr_end_char = ocr_start_char + len(ocr_text)

    enc = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    ocr_token_indices = [
        i for i, (ts, te) in enumerate(offsets)
        if ts < ocr_end_char and te > ocr_start_char and ts < te and i < len(relevance_norm)
    ]
    if not ocr_token_indices:
        return set()

    abs_rel = [abs(float(relevance_norm[i])) for i in ocr_token_indices]
    word_spans = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", ocr_text)]

    word_rel = {}
    for i, idx in enumerate(ocr_token_indices):
        tok_start, tok_end = offsets[idx]
        local_start = max(tok_start, ocr_start_char) - ocr_start_char
        local_end = min(tok_end, ocr_end_char) - ocr_start_char
        local_mid = (local_start + local_end - 1) // 2
        if local_mid < 0 or local_mid >= len(ocr_text):
            continue
        for w_start, w_end, word in word_spans:
            if w_start <= local_mid < w_end:
                word_rel.setdefault(word, []).append(abs_rel[i])
                break

    scored = sorted(
        ((normalize_word(w), float(np.mean(v))) for w, v in word_rel.items()),
        key=lambda x: x[1], reverse=True,
    )
    return set([w for w, _ in scored if w and w not in STOPWORDS and len(w) > 1][:n_words])


async def categorize_words(client, model, language, sample_id, meme_text, pred_labels, all_words, semaphore):
    if not all_words:
        return {"id": sample_id, "word_categories": {}}

    prompt = CATEGORIZE_PROMPT.format(
        lang_name=LANG_NAME.get(language, "Ukrainian"),
        meme_text=meme_text,
        pred_labels=", ".join(pred_labels),
        word_list=", ".join(sorted(all_words)),
    )

    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a linguist. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_completion_tokens=300,
            )
            raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
            return {"id": sample_id, "word_categories": json.loads(repair_json(raw))}
        except Exception as e:
            print(f"  Categorize error [{sample_id}]: {e}")
            return {"id": sample_id, "word_categories": {}, "error": str(e)}


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def load_lrp_records(lrp_text_dir):
    records = {}
    if not lrp_text_dir.exists():
        return records
    for p in lrp_text_dir.glob("*.json"):
        if p.name == "failures.json":
            continue
        try:
            r = json.loads(p.read_text())
            records[r["id"]] = r
        except Exception:
            pass
    return records


async def process_mode(mode, cfg, id_to_text, tokenizer, client):
    mode_key = mode.replace("+", "_")
    expl_file = Path(cfg.vlm_explanations_dir) / mode_key / "self_reported.json"
    lrp_text_dir = Path(cfg.lrp_results_dir) / f"text_{mode_key}"
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    explanations = json.loads(expl_file.read_text())
    lrp_records = load_lrp_records(lrp_text_dir)

    semaphore = asyncio.Semaphore(cfg.max_concurrent)
    cat_tasks = []
    sample_words = {}

    for rec in explanations:
        sid = rec["id"]
        meme_text = id_to_text.get(sid, "")
        pred_labels = rec.get("pred_labels", [])
        llm_words = safe_set(rec.get("referred_words", []), top_k=cfg.top_k_words)

        lrp_rec = lrp_records.get(sid)
        if lrp_rec and meme_text and llm_words:
            lrp_words = get_lrp_top_words(lrp_rec, meme_text, n_words=len(llm_words), tokenizer=tokenizer)
        else:
            lrp_words = set()

        sample_words[sid] = {"llm": llm_words, "lrp": lrp_words}
        cat_tasks.append(categorize_words(
            client, cfg.gpt_model, cfg.language, sid, meme_text, pred_labels,
            llm_words | lrp_words, semaphore,
        ))

    print(f"[{mode}] Categorizing words for {len(cat_tasks)} samples...")
    cat_results = await asyncio.gather(*cat_tasks)
    id_to_cats = {r["id"]: r["word_categories"] for r in cat_results}

    rows = []
    for rec in tqdm(explanations, desc=f"Build rows [{mode}]"):
        sid = rec["id"]
        meme_text = id_to_text.get(sid, "")
        gold_labels = rec.get("gold_labels", [])
        pred_labels = rec.get("pred_labels", [])

        llm_words = sample_words[sid]["llm"]
        lrp_words = sample_words[sid]["lrp"]
        word_cats = id_to_cats.get(sid, {})

        inter = llm_words & lrp_words
        union = llm_words | lrp_words
        jaccard = len(inter) / len(union) if union else 0.0

        def count_cat(words, cat):
            return sum(1 for w in words if word_cats.get(w) == cat)

        rows.append({
            "sample_id":    sid,
            "mode":         mode,
            "f1":           multilabel_f1(gold_labels, pred_labels),
            "exact_match":  int(set(gold_labels) == set(pred_labels)),
            "is_trivial":   int(len(gold_labels) == 0 and len(pred_labels) == 0),
            "n_gold_labels": len(set(gold_labels)),
            "n_pred_labels": len(set(pred_labels)),
            "techniques":   "|".join(sorted(set(gold_labels))),
            "text_length":  len(meme_text.split()),
            "has_lrp_text": int(sid in lrp_records),
            "n_llm_words":  len(llm_words),
            "llm_words":    list(llm_words),
            "lrp_words":    list(lrp_words),
            "word_categories": word_cats,
            "jaccard":      jaccard,
            "llm_entity":    count_cat(llm_words, "ENTITY"),
            "llm_emotional": count_cat(llm_words, "EMOTIONAL"),
            "llm_rhetoric":  count_cat(llm_words, "RHETORIC"),
            "llm_content":   count_cat(llm_words, "CONTENT"),
            "lrp_entity":    count_cat(lrp_words, "ENTITY"),
            "lrp_emotional": count_cat(lrp_words, "EMOTIONAL"),
            "lrp_rhetoric":  count_cat(lrp_words, "RHETORIC"),
            "lrp_content":   count_cat(lrp_words, "CONTENT"),
        })

    df = pd.DataFrame(rows)
    out_csv = out_dir / f"{mode_key}_processed.csv"
    out_parquet = out_dir / f"{mode_key}_processed.parquet"
    df.to_csv(out_csv, index=False)
    df.to_parquet(out_parquet, index=False)
    print(f"[{mode}] Saved {len(df)} rows → {out_parquet}")


def parse_args():
    p = argparse.ArgumentParser(description="Consolidate LRP + verbalized data into per-mode parquet")
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-4B-Instruct",
                   help="Tokenizer source; must match the model used by run_inference.py / run_lrp_text.py.")
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--language", default="uk", choices=["uk", "en"])
    p.add_argument("--split", default="test")
    p.add_argument("--inference_cache_dir", required=True)
    p.add_argument("--lrp_results_dir", required=True)
    p.add_argument("--vlm_explanations_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--modes", nargs="+", default=["text", "image+text"])
    p.add_argument("--top_k_words", type=int, default=5)
    p.add_argument("--gpt_model", default="gpt-4o")
    p.add_argument("--max_concurrent", type=int, default=20)
    return p.parse_args()


async def amain():
    cfg = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable must be set.")
    client = AsyncOpenAI(api_key=api_key)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.output_dir) / "run_config.json").write_text(json.dumps(vars(cfg), indent=2, default=str))

    dataset = load_jsonl(Path(cfg.dataset_path) / "annotations" / f"{cfg.split}.jsonl")
    id_to_text = {item["id"]: item.get("text", "") for item in dataset}
    print(f"Loaded {len(dataset)} samples")

    tokenizer = AutoProcessor.from_pretrained(cfg.model_id).tokenizer

    for mode in cfg.modes:
        await process_mode(mode, cfg, id_to_text, tokenizer, client)


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
