"""Parse verbalized explanations (`referred_words`, `referred_image_concepts`) from inference cache.

Normalises tokens, deduplicates, applies stopword filter, caps at K. Writes one JSON list per mode.

Usage:
    python run_parse_explanations.py \\
        --inference_cache_dir ./data_trans_qwen4/inference_cache \\
        --output_dir ./data_trans_qwen4/vlm_explanations \\
        --modes text image image+text
"""
import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm


STOPWORDS_EN = {
    "a", "an", "the", "of", "in", "on", "at", "to", "with", "and", "or",
    "is", "are", "was", "be", "this", "that", "it", "its", "by", "for",
}
STOPWORDS_UK = {
    "я", "ти", "ви", "він", "вона", "вони", "ми", "це", "те", "ті",
    "мене", "мені", "мій", "моя", "себе", "твоя", "вас", "вам",
    "їх", "наші", "ваші", "всі", "усі", "всіх", "тими", "ній",
    "не", "ні", "за", "на", "що", "так", "але", "та", "бо", "і",
    "щоб", "аби", "якщо", "коли", "чи", "то", "ще", "вже", "теж",
    "від", "для", "до", "в", "у", "з", "по", "без", "через", "перед",
    "е", "с", "нам", "наше",
}

# Notebook ran with effectively empty stopword set (whitespace-only); preserve that default.
DEFAULT_STOPWORDS = {" "}


def normalize_word(word):
    word = re.sub(r"^\W+|\W+$", "", str(word), flags=re.UNICODE)
    return word.lower().strip()


def clean_word_list(words, stopwords, max_k=10):
    """Normalise, deduplicate, drop stopwords, return up to max_k words."""
    seen, out = set(), []
    for w in (words or []):
        n = normalize_word(w)
        if n and n not in stopwords and n not in seen and len(n) > 1:
            seen.add(n)
            out.append(n)
        if len(out) >= max_k:
            break
    return out


def parse_self_reported(mode, cache_root, out_root, stopwords, max_k):
    mode_key = mode.replace("+", "_")
    cache_dir = cache_root / mode_key
    out_dir = out_root / mode_key
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((cache_dir / "manifest.json").read_text())
    results = []
    empty_count = 0

    for entry in tqdm(manifest, desc=f"Parse self-report [{mode}]"):
        record = json.loads(Path(entry["path"]).read_text())

        referred_words = clean_word_list(record.get("referred_words", []), stopwords, max_k)
        referred_concepts = clean_word_list(record.get("referred_image_concepts", []), stopwords, max_k)

        if not referred_words and not referred_concepts:
            empty_count += 1

        results.append({
            "id":                          record["id"],
            "mode":                        mode,
            "gold_labels":                 record.get("gold_labels", []),
            "pred_labels":                 record.get("pred_labels", []),
            "referred_words_raw":          record.get("referred_words", []),
            "referred_words":              referred_words,
            "referred_image_concepts_raw": record.get("referred_image_concepts", []),
            "referred_image_concepts":     referred_concepts,
            "parse_ok":                    record.get("parse_ok", False),
        })

    (out_dir / "self_reported.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )
    print(f"[{mode}] {len(results)} records saved, {empty_count} with empty explanations")


def parse_args():
    p = argparse.ArgumentParser(description="Parse verbalized explanations from inference cache")
    p.add_argument("--inference_cache_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--modes", nargs="+", default=["text", "image", "image+text"])
    p.add_argument("--max_k", type=int, default=10)
    p.add_argument(
        "--stopwords",
        choices=["none", "en", "uk", "both"],
        default="none",
        help="Stopword set to drop. 'none' matches the notebook default (only whitespace dropped).",
    )
    return p.parse_args()


def resolve_stopwords(choice):
    if choice == "en":
        return STOPWORDS_EN | DEFAULT_STOPWORDS
    if choice == "uk":
        return STOPWORDS_UK | DEFAULT_STOPWORDS
    if choice == "both":
        return STOPWORDS_EN | STOPWORDS_UK | DEFAULT_STOPWORDS
    return set(DEFAULT_STOPWORDS)


def main():
    cfg = parse_args()
    cache_root = Path(cfg.inference_cache_dir)
    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run_config.json").write_text(json.dumps(vars(cfg), indent=2, default=str))

    stopwords = resolve_stopwords(cfg.stopwords)

    for mode in cfg.modes:
        parse_self_reported(mode, cache_root, out_root, stopwords, cfg.max_k)


if __name__ == "__main__":
    main()
