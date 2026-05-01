# Cross-Cultural Propaganda Detection on Ukrainian Memes

Master's thesis code (UCU APPS, 2026): cross-lingual / cross-cultural propaganda-technique detection on Ukrainian memes, with token-level attribution and confidence calibration on Qwen3-VL-4B/8B, Gemma-3-4B and GPT-4.1-mini.

## Layout

| Directory | Purpose |
|-----------|---------|
| `datasets/` | `propaganda_950/` (English source), `translated/` (machine-translated to Ukrainian), `ukrainian/` (PTM-UA + UkrMeme) — annotation JSONLs and meme images. |
| `scrape/` | Telegram + Google-Drive scraper for the Ukrainian dataset (`parse.py`, `t.csv`). |
| `translate/` | OCR + GPT translation + LaMa inpainting pipeline producing `datasets/translated/`. |
| `calibration/` | Verbalised- and consistency-based calibration: vLLM inference, Platt / temperature scaling, ECE / Brier reporting. |
| `explainability/` | Token-level LRP vs verbalised attribution, word-category analysis, low-hF1 disagreement inspection. |
| `src/` | Shared library — `hierarchical_f1.py` (SemEval ancestor-based hF1) and `utils.py`. |
| `compute_metrics.py` | Hierarchical F1 across every cache directory in one run. |
| `annotator_agreement.ipynb` | IAA on the UkrMeme test set (Cohen κ, augmented κ, MASI Krippendorff α, bootstrapped multi-label IAA per Marchal 2022). |

## Install

```sh
uv sync
source .venv/bin/activate
```

Python `3.12` (see `.python-version`).

## Environment variables

Required by some scripts (rotate any keys that ever appeared in git history).

| Variable | Used by |
|----------|---------|
| `OPENAI_API_KEY` | `translate/translate.py`, `explainability/run_processing.py`, `explainability/notebooks/04_*`, `05_*` |
| `TELEGRAM_API_ID`, `TELEGRAM_API_HASH` | `scrape/parse.py` |

## End-to-end pipeline

1. **Scrape** UkrMeme (`scrape/parse.py`) → `datasets/ukrainian/`.
2. **Translate** Propaganda 950 → PTM-UA (`translate/translate.py`) → `datasets/translated/`.
3. **Inference + calibration** (`calibration/run_all.sh`) — verbalised confidence on the full grid (4 models × 3 datasets × 3 modalities); requires a vLLM server.
4. **Explainability** (`explainability/run_all.sh`) — inference → text-LRP → parse explanations → processing (Jaccard, GPT word-categorisation).
5. **Aggregate metrics** (`python compute_metrics.py`) — hF1 over every cache, including a `[UA, NO_PROPAGANDA excluded]` split for UkrMeme.
6. **Analysis notebooks** in `explainability/notebooks/07_analysis.ipynb` and `calibration/{calibration_comparison,post_hot_platt_scaling}.ipynb`.

Per-subdirectory READMEs document the exact script flags.

## Models

- `Qwen/Qwen3-VL-4B-Instruct`, `Qwen/Qwen3-VL-8B-Instruct` (open, served via vLLM or HF transformers)
- `google/gemma-3-4b-it` (calibration only)
- `gpt-4.1-mini` (calibration baseline; calibration code only)
- `gpt-4o` (word categorisation in `run_processing.py`)

## Citation

Andrushko, N. (2026). *Cross-cultural propaganda-technique detection on Ukrainian memes with token-level attribution and confidence calibration*. Master's thesis, UCU APPS.
