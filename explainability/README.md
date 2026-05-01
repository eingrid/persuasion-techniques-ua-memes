# Explainability — Token-Level Attribution

Compares LRP (gradient × input embedding) against the model's own verbalised `referred_words` on Qwen3-VL-4B/8B. Produces the word-category and Jaccard analyses cited in §5.3 of the thesis.

## Layout

```
explainability/
├── run_inference.py           # Qwen3-VL inference, all three modes, caches predictions + verbalised words
├── run_lrp_text.py            # token-level LRP on cached output_ids (lxt + zennit AttnLRP)
├── run_parse_explanations.py  # normalise + dedupe referred_words / referred_image_concepts
├── run_processing.py          # align LRP ↔ LLM, GPT-4o word-categorisation, Jaccard, F1 → parquet
├── run_all.sh                 # 4-stage pipeline driver, 2 models × 3 datasets
├── notebooks/                 # interactive single-cell variants of the scripts + 07_analysis
├── image_branch/              # discontinued visual-LRP / heatmap-projection notebooks (kept for reference)
├── report/                    # finished figures + word-frequency tables used in the thesis
└── data_*/                    # per-cell caches (gitignored)
```

## Pipeline (per `model × dataset` cell)

1. **`run_inference.py`** — Qwen3-VL.from_pretrained → for each mode (`text`, `image`, `image+text`) writes one JSON per sample with `pred_labels`, `referred_words`, `referred_image_concepts`, `input_ids`, `output_ids`.
2. **`run_lrp_text.py`** — patches Qwen3-VL with `lxt.efficient` AttnLRP rules, replays the cached `output_ids`, backprops on the predicted-label logits, writes per-sample `relevance_norm` over prompt tokens.
3. **`run_parse_explanations.py`** — offline; lower-cases, deduplicates and stop-word filters the verbalised word lists.
4. **`run_processing.py`** — projects LRP token relevance back onto whitespace words via tokenizer offset mapping, picks top-K, asks GPT-4o to label every word as `ENTITY / EMOTIONAL / RHETORIC / CONTENT / OTHER`, and writes `<mode>_processed.{csv,parquet}`. Validated 80.3% on a 61-word sample (manual reclassification).

`run_all.sh` chains all four steps for the six cells in the thesis grid (2 Qwen sizes × 3 datasets). UkrMeme uses `--include-no-propaganda`; Propaganda 950 and PTM-UA use `--no-include-no-propaganda` (the `NO_PROPAGANDA` label and clause are never advertised to the model when the dataset has no abstention class).

## Single-cell run

```sh
.venv/bin/python explainability/run_inference.py \
    --model_id Qwen/Qwen3-VL-4B-Instruct \
    --dataset_path datasets/translated/ \
    --language uk \
    --output_dir explainability/data_trans_qwen4/inference_cache \
    --no-include-no-propaganda

.venv/bin/python explainability/run_lrp_text.py \
    --model_id Qwen/Qwen3-VL-4B-Instruct \
    --inference_cache_dir explainability/data_trans_qwen4/inference_cache \
    --output_dir explainability/data_trans_qwen4/lrp_results

.venv/bin/python explainability/run_parse_explanations.py \
    --inference_cache_dir explainability/data_trans_qwen4/inference_cache \
    --output_dir explainability/data_trans_qwen4/vlm_explanations

OPENAI_API_KEY=... .venv/bin/python explainability/run_processing.py \
    --model_id Qwen/Qwen3-VL-4B-Instruct \
    --dataset_path datasets/translated/ \
    --language uk \
    --inference_cache_dir explainability/data_trans_qwen4/inference_cache \
    --lrp_results_dir explainability/data_trans_qwen4/lrp_results \
    --vlm_explanations_dir explainability/data_trans_qwen4/vlm_explanations \
    --output_dir explainability/data_trans_qwen4/processed
```

## Notebooks

| Notebook | Role |
|----------|------|
| `notebooks/01_inference.ipynb` | Interactive demo of `run_inference.py`. |
| `notebooks/02_lrp_text.ipynb` | Interactive demo of `run_lrp_text.py`; useful for inspecting individual samples. |
| `notebooks/04_vlm_explanations.ipynb` | Interactive demo of `run_parse_explanations.py`. |
| `notebooks/05_processing.ipynb` | Interactive demo of `run_processing.py`. |
| `notebooks/07_analysis.ipynb` | Reads the per-mode parquet and produces the word-category bars + low-Jaccard disagreement table for §5.3. |

## Image branch (discontinued)

`image_branch/` keeps `03_lrp_visual.ipynb`, `06_connecting_lrp_and_llm.ipynb`, `06b_visualization.ipynb` for transparency. Image-level attribution was abandoned because object-level verbalisations were too vague / repetitive at the tested model scales (see §6.2 of the thesis); these notebooks are not part of the reproducible pipeline.
