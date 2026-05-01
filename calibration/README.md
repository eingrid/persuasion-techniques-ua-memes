# Calibration

Verbalised- and consistency-based confidence calibration on Qwen3-VL-4B/8B, Gemma-3-4B and GPT-4.1-mini across Propaganda 950 / PTM-UA / UkrMeme × {`text`, `image`, `image+text`}. Produces the ECE / Brier numbers and Platt-scaling figures in §5.4 of the thesis.

## Layout

```
calibration/
├── run_calibration.py            # vLLM client; per-label verbalised confidence inference
├── run_consistency.py            # repeated-sampling inference; label frequency = confidence
├── run_evaluation.py             # full grid: ECE / Brier / Platt / temperature / cross-dataset transfer → results/
├── run_all.sh                    # grid driver for run_calibration.py
├── run_consistency_all.sh        # grid driver for run_consistency.py
├── calibration_metrics.py        # ECE, MCE, ACE, Brier, log-loss, reliability diagrams
├── loader.py                     # uniform Cell loader for self-reported + consistency caches
├── reliability_plots.py          # grouped-bar reliability figures
├── ece_heatmap.py                # ECE heatmap across model × dataset × modality
├── confidence_distribution.py    # confidence histograms
├── label_distribution.py         # gold-label frequency plots
├── calibration_comparison.ipynb  # cross-dataset / cross-model ECE analysis (Q2, Q3)
├── post_hot_platt_scaling.ipynb  # raw vs temperature vs Platt; cross-dataset Platt transfer
├── cache_self_reported/          # verbalised inference caches (gitignored)
├── consistency_calibration/      # repeated-sampling caches (gitignored)
└── results/                      # generated figures + calibration.{txt,csv,json} (gitignored)
```

## Inference (vLLM)

`run_calibration.py` and `run_consistency.py` talk to a local vLLM OpenAI-compatible server. Start it before running either script:

```sh
vllm serve Qwen/Qwen3-VL-4B-Instruct \
    --dtype bfloat16 \
    --limit-mm-per-prompt.video 0
```

Then per cell:

```sh
.venv/bin/python calibration/run_calibration.py \
    --model_id Qwen/Qwen3-VL-4B-Instruct \
    --dataset_path datasets/translated/ \
    --language uk \
    --output_dir calibration/cache_self_reported/cache_calibration_translated \
    --modes text image image+text
```

`run_all.sh` wraps the full grid (4 models × 3 datasets); it boots vLLM with the right per-model flags. Same shape for `run_consistency_all.sh`.

## Aggregation + analysis

After all caches are populated:

```sh
.venv/bin/python calibration/run_evaluation.py
```

writes `results/calibration.{txt,csv,json}` with per-cell macro/micro ECE, ACE, Brier, fitted Platt parameters and cross-dataset Platt transfer metrics.

The two notebooks render the figures: `calibration_comparison.ipynb` for cross-dataset / model-size questions, and `post_hot_platt_scaling.ipynb` for the raw → temperature → Platt comparison plus the cross-dataset transfer heatmap.

## Environment

- vLLM server reachable at `http://localhost:8000/v1` (override with `--api_base`).
- For GPT-4.1-mini cells, `OPENAI_API_KEY` (the same vLLM-style client is reused with the real OpenAI base URL).
