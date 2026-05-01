#!/usr/bin/env bash
# Full text-explainability pipeline per (model × dataset):
#   1. inference   → cache labels + referred_words / referred_image_concepts
#   2. lrp_text    → token-level LRP relevance from cached output_ids
#   3. parse_expl  → normalise & dedupe verbalized words
#   4. processing  → align LRP ↔ verbalized, GPT word categorisation, jaccard, F1
#
# Stage 4 calls OpenAI; export OPENAI_API_KEY first.
# Steps 2–4 read directly from the cache produced by step 1, so they're cheap to re-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"

INFERENCE_PY="$SCRIPT_DIR/run_inference.py"
LRP_TEXT_PY="$SCRIPT_DIR/run_lrp_text.py"
PARSE_EXPL_PY="$SCRIPT_DIR/run_parse_explanations.py"
PROCESS_PY="$SCRIPT_DIR/run_processing.py"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_cell() {
    local model_id="$1"
    local dataset_dir="$2"
    local out_dir="$3"
    local lang="$4"
    local no_prop_flag="$5"  # "--include-no-propaganda" or "--no-include-no-propaganda"

    local data_root="$SCRIPT_DIR/${out_dir}"
    local dataset_path="$REPO_ROOT/datasets/${dataset_dir}/"
    local inference_cache="${data_root}/inference_cache"
    local lrp_results="${data_root}/lrp_results"
    local vlm_explanations="${data_root}/vlm_explanations"
    local processed="${data_root}/processed"

    log "=== ${model_id} | ${dataset_dir} | ${no_prop_flag} ==="

    log "  [1/4] inference"
    "$PYTHON" "$INFERENCE_PY" \
        --model_id     "$model_id" \
        --dataset_path "$dataset_path" \
        --language     "$lang" \
        --output_dir   "$inference_cache" \
        --modes        text image "image+text" \
        "$no_prop_flag"

    log "  [2/4] lrp_text"
    "$PYTHON" "$LRP_TEXT_PY" \
        --model_id           "$model_id" \
        --inference_cache_dir "$inference_cache" \
        --output_dir         "$lrp_results" \
        --modes              text "image+text"

    log "  [3/4] parse_explanations"
    "$PYTHON" "$PARSE_EXPL_PY" \
        --inference_cache_dir "$inference_cache" \
        --output_dir         "$vlm_explanations" \
        --modes              text image "image+text"

    log "  [4/4] processing"
    "$PYTHON" "$PROCESS_PY" \
        --model_id            "$model_id" \
        --dataset_path        "$dataset_path" \
        --language            "$lang" \
        --inference_cache_dir "$inference_cache" \
        --lrp_results_dir     "$lrp_results" \
        --vlm_explanations_dir "$vlm_explanations" \
        --output_dir          "$processed" \
        --modes               text "image+text"
}

# UkrMeme has NO_PROPAGANDA gold; translated and PTM-EN do not.
NP_ON="--include-no-propaganda"
NP_OFF="--no-include-no-propaganda"

run_cell "Qwen/Qwen3-VL-4B-Instruct" "translated"     "data_trans_qwen4"         "uk" "$NP_OFF"
run_cell "Qwen/Qwen3-VL-4B-Instruct" "ukrainian"      "data_ukr_qwen"            "uk" "$NP_ON"
run_cell "Qwen/Qwen3-VL-4B-Instruct" "propaganda_950" "data_propaganda950_qwen4" "en" "$NP_OFF"

run_cell "Qwen/Qwen3-VL-8B-Instruct" "translated"     "data_trans_qwen_8"        "uk" "$NP_OFF"
run_cell "Qwen/Qwen3-VL-8B-Instruct" "ukrainian"      "data_ukr_qwen_8"          "uk" "$NP_ON"
run_cell "Qwen/Qwen3-VL-8B-Instruct" "propaganda_950" "data_propaganda950_qwen8" "en" "$NP_OFF"

log "All cells complete."
