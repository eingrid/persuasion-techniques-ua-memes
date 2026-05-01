#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/../.venv/bin/python"
RUN="$SCRIPT_DIR/run_calibration.py"

API_BASE="http://localhost:8000/v1"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# start_vllm <model_id> [extra vllm args...]
start_vllm() {
    local model_id="$1"
    shift
    log "Starting vLLM server for $model_id ..."
    "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$model_id" \
        --port 8000 \
        "$@" &
    VLLM_PID=$!

    log "Waiting for vLLM server (PID=$VLLM_PID) ..."
    for i in $(seq 1 120); do
        if curl -s "$API_BASE/models" > /dev/null 2>&1; then
            log "vLLM server ready."
            return 0
        fi
        sleep 2
    done
    log "ERROR: vLLM server did not start within 240s"
    kill "$VLLM_PID" 2>/dev/null || true
    exit 1
}

stop_vllm() {
    if [ -n "${VLLM_PID:-}" ]; then
        log "Stopping vLLM server (PID=$VLLM_PID) ..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        unset VLLM_PID
        sleep 5
    fi
}

trap stop_vllm EXIT

# Flag sets --------------------------------------------------------------------
QWEN_ARGS=(
    --dtype bfloat16
    --limit-mm-per-prompt.video 0
    --async-scheduling
    --gpu-memory-utilization 0.85
    --max-num-seqs 128
    --max-model-len 5000
)

GEMMA_ARGS=(
    --dtype bfloat16
    # --limit-mm-per-prompt audio=0
    --gpu-memory-utilization 0.85
    --max-num-seqs 64
    --max-model-len 5000
)

# # ── 4B ───────────────────────────────────────────────────────────────────────
# start_vllm "Qwen/Qwen3-VL-4B-Instruct" "${QWEN_ARGS[@]}"

# log "=== 4B | translated (uk) ==="
# "$PYTHON" "$RUN" \
#     --model_id   "Qwen/Qwen3-VL-4B-Instruct" \
#     --dataset_path "$SCRIPT_DIR/../datasets/translated/" \
#     --language   uk \
#     --output_dir "$SCRIPT_DIR/cache_calibration_translated" \
#     --modes text image "image+text" \
#     --api_base   "$API_BASE"

# log "=== 4B | ukrainian (uk) ==="
# "$PYTHON" "$RUN" \
#     --model_id   "Qwen/Qwen3-VL-4B-Instruct" \
#     --dataset_path "$SCRIPT_DIR/../datasets/ukrainian/" \
#     --language   uk \
#     --output_dir "$SCRIPT_DIR/cache_calibration_ukrainian" \
#     --modes text image "image+text" \
#     --api_base   "$API_BASE"

# log "=== 4B | propaganda_950 (en) ==="
# "$PYTHON" "$RUN" \
#     --model_id   "Qwen/Qwen3-VL-4B-Instruct" \
#     --dataset_path "/home/nazara/Data2/DIPLOMA/datasets/propaganda_950/" \
#     --language   en \
#     --output_dir "$SCRIPT_DIR/cache_calibration_propaganda950" \
#     --modes text image "image+text" \
#     --api_base   "$API_BASE"

# stop_vllm

# # ── 8B ───────────────────────────────────────────────────────────────────────
# start_vllm "Qwen/Qwen3-VL-8B-Instruct" "${QWEN_ARGS[@]}"

# log "=== 8B | translated (uk) ==="
# "$PYTHON" "$RUN" \
#     --model_id   "Qwen/Qwen3-VL-8B-Instruct" \
#     --dataset_path "$SCRIPT_DIR/../datasets/translated/" \
#     --language   uk \
#     --output_dir "$SCRIPT_DIR/cache_calibration_qwen8b_translated" \
#     --modes text image "image+text" \
#     --api_base   "$API_BASE"

# log "=== 8B | ukrainian (uk) ==="
# "$PYTHON" "$RUN" \
#     --model_id   "Qwen/Qwen3-VL-8B-Instruct" \
#     --dataset_path "$SCRIPT_DIR/../datasets/ukrainian/" \
#     --language   uk \
#     --output_dir "$SCRIPT_DIR/cache_calibration_qwen8b_ukrainian" \
#     --modes text image "image+text" \
#     --api_base   "$API_BASE"

# log "=== 8B | propaganda_950 (en) ==="
# "$PYTHON" "$RUN" \
#     --model_id   "Qwen/Qwen3-VL-8B-Instruct" \
#     --dataset_path "/home/nazara/Data2/DIPLOMA/datasets/propaganda_950/" \
#     --language   en \
#     --output_dir "$SCRIPT_DIR/cache_calibration_qwen8b_propaganda950" \
#     --modes text image "image+text" \
#     --api_base   "$API_BASE"

# stop_vllm

# ── Gemma 3 4B ───────────────────────────────────────────────────────────────
# start_vllm "google/gemma-3-4b-it" "${GEMMA_ARGS[@]}"

# log "=== Gemma 3 4B | propaganda_950 (en) ==="
# "$PYTHON" "$RUN" \
#     --model_id   "google/gemma-3-4b-it" \
#     --dataset_path "/home/nazara/Data2/DIPLOMA/datasets/propaganda_950/" \
#     --language   en \
#     --output_dir "$SCRIPT_DIR/cache_calibration_gemma3_4b_propaganda950" \
#     --modes image "image+text" text \
#     --api_base   "$API_BASE"

# log "=== Gemma 3 4B | translated (uk) ==="
# "$PYTHON" "$RUN" \
#     --model_id   "google/gemma-3-4b-it" \
#     --dataset_path "$SCRIPT_DIR/../datasets/translated/" \
#     --language   uk \
#     --output_dir "$SCRIPT_DIR/cache_calibration_gemma3_4b_translated" \
#     --modes text image "image+text" \
#     --api_base   "$API_BASE"

# log "=== Gemma 3 4B | ukrainian (uk) ==="
# "$PYTHON" "$RUN" \
#     --model_id   "google/gemma-3-4b-it" \
#     --dataset_path "$SCRIPT_DIR/../datasets/ukrainian/" \
#     --language   uk \
#     --output_dir "$SCRIPT_DIR/cache_calibration_gemma3_4b_ukrainian" \
#     --modes text image "image+text" \
#     --api_base   "$API_BASE"

# stop_vllm

# ── GPT-4.1-mini (OpenAI API, no vLLM) ───────────────────────────────────────
OPENAI_BASE="https://api.openai.com/v1"

log "=== GPT-4.1-mini | propaganda_950 (en) ==="
"$PYTHON" "$RUN" \
    --model_id   "gpt-4.1-mini" \
    --dataset_path "/home/nazara/Data2/DIPLOMA/datasets/propaganda_950/" \
    --language   en \
    --output_dir "$SCRIPT_DIR/cache_calibration_gpt41mini_propaganda950" \
    --modes text image "image+text" \
    --api_base   "$OPENAI_BASE"

log "=== GPT-4.1-mini | translated (uk) ==="
"$PYTHON" "$RUN" \
    --model_id   "gpt-4.1-mini" \
    --dataset_path "$SCRIPT_DIR/../datasets/translated/" \
    --language   uk \
    --output_dir "$SCRIPT_DIR/cache_calibration_gpt41mini_translated" \
    --modes text image "image+text" \
    --api_base   "$OPENAI_BASE"

log "=== GPT-4.1-mini | ukrainian (uk) ==="
"$PYTHON" "$RUN" \
    --model_id   "gpt-4.1-mini" \
    --dataset_path "$SCRIPT_DIR/../datasets/ukrainian/" \
    --language   uk \
    --output_dir "$SCRIPT_DIR/cache_calibration_gpt41mini_ukrainian" \
    --modes text image "image+text" \
    --api_base   "$OPENAI_BASE"

log "=== All done ==="