#!/bin/bash
# vllm_image_and_text_testing-v2.sh
# Run on: host machine (Terminal 2, while server is running)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model name - change this or pass as argument
MODEL="${1:-Qwen/Qwen3-VL-32B-Instruct-FP8}"
LATEST_VLLM_VERSION="25.12.post1-py3" # Added this variable to prevent script failure

# Other settings
NUM_PROMPTS="${2:-100}"
MAX_CONCURRENCY=10
NUM_WARMUPS=10
INPUT_LEN=1000
OUTPUT_LEN=300
RANGE_RATIO=0.3
SEED=42

# =============================================================================
# DEVICE DETECTION
# =============================================================================

CUDA_DEVICE_RAW=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
if [ -z "$CUDA_DEVICE_RAW" ]; then
    CUDA_DEVICE_NAME="Unknown_GPU"
else
    CUDA_DEVICE_NAME=$(echo "$CUDA_DEVICE_RAW" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/__*/_/g' | sed 's/_$//')
fi

echo "Detected GPU: ${CUDA_DEVICE_RAW}"
echo "Sanitized name: ${CUDA_DEVICE_NAME}"

# =============================================================================
# OUTPUT DIRECTORY SETUP
# =============================================================================

OUTPUT_DIR="./benchmark_results/${CUDA_DEVICE_NAME}"
mkdir -p "${OUTPUT_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MODEL_SHORT=$(echo "${MODEL}" | sed 's/.*\///' | sed 's/[^a-zA-Z0-9]/_/g')
OUTPUT_FILE="${OUTPUT_DIR}/vLLM_${CUDA_DEVICE_NAME}_${MODEL_SHORT}_${TIMESTAMP}.txt"

# =============================================================================
# LOGGING FUNCTION
# =============================================================================

log() {
    echo "$1" | tee -a "${OUTPUT_FILE}"
}

# =============================================================================
# ANSI STRIPPING FUNCTION
# =============================================================================

strip_ansi() {
    # Remove ANSI escape codes from input
    sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' | sed 's/\x1b\][^\x07]*\x07//g' | tr -d '\r'
}

# =============================================================================
# CSV PARSING FUNCTION
# =============================================================================

parse_and_save_csv() {
    local temp_file="$1"
    local num_images="$2"
    local csv_file="$3"
    
    # Create a cleaned version of the temp file (strip ANSI codes)
    local clean_file=$(mktemp)
    cat "$temp_file" | strip_ansi > "$clean_file"
    
    # Debug: show what we're parsing (comment out in production)
    # echo "=== DEBUG: Cleaned output ===" 
    # cat "$clean_file"
    # echo "=== END DEBUG ==="
    
    # Parse metrics - using more flexible patterns
    successful_requests=$(grep -i "Successful requests" "$clean_file" | grep -oE '[0-9]+$' | tail -1)
    failed_requests=$(grep -i "Failed requests" "$clean_file" | grep -oE '[0-9]+$' | tail -1)
    max_concurrency_result=$(grep -i "Maximum request concurrency" "$clean_file" | grep -oE '[0-9.]+$' | tail -1)
    benchmark_duration=$(grep -i "Benchmark duration" "$clean_file" | grep -oE '[0-9.]+' | tail -1)
    total_input_tokens=$(grep -i "Total input tokens" "$clean_file" | grep -oE '[0-9]+$' | tail -1)
    total_generated_tokens=$(grep -i "Total generated tokens" "$clean_file" | grep -oE '[0-9]+$' | tail -1)
    request_throughput=$(grep -i "Request throughput" "$clean_file" | grep -oE '[0-9.]+' | tail -1)
    output_token_throughput=$(grep -i "Output token throughput" "$clean_file" | grep -oE '[0-9.]+' | tail -1)
    peak_output_throughput=$(grep -i "Peak output token throughput" "$clean_file" | grep -oE '[0-9.]+' | tail -1)
    peak_concurrent_requests=$(grep -i "Peak concurrent requests" "$clean_file" | grep -oE '[0-9]+$' | tail -1)
    total_token_throughput=$(grep -i "Total Token throughput" "$clean_file" | grep -oE '[0-9.]+' | tail -1)
    
    ttft_mean=$(grep -i "Mean TTFT" "$clean_file" | grep -oE '[0-9.]+' | tail -1)
    ttft_median=$(grep -i "Median TTFT" "$clean_file" | grep -oE '[0-9.]+' | tail -1)
    ttft_p99=$(grep -i "P99 TTFT" "$clean_file" | grep -oE '[0-9.]+' | tail -1)
    
    tpot_mean=$(grep -i "Mean TPOT" "$clean_file" | grep -oE '[0-9.]+' | tail -1)
    tpot_median=$(grep -i "Median TPOT" "$clean_file" | grep -oE '[0-9.]+' | tail -1)
    tpot_p99=$(grep -i "P99 TPOT" "$clean_file" | grep -oE '[0-9.]+' | tail -1)
    
    itl_mean=$(grep -i "Mean ITL" "$clean_file" | grep -oE '[0-9.]+' | tail -1)
    itl_median=$(grep -i "Median ITL" "$clean_file" | grep -oE '[0-9.]+' | tail -1)
    itl_p99=$(grep -i "P99 ITL" "$clean_file" | grep -oE '[0-9.]+' | tail -1)
    
    # Clean up
    rm -f "$clean_file"
    
    cat << EOF > "${csv_file}"
metric,value,unit
device,${CUDA_DEVICE_RAW},
device_sanitized,${CUDA_DEVICE_NAME},
model,${MODEL},
timestamp,${TIMESTAMP},
num_prompts,${NUM_PROMPTS},
max_concurrency,${MAX_CONCURRENCY},
num_warmups,${NUM_WARMUPS},
input_len,${INPUT_LEN},tokens
output_len,${OUTPUT_LEN},tokens
range_ratio,${RANGE_RATIO},
seed,${SEED},
images_per_request,${num_images},
image_bucket,"512x512(20%) 720x1280(30%) 1024x1024(20%) 1080x1920(30%)",
successful_requests,${successful_requests:-N/A},
failed_requests,${failed_requests:-N/A},
benchmark_duration,${benchmark_duration:-N/A},s
total_input_tokens,${total_input_tokens:-N/A},tokens
total_generated_tokens,${total_generated_tokens:-N/A},tokens
request_throughput,${request_throughput:-N/A},req/s
output_token_throughput,${output_token_throughput:-N/A},tok/s
peak_output_throughput,${peak_output_throughput:-N/A},tok/s
peak_concurrent_requests,${peak_concurrent_requests:-N/A},
total_token_throughput,${total_token_throughput:-N/A},tok/s
ttft_mean,${ttft_mean:-N/A},ms
ttft_median,${ttft_median:-N/A},ms
ttft_p99,${ttft_p99:-N/A},ms
tpot_mean,${tpot_mean:-N/A},ms
tpot_median,${tpot_median:-N/A},ms
tpot_p99,${tpot_p99:-N/A},ms
itl_mean,${itl_mean:-N/A},ms
itl_median,${itl_median:-N/A},ms
itl_p99,${itl_p99:-N/A},ms
EOF

    log "CSV saved to: ${csv_file}"
}

# =============================================================================
# BENCHMARK FUNCTION
# =============================================================================

run_benchmark() {
    local num_images=$1
    local test_name=$2
    
    local csv_file="${OUTPUT_DIR}/vLLM_${CUDA_DEVICE_NAME}_${MODEL_SHORT}_samples_${NUM_PROMPTS}_images_${num_images}_${TIMESTAMP}.csv"
    local temp_file=$(mktemp)
    
    log ""
    log "============================================================"
    log "TEST: ${test_name}"
    log "Images per request: ${num_images}"
    log "Started: $(date)"
    log "============================================================"
    log ""
    
    # Run benchmark with TTY emulation for progress bars
    # Use 'script' to capture output while preserving TTY behavior
    script -q -e -c "docker run -t --rm --network host \
        nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION} \
        vllm bench serve \
        --backend openai-chat \
        --model \"${MODEL}\" \
        --endpoint /v1/chat/completions \
        --base-url http://localhost:8000 \
        --dataset-name random-mm \
        --num-prompts ${NUM_PROMPTS} \
        --max-concurrency ${MAX_CONCURRENCY} \
        --num-warmups ${NUM_WARMUPS} \
        --random-prefix-len 0 \
        --random-input-len ${INPUT_LEN} \
        --random-output-len ${OUTPUT_LEN} \
        --random-range-ratio ${RANGE_RATIO} \
        --random-mm-base-items-per-request ${num_images} \
        --random-mm-num-mm-items-range-ratio 0.0 \
        --random-mm-limit-mm-per-prompt '{\"image\": 1, \"video\": 0}' \
        --random-mm-bucket-config '{(512, 512, 1): 0.2, (720, 1280, 1): 0.3, (1024, 1024, 1): 0.2, (1080, 1920, 1): 0.3}' \
        --request-rate inf \
        --ignore-eos \
        --seed ${SEED}" "${temp_file}"
    
    # Also append to the main output file (cleaned)
    cat "${temp_file}" | strip_ansi >> "${OUTPUT_FILE}"
    
    parse_and_save_csv "$temp_file" "$num_images" "$csv_file"
    rm -f "$temp_file"
    
    log ""
    log "Completed: $(date)"
    log ""
}

# =============================================================================
# MAIN
# =============================================================================

# Write metadata header
cat << EOF | tee "${OUTPUT_FILE}"
##############################################################
# vLLM BENCHMARK RESULTS
##############################################################

DEVICE INFO
-----------
Device (raw):       ${CUDA_DEVICE_RAW}
Device (sanitized): ${CUDA_DEVICE_NAME}
$(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader 2>/dev/null || echo "GPU info not available")

METADATA
--------
Date:            $(date)
Hostname:        $(hostname)
Model:           ${MODEL}
Num Prompts:     ${NUM_PROMPTS}
Max Concurrency: ${MAX_CONCURRENCY}
Num Warmups:     ${NUM_WARMUPS}
Input Length:    ${INPUT_LEN} (±${RANGE_RATIO})
Output Length:   ${OUTPUT_LEN} (±${RANGE_RATIO})
Seed:            ${SEED}

##############################################################
# TEST RESULTS
##############################################################
EOF

# Run tests
run_benchmark 0 "Text-Only (0 images)"
run_benchmark 1 "Single Image (1 image)"

# Write footer
log ""
log "##############################################################"
log "# vLLM BENCHMARK COMPLETE"
log "##############################################################"
log "Results saved to: ${OUTPUT_FILE}"

echo ""
echo "============================================================"
echo "DEVICE: ${CUDA_DEVICE_RAW}"
echo "============================================================"
echo "OUTPUT FILES:"
echo "  Directory: ${OUTPUT_DIR}/"
echo "  Text: ${OUTPUT_FILE}"
echo "============================================================"