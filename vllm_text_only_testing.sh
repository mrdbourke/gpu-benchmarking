#!/bin/bash
# vllm_text_only_testing.sh
# Run on: host machine (Terminal 2, while server is running)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model name - change this or pass as argument
MODEL="${1:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8}"
LATEST_VLLM_VERSION="25.12.post1-py3"

# Other settings
NUM_PROMPTS=100
MAX_CONCURRENCY=10
NUM_WARMUPS=10
INPUT_LEN=1000
OUTPUT_LEN=300
RANGE_RATIO=0.3
SEED=42

# =============================================================================
# MODEL-SPECIFIC SETTINGS
# =============================================================================

# Detect if model needs thinking disabled (Nemotron models)
DISABLE_THINKING=false
if echo "${MODEL}" | grep -iq "nemotron"; then
    echo "Detected Nemotron model - disabling thinking mode"
    DISABLE_THINKING=true
fi

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
OUTPUT_FILE="${OUTPUT_DIR}/vLLM_${CUDA_DEVICE_NAME}_${MODEL_SHORT}_text_only_${TIMESTAMP}.txt"

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
    local csv_file="$2"
    
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
test_type,text_only,
thinking_disabled,${DISABLE_THINKING},
num_prompts,${NUM_PROMPTS},
max_concurrency,${MAX_CONCURRENCY},
num_warmups,${NUM_WARMUPS},
input_len,${INPUT_LEN},tokens
output_len,${OUTPUT_LEN},tokens
range_ratio,${RANGE_RATIO},
seed,${SEED},
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
    local csv_file="${OUTPUT_DIR}/vLLM_${CUDA_DEVICE_NAME}_${MODEL_SHORT}_text_only_samples_${NUM_PROMPTS}_${TIMESTAMP}.csv"
    local temp_file=$(mktemp)
    
    log ""
    log "============================================================"
    log "TEST: Text-Only Benchmark"
    log "Started: $(date)"
    if [ "${DISABLE_THINKING}" = true ]; then
        log "Note: Thinking mode DISABLED for this model"
    fi
    log "============================================================"
    log ""
    
    # Build and run the benchmark command
    # Note: We use a temp script file to handle complex quoting for --extra-body
    local cmd_file=$(mktemp)
    
    cat << 'CMDEOF' > "${cmd_file}"
#!/bin/bash
docker run -t --rm --network host \
    nvcr.io/nvidia/vllm:VLLM_VERSION_PLACEHOLDER \
    vllm bench serve \
    --backend openai-chat \
    --model "MODEL_PLACEHOLDER" \
    --endpoint /v1/chat/completions \
    --base-url http://localhost:8000 \
    --dataset-name random \
    --num-prompts NUM_PROMPTS_PLACEHOLDER \
    --max-concurrency MAX_CONCURRENCY_PLACEHOLDER \
    --num-warmups NUM_WARMUPS_PLACEHOLDER \
    --random-prefix-len 0 \
    --random-input-len INPUT_LEN_PLACEHOLDER \
    --random-output-len OUTPUT_LEN_PLACEHOLDER \
    --random-range-ratio RANGE_RATIO_PLACEHOLDER \
    --request-rate inf \
    --ignore-eos \
    --seed SEED_PLACEHOLDER \
    EXTRA_BODY_PLACEHOLDER
CMDEOF

    # Replace placeholders
    sed -i "s|VLLM_VERSION_PLACEHOLDER|${LATEST_VLLM_VERSION}|g" "${cmd_file}"
    sed -i "s|MODEL_PLACEHOLDER|${MODEL}|g" "${cmd_file}"
    sed -i "s|NUM_PROMPTS_PLACEHOLDER|${NUM_PROMPTS}|g" "${cmd_file}"
    sed -i "s|MAX_CONCURRENCY_PLACEHOLDER|${MAX_CONCURRENCY}|g" "${cmd_file}"
    sed -i "s|NUM_WARMUPS_PLACEHOLDER|${NUM_WARMUPS}|g" "${cmd_file}"
    sed -i "s|INPUT_LEN_PLACEHOLDER|${INPUT_LEN}|g" "${cmd_file}"
    sed -i "s|OUTPUT_LEN_PLACEHOLDER|${OUTPUT_LEN}|g" "${cmd_file}"
    sed -i "s|RANGE_RATIO_PLACEHOLDER|${RANGE_RATIO}|g" "${cmd_file}"
    sed -i "s|SEED_PLACEHOLDER|${SEED}|g" "${cmd_file}"
    
    # Add extra-body for thinking models
    if [ "${DISABLE_THINKING}" = true ]; then
        sed -i 's|EXTRA_BODY_PLACEHOLDER|--extra-body '\''{"chat_template_kwargs":{"enable_thinking":false}}'\''|g' "${cmd_file}"
    else
        sed -i 's|EXTRA_BODY_PLACEHOLDER||g' "${cmd_file}"
    fi
    
    chmod +x "${cmd_file}"
    
    # Debug: show the command (optional - uncomment to verify)
    # echo "=== Generated command ===" && cat "${cmd_file}" && echo "=== End command ==="
    
    # Run benchmark with TTY emulation for progress bars
    script -q -e -c "bash ${cmd_file}" "${temp_file}"
    
    rm -f "${cmd_file}"
    
    # Also append to the main output file (cleaned)
    cat "${temp_file}" | strip_ansi >> "${OUTPUT_FILE}"
    
    parse_and_save_csv "$temp_file" "$csv_file"
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
# vLLM TEXT-ONLY BENCHMARK RESULTS
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
Test Type:       Text-Only
Thinking Mode:   $([ "${DISABLE_THINKING}" = true ] && echo "DISABLED" || echo "Default")
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

# Run text-only benchmark
run_benchmark

# Write footer
log ""
log "##############################################################"
log "# vLLM TEXT-ONLY BENCHMARK COMPLETE"
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