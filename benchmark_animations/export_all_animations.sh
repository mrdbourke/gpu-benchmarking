#!/usr/bin/env bash
set -euo pipefail

QUALITY_ARG="${1:-k}"

case "${QUALITY_ARG}" in
  k) QUALITY_FLAG="-qk" ;;
  h) QUALITY_FLAG="-qh" ;;
  m) QUALITY_FLAG="-qm" ;;
  l) QUALITY_FLAG="-ql" ;;
  *)
    echo "Usage: $0 [k|h|m|l]"
    echo "  k = 4k (default)"
    echo "  h = 1080p"
    echo "  m = 720p"
    echo "  l = 480p"
    exit 1
    ;;
 esac

ANIM_DIR="/Users/daniel/code/gpu-benchmarking/benchmark_animations"

run_manim() {
  local script="$1"
  local scene="$2"
  echo "Running: manim ${QUALITY_FLAG} ${script} ${scene}"
  manim "${QUALITY_FLAG}" "${script}" "${scene}"
}

run_manim "${ANIM_DIR}/vllm_input_manim.py" "VLLMInputThroughput"
run_manim "${ANIM_DIR}/vllm_output_manim.py" "VLLMOutputThroughput"
run_manim "${ANIM_DIR}/llama_cpp_input_manim.py" "LlamaCppInputThroughput"
run_manim "${ANIM_DIR}/llama_cpp_output_manim.py" "LlamaCppOutputThroughput"
run_manim "${ANIM_DIR}/llama_cpp_vs_vllm_input_manim.py" "LlamaCppVsVLLMInputThroughput"
run_manim "${ANIM_DIR}/llama_cpp_vs_vllm_output_manim.py" "LlamaCppVsVLLMOutputThroughput"
run_manim "${ANIM_DIR}/llm_finetune_time_manim.py" "LLMFineTuneTrainingTime"
run_manim "${ANIM_DIR}/image_generation_manim.py" "ImageGenerationThroughput"
run_manim "${ANIM_DIR}/object_detection_manim.py" "ObjectDetectionTrainingTime"

echo "All animations rendered."
