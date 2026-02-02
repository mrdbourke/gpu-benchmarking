#!/usr/bin/env python3
"""
LLM Inference Speed Benchmark Script

Tests LLM inference speed on recipe-to-image-prompt generation task.
Logs hardware metadata and saves all results to JSON.

Usage:
    python test_llm_inference.py                     # Default: 1000 samples
    python test_llm_inference.py --num_samples 100   # Custom sample count
    python test_llm_inference.py --max_tokens 500    # Custom max tokens
    python test_llm_inference.py --base_url http://localhost:8080  # Custom server

Run on: The machine running the LLM inference server (where llama.cpp/vLLM is hosted)
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Third-party imports
try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install datasets tqdm --break-system-packages")
    sys.exit(1)

# Local import - chat_completion.py should be in same directory
try:
    from chat_completion import chat_completion
except ImportError:
    print("Error: chat_completion.py not found in current directory")
    print("Make sure chat_completion.py is in the same folder as this script")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

RECIPE_BASE_PROMPT = """
You are a food photography prompt generator. Your task is to convert a recipe into an image generation prompt that showcases the final dish.

## Instructions

1. **Identify the Final Dish**
   - Read the recipe title and description
   - Look at the final step(s) to understand what the completed dish looks like
   - Note any unique presentation details mentioned

2. **Extract Visual Elements**
   - Main ingredient(s): What is the star of the dish?
   - Color: What colors should be prominent?
   - Texture: How should it look? (crispy, juicy, tender, flaky, glistening)
   - Cooking method result: What does this method produce visually?
   - Garnishes/toppings: Any finishing touches mentioned or implied?

3. **Check Reviews for Visual Cues**
   - Scan reviews for descriptive words about the outcome (juicy, sweet, tender, crispy, etc.)

4. **Build the Prompt Using This Structure**
   [Shot type] of [dish name] with [key visual qualities of main ingredient]. [Unique presentation details]. [Texture/finish description]. [Optional toppings/garnishes]. First-person perspective, casual food photo, [angle], shallow depth of field, natural lighting.

## Input

<recipe_markdown>
{recipe_markdown}
</recipe_markdown>

## Output

Return only the image generation prompt wrapped in tags like this:

<recipe_generation_prompt>
Your generated prompt here
</recipe_generation_prompt>
"""

DATASET_PATH = "mrdbourke/recipe-synthetic-images-10k"
RANDOM_SEED = 42


# ============================================================================
# HARDWARE DETECTION
# ============================================================================

def get_gpu_info() -> dict[str, Any]:
    """Get NVIDIA GPU information via nvidia-smi."""
    gpu_info = {
        "available": False,
        "gpus": [],
        "driver_version": None,
        "cuda_version": None,
    }
    
    try:
        # Get GPU details in JSON-like format
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            gpu_info["available"] = True
            
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 8:
                        gpu_info["gpus"].append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_total_mb": int(parts[2]),
                            "memory_free_mb": int(parts[3]),
                            "memory_used_mb": int(parts[4]),
                            "temperature_c": int(parts[5]) if parts[5] != "[N/A]" else None,
                            "utilization_percent": int(parts[6]) if parts[6] != "[N/A]" else None,
                            "power_draw_w": float(parts[7]) if parts[7] != "[N/A]" else None,
                        })
            
            # Get driver and CUDA version
            version_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if version_result.returncode == 0:
                gpu_info["driver_version"] = version_result.stdout.strip().split("\n")[0]
            
            # Parse CUDA version from nvidia-smi header
            header_result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if header_result.returncode == 0:
                for line in header_result.stdout.split("\n"):
                    if "CUDA Version" in line:
                        cuda_part = line.split("CUDA Version:")[-1].strip()
                        gpu_info["cuda_version"] = cuda_part.split()[0] if cuda_part else None
                        break
                        
    except FileNotFoundError:
        pass  # nvidia-smi not available
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        gpu_info["error"] = str(e)
    
    return gpu_info


def get_cpu_info() -> dict[str, Any]:
    """Get CPU information."""
    cpu_info = {
        "processor": platform.processor(),
        "architecture": platform.machine(),
        "physical_cores": None,
        "total_cores": None,
        "model_name": None,
    }
    
    try:
        import multiprocessing
        cpu_info["total_cores"] = multiprocessing.cpu_count()
    except Exception:
        pass
    
    # Try to get more detailed CPU info on Linux
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
            for line in cpuinfo.split("\n"):
                if "model name" in line:
                    cpu_info["model_name"] = line.split(":")[1].strip()
                    break
            
            # Count physical cores
            physical_ids = set()
            core_ids = set()
            current_physical_id = None
            for line in cpuinfo.split("\n"):
                if "physical id" in line:
                    current_physical_id = line.split(":")[1].strip()
                    physical_ids.add(current_physical_id)
                if "core id" in line and current_physical_id is not None:
                    core_ids.add((current_physical_id, line.split(":")[1].strip()))
            
            if core_ids:
                cpu_info["physical_cores"] = len(core_ids)
    except Exception:
        pass
    
    return cpu_info


def get_memory_info() -> dict[str, Any]:
    """Get system memory information."""
    mem_info = {
        "total_gb": None,
        "available_gb": None,
        "used_gb": None,
    }
    
    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = f.read()
            for line in meminfo.split("\n"):
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                    mem_info["total_gb"] = round(total_kb / 1024 / 1024, 2)
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
                    mem_info["available_gb"] = round(avail_kb / 1024 / 1024, 2)
            
            if mem_info["total_gb"] and mem_info["available_gb"]:
                mem_info["used_gb"] = round(mem_info["total_gb"] - mem_info["available_gb"], 2)
    except Exception:
        pass
    
    return mem_info


def get_hardware_metadata() -> dict[str, Any]:
    """Collect all hardware metadata."""
    return {
        "timestamp": datetime.now().isoformat(),
        "hostname": platform.node(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "python_version": platform.python_version(),
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "gpu": get_gpu_info(),
    }


def print_hardware_summary(metadata: dict[str, Any]) -> None:
    """Print a formatted hardware summary to terminal."""
    print("\n" + "=" * 70)
    print("HARDWARE CONFIGURATION")
    print("=" * 70)
    
    print(f"\n📅 Timestamp: {metadata['timestamp']}")
    print(f"🖥️  Hostname: {metadata['hostname']}")
    print(f"🐧 OS: {metadata['os']['system']} {metadata['os']['release']}")
    print(f"🐍 Python: {metadata['python_version']}")
    
    cpu = metadata["cpu"]
    print(f"\n💻 CPU: {cpu.get('model_name', cpu.get('processor', 'Unknown'))}")
    if cpu.get("physical_cores"):
        print(f"   Cores: {cpu['physical_cores']} physical, {cpu.get('total_cores', 'N/A')} total")
    
    mem = metadata["memory"]
    if mem.get("total_gb"):
        print(f"\n🧠 RAM: {mem['total_gb']} GB total, {mem.get('available_gb', 'N/A')} GB available")
    
    gpu = metadata["gpu"]
    if gpu["available"]:
        print(f"\n🎮 GPU(s): {len(gpu['gpus'])} detected")
        print(f"   Driver: {gpu.get('driver_version', 'N/A')}, CUDA: {gpu.get('cuda_version', 'N/A')}")
        for g in gpu["gpus"]:
            print(f"   [{g['index']}] {g['name']}")
            print(f"       Memory: {g['memory_used_mb']} / {g['memory_total_mb']} MB")
            if g.get('temperature_c'):
                print(f"       Temp: {g['temperature_c']}°C, Power: {g.get('power_draw_w', 'N/A')}W")
    else:
        print("\n🎮 GPU: No NVIDIA GPU detected")
    
    print("=" * 70 + "\n")


# ============================================================================
# INFERENCE BENCHMARK
# ============================================================================

def run_inference_benchmark(
    num_samples: int = 1000,
    max_tokens: int = 1000,
    enable_thinking: bool = False,
    base_url: str = "http://localhost:30000",
    output_dir: str = "benchmark_results",
) -> dict[str, Any]:
    """
    Run inference benchmark on recipe dataset.
    
    Args:
        num_samples: Number of samples to process
        max_tokens: Maximum tokens per response
        enable_thinking: Whether to enable thinking/reasoning mode
        base_url: LLM server URL
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing all benchmark results and metadata
    """
    
    # Collect hardware metadata first
    print("🔍 Collecting hardware information...")
    hardware_metadata = get_hardware_metadata()
    print_hardware_summary(hardware_metadata)
    
    # Load and sample dataset
    print(f"📦 Loading dataset: {DATASET_PATH}")
    dataset = load_dataset(path=DATASET_PATH)
    
    print(f"🎲 Sampling {num_samples} recipes (seed={RANDOM_SEED})...")
    sampled_dataset = dataset["train"].shuffle(seed=RANDOM_SEED).select(range(num_samples))
    
    # Initialize results structure
    results = {
        "metadata": {
            "hardware": hardware_metadata,
            "benchmark_config": {
                "dataset_path": DATASET_PATH,
                "num_samples": num_samples,
                "random_seed": RANDOM_SEED,
                "max_tokens": max_tokens,
                "enable_thinking": enable_thinking,
                "base_url": base_url,
            },
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_duration_seconds": None,
        },
        "summary_statistics": {},
        "individual_results": [],
    }
    
    # Run inference
    print(f"\n🚀 Starting inference on {num_samples} samples...")
    print(f"   Server: {base_url}")
    print(f"   Max tokens: {max_tokens}")
    print(f"   Thinking mode: {'ON' if enable_thinking else 'OFF'}\n")
    
    start_time = time.time()
    successful = 0
    failed = 0
    
    # Timing accumulators
    total_prompt_tokens = 0
    total_predicted_tokens = 0
    total_prompt_ms = 0
    total_predicted_ms = 0
    tokens_per_second_list = []
    
    for i, sample in enumerate(tqdm(sampled_dataset, desc="Processing recipes")):
        sample_result = {
            "index": i,
            "recipe_id": sample.get("id"),
            "recipe_name": sample.get("name"),
            "success": False,
            "error": None,
            "response": None,
            "timings": None,
            "model": None,
        }
        
        try:
            # Build prompt
            prompt = RECIPE_BASE_PROMPT.format(recipe_markdown=sample["recipe_markdown"])
            
            # Run inference
            response = chat_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                base_url=base_url,
            )
            
            # Extract results
            sample_result["success"] = True
            sample_result["model"] = response.get("model")
            sample_result["response"] = {
                "content": response["choices"][0]["message"]["content"],
                "finish_reason": response["choices"][0].get("finish_reason"),
            }
            sample_result["timings"] = response.get("timings", {})
            sample_result["usage"] = response.get("usage", {})
            
            # Accumulate statistics
            timings = response.get("timings", {})
            if timings:
                total_prompt_tokens += timings.get("prompt_n", 0)
                total_predicted_tokens += timings.get("predicted_n", 0)
                total_prompt_ms += timings.get("prompt_ms", 0)
                total_predicted_ms += timings.get("predicted_ms", 0)
                
                if timings.get("predicted_per_second"):
                    tokens_per_second_list.append(timings["predicted_per_second"])
            
            successful += 1
            
        except Exception as e:
            sample_result["error"] = str(e)
            failed += 1
        
        results["individual_results"].append(sample_result)
    
    # Calculate final statistics
    end_time = time.time()
    total_duration = end_time - start_time
    
    results["metadata"]["end_time"] = datetime.now().isoformat()
    results["metadata"]["total_duration_seconds"] = round(total_duration, 2)
    
    # Summary statistics
    results["summary_statistics"] = {
        "total_samples": num_samples,
        "successful": successful,
        "failed": failed,
        "success_rate": round(successful / num_samples * 100, 2),
        "total_duration_seconds": round(total_duration, 2),
        "avg_time_per_sample_seconds": round(total_duration / num_samples, 3),
        "samples_per_minute": round(num_samples / total_duration * 60, 2),
        "total_prompt_tokens": total_prompt_tokens,
        "total_predicted_tokens": total_predicted_tokens,
        "total_tokens": total_prompt_tokens + total_predicted_tokens,
        "avg_prompt_tokens_per_sample": round(total_prompt_tokens / successful, 1) if successful else 0,
        "avg_predicted_tokens_per_sample": round(total_predicted_tokens / successful, 1) if successful else 0,
        "total_prompt_time_seconds": round(total_prompt_ms / 1000, 2),
        "total_generation_time_seconds": round(total_predicted_ms / 1000, 2),
        "avg_tokens_per_second": round(sum(tokens_per_second_list) / len(tokens_per_second_list), 2) if tokens_per_second_list else 0,
        "min_tokens_per_second": round(min(tokens_per_second_list), 2) if tokens_per_second_list else 0,
        "max_tokens_per_second": round(max(tokens_per_second_list), 2) if tokens_per_second_list else 0,
    }
    
    return results


def save_results(results: dict[str, Any], output_dir: str = "benchmark_results") -> str:
    """Save results to JSON file and return filepath."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp and model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "unknown"
    
    # Try to extract model name from results
    if results["individual_results"]:
        for r in results["individual_results"]:
            if r.get("model"):
                # Clean up model name for filename
                model_name = r["model"].replace("/", "_").replace(".", "_").split(".gguf")[0]
                break
    
    filename = f"benchmark_{model_name}_{timestamp}.json"
    filepath = output_path / filename
    
    # Save to JSON
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    return str(filepath)


def print_summary(results: dict[str, Any]) -> None:
    """Print a formatted summary of benchmark results."""
    stats = results["summary_statistics"]
    config = results["metadata"]["benchmark_config"]
    
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 Samples: {stats['successful']}/{stats['total_samples']} successful ({stats['success_rate']}%)")
    print(f"⏱️  Total time: {stats['total_duration_seconds']} seconds ({stats['total_duration_seconds']/60:.1f} min)")
    print(f"🚀 Throughput: {stats['samples_per_minute']:.1f} samples/minute")
    print(f"⚡ Avg time per sample: {stats['avg_time_per_sample_seconds']:.3f} seconds")
    
    print(f"\n📝 Token Statistics:")
    print(f"   Total tokens processed: {stats['total_tokens']:,}")
    print(f"   Prompt tokens: {stats['total_prompt_tokens']:,} (avg {stats['avg_prompt_tokens_per_sample']:.0f}/sample)")
    print(f"   Generated tokens: {stats['total_predicted_tokens']:,} (avg {stats['avg_predicted_tokens_per_sample']:.0f}/sample)")
    
    print(f"\n⚡ Generation Speed:")
    print(f"   Average: {stats['avg_tokens_per_second']:.1f} tokens/sec")
    print(f"   Range: {stats['min_tokens_per_second']:.1f} - {stats['max_tokens_per_second']:.1f} tokens/sec")
    
    print("=" * 70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM inference speed on recipe dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_llm_inference.py                          # Default: 1000 samples
  python test_llm_inference.py --num_samples 100        # Quick test with 100 samples
  python test_llm_inference.py --max_tokens 500         # Limit output tokens
  python test_llm_inference.py --base_url http://localhost:8080
        """
    )
    
    parser.add_argument(
        "--num_samples", "-n",
        type=int,
        default=1000,
        help="Number of samples to benchmark (default: 1000)"
    )
    parser.add_argument(
        "--max_tokens", "-t",
        type=int,
        default=1000,
        help="Maximum tokens per response (default: 1000)"
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking/reasoning mode"
    )
    parser.add_argument(
        "--base_url", "-u",
        type=str,
        default="http://localhost:30000",
        help="LLM server URL (default: http://localhost:30000)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "🏁 " * 20)
    print("LLM INFERENCE BENCHMARK")
    print("🏁 " * 20)
    
    # Run benchmark
    results = run_inference_benchmark(
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        enable_thinking=args.enable_thinking,
        base_url=args.base_url,
        output_dir=args.output_dir,
    )
    
    # Save results
    filepath = save_results(results, args.output_dir)
    print(f"💾 Results saved to: {filepath}")
    
    # Print summary
    print_summary(results)
    
    print(f"✅ Benchmark complete! Full results: {filepath}\n")


if __name__ == "__main__":
    main()