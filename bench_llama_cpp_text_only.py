#!/usr/bin/env python3
"""bench_llama_cpp.py - Run on: client machine"""

import argparse
import requests
import random
import subprocess
import re
import os
from datetime import datetime
from tqdm import tqdm

# Top 100 most common English words + original words for diversity
WORD_LIST = [
    # Original words
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and", "runs",
    # Top 100 common words
    "a", "about", "all", "also", "an", "as", "at", "be", "because", "but",
    "by", "can", "come", "could", "day", "do", "even", "find", "first", "for",
    "from", "get", "give", "go", "have", "he", "her", "here", "him", "his",
    "how", "i", "if", "in", "into", "it", "its", "just", "know", "like",
    "look", "make", "man", "many", "me", "more", "my", "new", "no", "not",
    "now", "of", "on", "one", "only", "or", "other", "our", "out", "people",
    "say", "see", "she", "so", "some", "take", "tell", "than", "that", "their",
    "them", "then", "there", "these", "they", "thing", "think", "this", "time", "to",
    "two", "up", "us", "use", "very", "want", "way", "we", "well", "what",
    "when", "which", "who", "will", "with", "would", "year", "you", "your",
    # Additional variety words
    "after", "again", "air", "animal", "answer", "any", "ask", "back", "bad", "been",
    "before", "begin", "between", "big", "book", "boy", "call", "came", "change", "city",
    "close", "country", "cut", "different", "does", "down", "each", "earth", "eat", "end",
    "enough", "every", "example", "eye", "face", "family", "far", "father", "feel", "few",
    "follow", "food", "form", "found", "friend", "girl", "good", "got", "government", "great",
    "group", "hand", "hard", "head", "help", "high", "home", "house", "idea", "important",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark llama-server completion performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--server-url", "-s",
        type=str,
        default="http://localhost:30000",
        help="llama-server URL"
    )
    parser.add_argument(
        "--num-prompts", "-n",
        type=int,
        default=100,
        help="Number of measured prompts (excluding warmups)"
    )
    parser.add_argument(
        "--warmups", "-w",
        type=int,
        default=10,
        help="Number of warmup runs to ignore"
    )
    parser.add_argument(
        "--input-range", "-i",
        type=int,
        nargs=2,
        default=[50, 1000],
        metavar=("MIN", "MAX"),
        help="Input token range [min, max]"
    )
    parser.add_argument(
        "--output-range", "-o",
        type=int,
        nargs=2,
        default=[50, 1000],
        metavar=("MIN", "MAX"),
        help="Output token range [min, max]"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Output directory for results"
    )
    
    return parser.parse_args()


def get_gpu_name() -> str:
    """Detect GPU name, sanitize for filenames."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        raw_name = result.stdout.strip().split("\n")[0]
        sanitized = re.sub(r'[^a-zA-Z0-9]', '_', raw_name)
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
        return sanitized if sanitized else "Unknown_GPU"
    except Exception:
        return "Unknown_GPU"


def get_model_info(server_url: str, timeout: int = 10) -> dict:
    """Fetch model info from llama-server."""
    model_info = {
        "model_path": "Unknown",
        "model_name": "Unknown_Model",
        "model_name_sanitized": "Unknown_Model",
    }
    
    try:
        # Try /props first (most detailed)
        response = requests.get(f"{server_url}/props", timeout=timeout)
        if response.ok:
            data = response.json()
            model_path = data.get("model_path", "")
            if model_path:
                model_info["model_path"] = model_path
                model_name = os.path.basename(model_path)
                model_info["model_name"] = model_name
                # Sanitize for filenames
                sanitized = re.sub(r'[^a-zA-Z0-9]', '_', model_name)
                sanitized = re.sub(r'_+', '_', sanitized).strip('_')
                model_info["model_name_sanitized"] = sanitized
                return model_info
        
        # Fallback: try /v1/models
        response = requests.get(f"{server_url}/v1/models", timeout=timeout)
        if response.ok:
            data = response.json()
            models = data.get("data", [])
            if models:
                model_name = models[0].get("id", "Unknown_Model")
                model_info["model_name"] = model_name
                sanitized = re.sub(r'[^a-zA-Z0-9]', '_', model_name)
                sanitized = re.sub(r'_+', '_', sanitized).strip('_')
                model_info["model_name_sanitized"] = sanitized
        
        return model_info
    except Exception:
        return model_info


def generate_prompt(approx_tokens: int) -> str:
    """Generate a prompt of approximately N tokens."""
    word_count = int(approx_tokens * 0.75)
    return " ".join(random.choices(WORD_LIST, k=word_count))


def run_single(server_url: str, prompt: str, n_predict: int, timeout: int) -> dict:
    """Run a single completion and return timings."""
    response = requests.post(
        f"{server_url}/completion",
        json={"prompt": prompt, "n_predict": n_predict, "stream": False, "cache_prompt": False},
        headers={"Content-Type": "application/json"},
        timeout=timeout
    )
    return response.json().get("timings", {})


def save_csv(args, measured: list, gpu_name: str, model_info: dict, timestamp: str) -> str:
    """Save results to CSV in standard naming format."""
    
    total_prompt_tokens = sum(r["prompt_n"] for r in measured)
    total_generated_tokens = sum(r["predicted_n"] for r in measured)
    
    # True aggregate throughput
    total_prompt_time = sum(r["prompt_n"] / r["prompt_per_sec"] for r in measured if r["prompt_per_sec"] > 0)
    total_gen_time = sum(r["predicted_n"] / r["predicted_per_sec"] for r in measured if r["predicted_per_sec"] > 0)
    
    true_prompt_throughput = total_prompt_tokens / total_prompt_time if total_prompt_time > 0 else 0
    true_gen_throughput = total_generated_tokens / total_gen_time if total_gen_time > 0 else 0
    total_runtime = total_prompt_time + total_gen_time
    
    avg_prompt_speed = sum(r["prompt_per_sec"] for r in measured) / len(measured)
    avg_gen_speed = sum(r["predicted_per_sec"] for r in measured) / len(measured)
    
    output_dir = f"{args.output_dir}/{gpu_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    model_short = model_info["model_name_sanitized"][:50]  # Truncate if too long
    csv_file = f"{output_dir}/llama_server_{gpu_name}_{model_short}_samples_{args.num_prompts}_{timestamp}.csv"
    
    with open(csv_file, "w") as f:
        f.write("metric,value,unit\n")
        f.write(f"device_sanitized,{gpu_name},\n")
        f.write(f"server,llama-server,\n")
        f.write(f"server_url,{args.server_url},\n")
        f.write(f"model_name,{model_info['model_name']},\n")
        f.write(f"model_path,{model_info['model_path']},\n")
        f.write(f"timestamp,{timestamp},\n")
        f.write(f"test_type,text_only,\n")
        f.write(f"num_prompts,{args.num_prompts},\n")
        f.write(f"warmup_runs,{args.warmups},\n")
        f.write(f"input_token_range,{args.input_range[0]}-{args.input_range[1]},tokens\n")
        f.write(f"output_token_range,{args.output_range[0]}-{args.output_range[1]},tokens\n")
        f.write(f"seed,{args.seed},\n")
        f.write(f"total_prompt_tokens,{total_prompt_tokens},tokens\n")
        f.write(f"total_generated_tokens,{total_generated_tokens},tokens\n")
        f.write(f"true_prompt_throughput,{true_prompt_throughput:.2f},tok/s\n")
        f.write(f"true_gen_throughput,{true_gen_throughput:.2f},tok/s\n")
        f.write(f"total_runtime,{total_runtime:.2f},seconds\n")
        f.write(f"avg_prompt_speed,{avg_prompt_speed:.2f},tok/s\n")
        f.write(f"avg_gen_speed,{avg_gen_speed:.2f},tok/s\n")
    
    print(f"\nCSV saved to: {csv_file}")
    return csv_file


def main():
    args = parse_args()
    
    random.seed(args.seed)
    
    gpu_name = get_gpu_name()
    model_info = get_model_info(args.server_url)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_runs = args.num_prompts + args.warmups
    
    print("=" * 60)
    print("LLAMA-SERVER BENCHMARK")
    print("=" * 60)
    print(f"GPU: {gpu_name}")
    print(f"Model: {model_info['model_name']}")
    print(f"Server: {args.server_url}")
    print(f"Runs: {total_runs} ({args.warmups} warmups, {args.num_prompts} measured)")
    print(f"Input tokens: {args.input_range[0]}-{args.input_range[1]}")
    print(f"Output tokens: {args.output_range[0]}-{args.output_range[1]}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    results = []
    
    pbar = tqdm(range(total_runs), desc="Benchmarking", unit="req")
    
    for i in pbar:
        input_tokens = random.randint(args.input_range[0], args.input_range[1])
        output_tokens = random.randint(args.output_range[0], args.output_range[1])
        prompt = generate_prompt(input_tokens)
        
        is_warmup = i < args.warmups
        
        try:
            timings = run_single(args.server_url, prompt, output_tokens, args.timeout)
        except requests.RequestException as e:
            pbar.write(f"ERROR on run {i+1}: {e}")
            continue
        
        result = {
            "prompt_n": timings.get("prompt_n", 0),
            "predicted_n": timings.get("predicted_n", 0),
            "prompt_per_sec": timings.get("prompt_per_second", 0),
            "predicted_per_sec": timings.get("predicted_per_second", 0),
            "is_warmup": is_warmup,
        }
        results.append(result)
        
        label = "WRM" if is_warmup else "RUN"
        pbar.set_postfix({
            "type": label,
            "in": f"{result['prompt_n']}tok",
            "out": f"{result['predicted_n']}tok",
            "pp": f"{result['prompt_per_sec']:.0f}t/s",
            "tg": f"{result['predicted_per_sec']:.1f}t/s",
        })
    
    pbar.close()
    
    # === TOTALS (excluding warmups) ===
    measured = [r for r in results if not r["is_warmup"]]
    
    if not measured:
        print("ERROR: No successful measured runs!")
        return
    
    total_prompt_tokens = sum(r["prompt_n"] for r in measured)
    total_generated_tokens = sum(r["predicted_n"] for r in measured)
    
    total_prompt_time = sum(r["prompt_n"] / r["prompt_per_sec"] for r in measured if r["prompt_per_sec"] > 0)
    total_gen_time = sum(r["predicted_n"] / r["predicted_per_sec"] for r in measured if r["predicted_per_sec"] > 0)
    
    true_prompt_throughput = total_prompt_tokens / total_prompt_time if total_prompt_time > 0 else 0
    true_gen_throughput = total_generated_tokens / total_gen_time if total_gen_time > 0 else 0
    
    avg_prompt_speed = sum(r["prompt_per_sec"] for r in measured) / len(measured)
    avg_gen_speed = sum(r["predicted_per_sec"] for r in measured) / len(measured)
    
    print("\n" + "=" * 60)
    print("RESULTS (excluding warmups)")
    print("=" * 60)
    print(f"Model: {model_info['model_name']}")
    print(f"Total prompt tokens:      {total_prompt_tokens:,}")
    print(f"Total generated tokens:   {total_generated_tokens:,}")
    print(f"True prompt throughput:   {true_prompt_throughput:.2f} tok/s")
    print(f"True gen throughput:      {true_gen_throughput:.2f} tok/s")
    print(f"Avg prompt speed:         {avg_prompt_speed:.2f} tok/s (per-run average)")
    print(f"Avg gen speed:            {avg_gen_speed:.2f} tok/s (per-run average)")
    
    save_csv(args, measured, gpu_name, model_info, timestamp)


if __name__ == "__main__":
    main()