#!/usr/bin/env python3
"""bench_llama_cpp_text_and_image.py - Run on: client machine

Benchmarks llama-server completion performance with multimodal (image) support enabled by default.

Examples:
    # Multimodal with default buckets (phone screenshots, mac screenshots,
    # instagram squares, phone photos - 25% each) and mixed prompts
    python bench_llama_cpp_text_and_image.py -s http://localhost:30000 -n 100

    # Multimodal with JSON extraction prompts only
    python bench_llama_cpp_text_and_image.py -s http://localhost:30000 -n 100 --prompt-type json

    # Multimodal with description prompts only
    python bench_llama_cpp_text_and_image.py -s http://localhost:30000 -n 100 --prompt-type description

    # Multimodal with custom bucket distribution
    python bench_llama_cpp_text_and_image.py -s http://localhost:30000 -n 100 \\
        --mm-bucket-config '{(1080, 1920, 1): 0.5, (1080, 1080, 1): 0.5}'

    # Text-only mode (no images)
    python bench_llama_cpp_text_and_image.py -s http://localhost:30000 -n 100 --text-only
"""

import argparse
import requests
import random
import subprocess
import re
import os
import base64
import ast
import io
import time
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

# Try to import PIL for image generation
try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

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

# Image description prompts for multimodal benchmarking
DESCRIPTION_PROMPTS = [
    "Describe this image in detail.",
    "What do you see in this image?",
    "Please analyze this image thoroughly.",
    "Explain what is shown in this picture.",
    "Give a detailed description of this image.",
    "What are the key elements in this image?",
    "Describe the contents of this image.",
    "What can you tell me about this image?",
]

# JSON extraction prompts for structured output benchmarking
JSON_EXTRACTION_PROMPTS = [
    "Extract all visible information from this image and return it as a JSON object with relevant keys.",
    "Analyze this image and output a structured JSON containing: objects, colors, text (if any), and scene description.",
    "Parse this image into a JSON structure with keys: 'main_subject', 'background', 'colors', 'mood', 'details'.",
    "Convert the visual information in this image to a JSON format with appropriate fields.",
    "Extract structured data from this image. Return JSON with: 'elements', 'layout', 'text_content', 'visual_style'.",
    "Identify and categorize all elements in this image. Output as JSON with nested objects for each category.",
    "Create a detailed JSON representation of this image including: 'objects': [], 'actions': [], 'setting': {}, 'attributes': {}.",
    "Transform this image into structured JSON data. Include all identifiable elements, their positions, and relationships.",
    "Output a JSON schema representing this image with fields for: content_type, primary_elements, secondary_elements, metadata.",
    "Analyze and extract: Return a JSON object with 'visual_elements', 'text_detected', 'dominant_colors', 'composition'.",
]

# Default multimodal bucket config with realistic image sizes
# Format: (height, width, num_images): probability
DEFAULT_MM_BUCKET_CONFIG = {
    # Phone screenshots (iPhone 14 Pro style, portrait)
    (2556, 1179, 1): 0.25,
    # Mac screenshots (1080p, landscape)
    (1080, 1920, 1): 0.25,
    # Square photos (Instagram standard)
    (1080, 1080, 1): 0.25,
    # Phone photos (iPhone 12MP, portrait orientation)
    (4032, 3024, 1): 0.25,
}

# Human-readable labels for default buckets (for display)
DEFAULT_MM_BUCKET_LABELS = {
    (2556, 1179, 1): "Phone screenshot (iPhone 14 Pro)",
    (1080, 1920, 1): "Mac screenshot (1080p)",
    (1080, 1080, 1): "Square photo (Instagram)",
    (4032, 3024, 1): "Phone photo (12MP portrait)",
}


def parse_bucket_config(config_str: str) -> dict:
    """
    Parse bucket config string into a dictionary.
    
    Format: '{(height, width, num_images): probability, ...}'
    Example: '{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}'
    
    Special value 'default' returns the DEFAULT_MM_BUCKET_CONFIG.
    
    Returns dict with tuple keys (height, width, num_images) and float probability values.
    """
    if not config_str:
        return DEFAULT_MM_BUCKET_CONFIG.copy()
    
    # Handle 'default' keyword
    if config_str.lower().strip() == 'default':
        return DEFAULT_MM_BUCKET_CONFIG.copy()
    
    try:
        # Use ast.literal_eval for safe parsing of the dict literal
        parsed = ast.literal_eval(config_str)
        
        if not isinstance(parsed, dict):
            raise ValueError("Config must be a dictionary")
        
        # Validate and normalize
        result = {}
        total_prob = 0.0
        
        for key, prob in parsed.items():
            if not isinstance(key, tuple) or len(key) != 3:
                raise ValueError(f"Keys must be tuples of (height, width, num_images), got: {key}")
            
            height, width, num_images = key
            if not all(isinstance(x, int) and x > 0 for x in [height, width, num_images]):
                raise ValueError(f"Height, width, and num_images must be positive integers: {key}")
            
            if not isinstance(prob, (int, float)) or prob < 0:
                raise ValueError(f"Probability must be non-negative number, got: {prob}")
            
            result[(height, width, num_images)] = float(prob)
            total_prob += prob
        
        # Normalize probabilities if they don't sum to 1
        if abs(total_prob - 1.0) > 0.001:
            print(f"Warning: Probabilities sum to {total_prob}, normalizing to 1.0")
            for key in result:
                result[key] /= total_prob
        
        return result
        
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Invalid bucket config format: {e}\n"
                        f"Expected format: '{{(height, width, num_images): probability, ...}}'\n"
                        f"Example: '{{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}}'")


def generate_synthetic_image(height: int, width: int) -> bytes:
    """
    Generate a synthetic random image of the specified dimensions.
    
    Returns PNG image as bytes.
    """
    if not HAS_PIL:
        raise ImportError("PIL and numpy are required for multimodal benchmarking. "
                         "Install with: pip install Pillow numpy")
    
    # Generate random RGB image
    random_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(random_array, 'RGB')
    
    # Convert to PNG bytes
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def image_bytes_to_data_uri(image_bytes: bytes) -> str:
    """Convert image bytes to base64 data URI."""
    base64_data = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/png;base64,{base64_data}"


def select_image_config(bucket_config: dict) -> tuple:
    """
    Select an image configuration based on probability distribution.
    
    Returns (height, width, num_images) tuple.
    """
    configs = list(bucket_config.keys())
    probs = list(bucket_config.values())
    
    # Use random.choices for weighted selection
    selected = random.choices(configs, weights=probs, k=1)[0]
    return selected


def get_prompt(prompt_type: str) -> tuple:
    """
    Get a prompt based on the specified type.
    
    Args:
        prompt_type: One of 'description', 'json', or 'mixed'
    
    Returns:
        Tuple of (prompt_text, prompt_category)
    """
    if prompt_type == 'description':
        return random.choice(DESCRIPTION_PROMPTS), 'description'
    elif prompt_type == 'json':
        return random.choice(JSON_EXTRACTION_PROMPTS), 'json'
    else:  # mixed
        if random.random() < 0.5:
            return random.choice(DESCRIPTION_PROMPTS), 'description'
        else:
            return random.choice(JSON_EXTRACTION_PROMPTS), 'json'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark llama-server completion performance (multimodal by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multimodal with default buckets and mixed prompts (DEFAULT)
  python bench_llama_cpp_text_and_image.py -s http://localhost:30000 -n 100

  # Multimodal with JSON extraction prompts only
  python bench_llama_cpp_text_and_image.py -s http://localhost:30000 -n 100 --prompt-type json

  # Multimodal with description prompts only  
  python bench_llama_cpp_text_and_image.py -s http://localhost:30000 -n 100 --prompt-type description

  # Multimodal with custom buckets
  python bench_llama_cpp_text_and_image.py -s http://localhost:30000 -n 100 \\
      --mm-bucket-config '{(1080, 1920, 1): 0.5, (1080, 1080, 1): 0.5}'

  # Text-only mode (no images)
  python bench_llama_cpp_text_and_image.py -s http://localhost:30000 -n 100 --text-only
        """
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
        help="Input token range [min, max] (for text-only mode)"
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
    parser.add_argument(
        "--mm-bucket-config",
        type=str,
        default=None,
        help="Multimodal image bucket config. Default uses standard mix "
             "(phone screenshots, mac screenshots, instagram squares, phone photos). "
             "Specify custom dict: '{(height, width, num_images): probability, ...}' "
             "Example: '{(1080, 1920, 1): 0.5, (1080, 1080, 1): 0.5}'"
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Run text-only benchmark (no images). Multimodal is the default."
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=['description', 'json', 'mixed'],
        default='mixed',
        help="Type of prompts to use: 'description' (image descriptions), "
             "'json' (structured JSON extraction), or 'mixed' (50/50 both). Default: mixed"
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


def run_single_text(server_url: str, prompt: str, n_predict: int, timeout: int) -> dict:
    """Run a single text-only completion and return timings."""
    response = requests.post(
        f"{server_url}/completion",
        json={"prompt": prompt, "n_predict": n_predict, "stream": False, "cache_prompt": False},
        headers={"Content-Type": "application/json"},
        timeout=timeout
    )
    return response.json().get("timings", {})


def run_single_multimodal(server_url: str, image_data_uris: list, text_prompt: str, 
                          n_predict: int, timeout: int) -> dict:
    """
    Run a single multimodal completion using the OpenAI-compatible chat API.
    
    Returns timing information extracted from the response.
    """
    # Build content array with images first, then text
    content = []
    for data_uri in image_data_uris:
        content.append({
            "type": "image_url",
            "image_url": {"url": data_uri}
        })
    content.append({
        "type": "text",
        "text": text_prompt
    })
    
    messages = [
        {"role": "system", "content": "You are an assistant who perfectly describes images and extracts structured data when requested."},
        {"role": "user", "content": content}
    ]
    
    payload = {
        "messages": messages,
        "max_tokens": n_predict,
        "stream": False,
    }
    
    start_time = time.perf_counter()
    
    response = requests.post(
        f"{server_url}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout
    )
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    data = response.json()
    
    # Extract timing info - chat completions API format
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    
    # Try to get detailed timings if available (llama-server specific)
    timings = data.get("timings", {})
    
    if timings:
        # llama-server provides detailed timings
        return timings
    else:
        # Calculate approximate timings from usage info
        # This is less accurate but works for OpenAI-compatible servers
        if completion_tokens > 0 and total_time > 0:
            # Estimate: assume 10% of time is prompt processing for multimodal
            prompt_time = total_time * 0.1
            gen_time = total_time * 0.9
            
            return {
                "prompt_n": prompt_tokens,
                "predicted_n": completion_tokens,
                "prompt_per_second": prompt_tokens / prompt_time if prompt_time > 0 else 0,
                "predicted_per_second": completion_tokens / gen_time if gen_time > 0 else 0,
                "total_time_ms": total_time * 1000,
            }
        else:
            return {
                "prompt_n": prompt_tokens,
                "predicted_n": completion_tokens,
                "prompt_per_second": 0,
                "predicted_per_second": 0,
            }


def calculate_per_bucket_stats(measured: list) -> dict:
    """
    Calculate per-bucket statistics for multimodal results.
    
    Returns dict keyed by (height, width, num_images) with stats for each bucket.
    """
    # Group results by bucket
    buckets = defaultdict(list)
    for r in measured:
        if "image_height" in r and "image_width" in r and "num_images" in r:
            bucket_key = (r["image_height"], r["image_width"], r["num_images"])
            buckets[bucket_key].append(r)
    
    # Calculate stats for each bucket
    per_bucket_stats = {}
    for bucket_key, results in buckets.items():
        total_prompt_tokens = sum(r["prompt_n"] for r in results)
        total_generated_tokens = sum(r["predicted_n"] for r in results)
        
        total_prompt_time = sum(
            r["prompt_n"] / r["prompt_per_sec"] 
            for r in results if r["prompt_per_sec"] > 0
        )
        total_gen_time = sum(
            r["predicted_n"] / r["predicted_per_sec"] 
            for r in results if r["predicted_per_sec"] > 0
        )
        
        true_prompt_throughput = total_prompt_tokens / total_prompt_time if total_prompt_time > 0 else 0
        true_gen_throughput = total_generated_tokens / total_gen_time if total_gen_time > 0 else 0
        
        avg_prompt_speed = sum(r["prompt_per_sec"] for r in results) / len(results) if results else 0
        avg_gen_speed = sum(r["predicted_per_sec"] for r in results) / len(results) if results else 0
        
        total_pixels = sum(r.get("total_pixels", 0) for r in results)
        
        per_bucket_stats[bucket_key] = {
            "count": len(results),
            "total_prompt_tokens": total_prompt_tokens,
            "total_generated_tokens": total_generated_tokens,
            "avg_prompt_tokens": total_prompt_tokens / len(results) if results else 0,
            "avg_generated_tokens": total_generated_tokens / len(results) if results else 0,
            "true_prompt_throughput": true_prompt_throughput,
            "true_gen_throughput": true_gen_throughput,
            "avg_prompt_speed": avg_prompt_speed,
            "avg_gen_speed": avg_gen_speed,
            "total_pixels": total_pixels,
            "total_prompt_time": total_prompt_time,
            "total_gen_time": total_gen_time,
        }
    
    return per_bucket_stats


def calculate_per_prompt_type_stats(measured: list) -> dict:
    """
    Calculate per-prompt-type statistics (description vs json).
    
    Returns dict keyed by prompt_type with stats for each type.
    """
    # Group results by prompt type
    by_type = defaultdict(list)
    for r in measured:
        prompt_type = r.get("prompt_type", "unknown")
        by_type[prompt_type].append(r)
    
    # Calculate stats for each type
    per_type_stats = {}
    for prompt_type, results in by_type.items():
        total_prompt_tokens = sum(r["prompt_n"] for r in results)
        total_generated_tokens = sum(r["predicted_n"] for r in results)
        
        total_prompt_time = sum(
            r["prompt_n"] / r["prompt_per_sec"] 
            for r in results if r["prompt_per_sec"] > 0
        )
        total_gen_time = sum(
            r["predicted_n"] / r["predicted_per_sec"] 
            for r in results if r["predicted_per_sec"] > 0
        )
        
        true_prompt_throughput = total_prompt_tokens / total_prompt_time if total_prompt_time > 0 else 0
        true_gen_throughput = total_generated_tokens / total_gen_time if total_gen_time > 0 else 0
        
        avg_prompt_speed = sum(r["prompt_per_sec"] for r in results) / len(results) if results else 0
        avg_gen_speed = sum(r["predicted_per_sec"] for r in results) / len(results) if results else 0
        
        per_type_stats[prompt_type] = {
            "count": len(results),
            "total_prompt_tokens": total_prompt_tokens,
            "total_generated_tokens": total_generated_tokens,
            "avg_prompt_tokens": total_prompt_tokens / len(results) if results else 0,
            "avg_generated_tokens": total_generated_tokens / len(results) if results else 0,
            "true_prompt_throughput": true_prompt_throughput,
            "true_gen_throughput": true_gen_throughput,
            "avg_prompt_speed": avg_prompt_speed,
            "avg_gen_speed": avg_gen_speed,
            "total_prompt_time": total_prompt_time,
            "total_gen_time": total_gen_time,
        }
    
    return per_type_stats


def save_csv(args, measured: list, gpu_name: str, model_info: dict, timestamp: str,
             is_multimodal: bool, bucket_config: dict = None, per_bucket_stats: dict = None,
             per_prompt_type_stats: dict = None) -> str:
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
    test_type = "multimodal" if is_multimodal else "text_only"
    csv_file = f"{output_dir}/llama_server_{gpu_name}_{model_short}_{test_type}_samples_{args.num_prompts}_{timestamp}.csv"
    
    with open(csv_file, "w") as f:
        f.write("metric,value,unit\n")
        f.write(f"device_sanitized,{gpu_name},\n")
        f.write(f"server,llama-server,\n")
        f.write(f"server_url,{args.server_url},\n")
        f.write(f"model_name,{model_info['model_name']},\n")
        f.write(f"model_path,{model_info['model_path']},\n")
        f.write(f"timestamp,{timestamp},\n")
        f.write(f"test_type,{test_type},\n")
        f.write(f"prompt_type,{args.prompt_type},\n")
        f.write(f"num_prompts,{args.num_prompts},\n")
        f.write(f"warmup_runs,{args.warmups},\n")
        
        if is_multimodal:
            # Format bucket config for CSV
            bucket_str = str(bucket_config).replace(",", ";") if bucket_config else "N/A"
            f.write(f"mm_bucket_config,\"{bucket_str}\",\n")
            
            # Calculate image statistics
            total_images = sum(r.get("num_images", 0) for r in measured)
            total_pixels = sum(r.get("total_pixels", 0) for r in measured)
            avg_images_per_req = total_images / len(measured) if measured else 0
            
            f.write(f"total_images,{total_images},images\n")
            f.write(f"total_pixels,{total_pixels},pixels\n")
            f.write(f"avg_images_per_request,{avg_images_per_req:.2f},images\n")
        else:
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
        
        # Write per-bucket metrics for multimodal
        if is_multimodal and per_bucket_stats:
            f.write("\n# Per-bucket metrics\n")
            for bucket_key, stats in per_bucket_stats.items():
                h, w, n = bucket_key
                prefix = f"bucket_{h}x{w}x{n}"
                f.write(f"{prefix}_count,{stats['count']},requests\n")
                f.write(f"{prefix}_total_prompt_tokens,{stats['total_prompt_tokens']},tokens\n")
                f.write(f"{prefix}_total_generated_tokens,{stats['total_generated_tokens']},tokens\n")
                f.write(f"{prefix}_avg_prompt_tokens,{stats['avg_prompt_tokens']:.1f},tokens\n")
                f.write(f"{prefix}_avg_generated_tokens,{stats['avg_generated_tokens']:.1f},tokens\n")
                f.write(f"{prefix}_true_prompt_throughput,{stats['true_prompt_throughput']:.2f},tok/s\n")
                f.write(f"{prefix}_true_gen_throughput,{stats['true_gen_throughput']:.2f},tok/s\n")
                f.write(f"{prefix}_avg_prompt_speed,{stats['avg_prompt_speed']:.2f},tok/s\n")
                f.write(f"{prefix}_avg_gen_speed,{stats['avg_gen_speed']:.2f},tok/s\n")
                f.write(f"{prefix}_total_pixels,{stats['total_pixels']},pixels\n")
        
        # Write per-prompt-type metrics for multimodal
        if is_multimodal and per_prompt_type_stats:
            f.write("\n# Per-prompt-type metrics\n")
            for prompt_type, stats in per_prompt_type_stats.items():
                prefix = f"prompt_{prompt_type}"
                f.write(f"{prefix}_count,{stats['count']},requests\n")
                f.write(f"{prefix}_total_prompt_tokens,{stats['total_prompt_tokens']},tokens\n")
                f.write(f"{prefix}_total_generated_tokens,{stats['total_generated_tokens']},tokens\n")
                f.write(f"{prefix}_avg_prompt_tokens,{stats['avg_prompt_tokens']:.1f},tokens\n")
                f.write(f"{prefix}_avg_generated_tokens,{stats['avg_generated_tokens']:.1f},tokens\n")
                f.write(f"{prefix}_true_prompt_throughput,{stats['true_prompt_throughput']:.2f},tok/s\n")
                f.write(f"{prefix}_true_gen_throughput,{stats['true_gen_throughput']:.2f},tok/s\n")
                f.write(f"{prefix}_avg_prompt_speed,{stats['avg_prompt_speed']:.2f},tok/s\n")
                f.write(f"{prefix}_avg_gen_speed,{stats['avg_gen_speed']:.2f},tok/s\n")
    
    print(f"\nCSV saved to: {csv_file}")
    return csv_file


def main():
    args = parse_args()
    
    random.seed(args.seed)
    if HAS_PIL:
        np.random.seed(args.seed)
    
    # Determine mode: multimodal is default unless --text-only is specified
    is_multimodal = not args.text_only
    bucket_config = None
    
    if is_multimodal:
        bucket_config = parse_bucket_config(args.mm_bucket_config)
        
        if not HAS_PIL:
            print("ERROR: Multimodal benchmarking requires PIL and numpy.")
            print("Install with: pip install Pillow numpy --break-system-packages")
            return
    
    gpu_name = get_gpu_name()
    model_info = get_model_info(args.server_url)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_runs = args.num_prompts + args.warmups
    
    print("=" * 60)
    print("LLAMA-SERVER BENCHMARK (Text + Image)")
    print("=" * 60)
    print(f"GPU: {gpu_name}")
    print(f"Model: {model_info['model_name']}")
    print(f"Server: {args.server_url}")
    print(f"Mode: {'Multimodal' if is_multimodal else 'Text-only'}")
    
    if is_multimodal:
        print(f"Prompt type: {args.prompt_type}")
    
    print(f"Runs: {total_runs} ({args.warmups} warmups, {args.num_prompts} measured)")
    
    if is_multimodal:
        print(f"Image buckets:")
        for (h, w, n), prob in bucket_config.items():
            # Try to get human-readable label
            label = DEFAULT_MM_BUCKET_LABELS.get((h, w, n), "")
            if label:
                print(f"  {h}x{w} ({n} img) - {label}: {prob*100:.1f}%")
            else:
                print(f"  {h}x{w} ({n} image{'s' if n > 1 else ''}): {prob*100:.1f}%")
    else:
        print(f"Input tokens: {args.input_range[0]}-{args.input_range[1]}")
    
    print(f"Output tokens: {args.output_range[0]}-{args.output_range[1]}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    results = []
    
    pbar = tqdm(range(total_runs), desc="Benchmarking", unit="req")
    
    for i in pbar:
        output_tokens = random.randint(args.output_range[0], args.output_range[1])
        is_warmup = i < args.warmups
        
        try:
            if is_multimodal:
                # Select image configuration
                height, width, num_images = select_image_config(bucket_config)
                
                # Generate synthetic images
                image_data_uris = []
                for _ in range(num_images):
                    img_bytes = generate_synthetic_image(height, width)
                    data_uri = image_bytes_to_data_uri(img_bytes)
                    image_data_uris.append(data_uri)
                
                # Select prompt based on type
                text_prompt, prompt_category = get_prompt(args.prompt_type)
                
                # Run multimodal completion
                timings = run_single_multimodal(
                    args.server_url, image_data_uris, text_prompt, output_tokens, args.timeout
                )
                
                result = {
                    "prompt_n": timings.get("prompt_n", 0),
                    "predicted_n": timings.get("predicted_n", 0),
                    "prompt_per_sec": timings.get("prompt_per_second", 0),
                    "predicted_per_sec": timings.get("predicted_per_second", 0),
                    "is_warmup": is_warmup,
                    "num_images": num_images,
                    "image_height": height,
                    "image_width": width,
                    "total_pixels": height * width * num_images,
                    "prompt_type": prompt_category,
                }
                
            else:
                # Text-only mode
                input_tokens = random.randint(args.input_range[0], args.input_range[1])
                prompt = generate_prompt(input_tokens)
                
                timings = run_single_text(args.server_url, prompt, output_tokens, args.timeout)
                
                result = {
                    "prompt_n": timings.get("prompt_n", 0),
                    "predicted_n": timings.get("predicted_n", 0),
                    "prompt_per_sec": timings.get("prompt_per_second", 0),
                    "predicted_per_sec": timings.get("predicted_per_second", 0),
                    "is_warmup": is_warmup,
                }
            
            results.append(result)
            
            label = "WRM" if is_warmup else "RUN"
            postfix = {
                "type": label,
                "in": f"{result['prompt_n']}tok",
                "out": f"{result['predicted_n']}tok",
                "pp": f"{result['prompt_per_sec']:.0f}t/s",
                "tg": f"{result['predicted_per_sec']:.1f}t/s",
            }
            
            if is_multimodal:
                postfix["img"] = f"{result['image_height']}x{result['image_width']}"
                postfix["ptype"] = result["prompt_type"][:4]
            
            pbar.set_postfix(postfix)
            
        except requests.RequestException as e:
            pbar.write(f"ERROR on run {i+1}: {e}")
            continue
        except Exception as e:
            pbar.write(f"ERROR on run {i+1}: {e}")
            continue
    
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
    print(f"RESULTS (excluding warmups) - {'Multimodal' if is_multimodal else 'Text-only'}")
    print("=" * 60)
    print(f"Model: {model_info['model_name']}")
    
    if is_multimodal:
        total_images = sum(r.get("num_images", 0) for r in measured)
        total_pixels = sum(r.get("total_pixels", 0) for r in measured)
        print(f"Total images processed:   {total_images:,}")
        print(f"Total pixels:             {total_pixels:,}")
    
    print(f"Total prompt tokens:      {total_prompt_tokens:,}")
    print(f"Total generated tokens:   {total_generated_tokens:,}")
    print(f"True prompt throughput:   {true_prompt_throughput:.2f} tok/s")
    print(f"True gen throughput:      {true_gen_throughput:.2f} tok/s")
    print(f"Avg prompt speed:         {avg_prompt_speed:.2f} tok/s (per-run average)")
    print(f"Avg gen speed:            {avg_gen_speed:.2f} tok/s (per-run average)")
    
    # Per-bucket metrics for multimodal
    per_bucket_stats = {}
    per_prompt_type_stats = {}
    
    if is_multimodal:
        per_bucket_stats = calculate_per_bucket_stats(measured)
        print("\n" + "-" * 60)
        print("PER-BUCKET METRICS")
        print("-" * 60)
        
        for bucket_key, stats in per_bucket_stats.items():
            h, w, n = bucket_key
            label = DEFAULT_MM_BUCKET_LABELS.get(bucket_key, f"{h}x{w} ({n} img)")
            print(f"\n{label}:")
            print(f"  Requests:             {stats['count']:,}")
            print(f"  Total prompt tokens:  {stats['total_prompt_tokens']:,}")
            print(f"  Total gen tokens:     {stats['total_generated_tokens']:,}")
            print(f"  Avg prompt tokens:    {stats['avg_prompt_tokens']:.1f}")
            print(f"  Avg gen tokens:       {stats['avg_generated_tokens']:.1f}")
            print(f"  Prompt throughput:    {stats['true_prompt_throughput']:.2f} tok/s")
            print(f"  Gen throughput:       {stats['true_gen_throughput']:.2f} tok/s")
            print(f"  Avg prompt speed:     {stats['avg_prompt_speed']:.2f} tok/s")
            print(f"  Avg gen speed:        {stats['avg_gen_speed']:.2f} tok/s")
        
        # Per-prompt-type metrics
        per_prompt_type_stats = calculate_per_prompt_type_stats(measured)
        print("\n" + "-" * 60)
        print("PER-PROMPT-TYPE METRICS")
        print("-" * 60)
        
        for prompt_type, stats in per_prompt_type_stats.items():
            label = "Description prompts" if prompt_type == "description" else "JSON extraction prompts"
            print(f"\n{label}:")
            print(f"  Requests:             {stats['count']:,}")
            print(f"  Total prompt tokens:  {stats['total_prompt_tokens']:,}")
            print(f"  Total gen tokens:     {stats['total_generated_tokens']:,}")
            print(f"  Avg prompt tokens:    {stats['avg_prompt_tokens']:.1f}")
            print(f"  Avg gen tokens:       {stats['avg_generated_tokens']:.1f}")
            print(f"  Prompt throughput:    {stats['true_prompt_throughput']:.2f} tok/s")
            print(f"  Gen throughput:       {stats['true_gen_throughput']:.2f} tok/s")
            print(f"  Avg prompt speed:     {stats['avg_prompt_speed']:.2f} tok/s")
            print(f"  Avg gen speed:        {stats['avg_gen_speed']:.2f} tok/s")
    
    save_csv(args, measured, gpu_name, model_info, timestamp, is_multimodal, 
             bucket_config, per_bucket_stats, per_prompt_type_stats)


if __name__ == "__main__":
    main()