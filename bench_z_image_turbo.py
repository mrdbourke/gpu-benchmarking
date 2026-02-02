#!/usr/bin/env python3
"""bench_z_image_turbo.py - Run on: client machine with GPU

Benchmark script for Z-Image-Turbo image generation using Hugging Face Diffusers.
Measures image generation throughput (images/second) on target hardware.

Requirements:
    pip install git+https://github.com/huggingface/diffusers --break-system-packages
    pip install torch transformers accelerate tqdm sentencepiece --break-system-packages

Usage:
    python bench_zimage.py --num-images 100 --warmups 10 --steps 9
"""

import argparse
import random
import subprocess
import re
import os
import time
import gc
from datetime import datetime
from typing import Optional

import torch
from tqdm import tqdm

# Diverse prompt templates for benchmarking
PROMPT_TEMPLATES = [
    "A photorealistic portrait of a {adj} {subject} in {setting}",
    "A {style} painting of {subject} during {time}",
    "A {adj} {subject} {action} in a {setting}",
    "{subject} with {detail}, {style} style, {lighting} lighting",
    "A cinematic shot of {subject} in {setting}, {mood} atmosphere",
    "Professional photograph of {adj} {subject}, {lighting} lighting",
    "Digital art of {subject} {action}, {style} style",
    "A {mood} scene featuring {subject} in {setting}",
    "{style} illustration of {adj} {subject} with {detail}",
    "Hyperrealistic {subject} in {setting}, {time}, {lighting} lighting",
]

SUBJECTS = [
    "cat", "dog", "robot", "astronaut", "wizard", "dragon", "forest", "city",
    "mountain", "ocean", "spaceship", "castle", "garden", "desert", "warrior",
    "scientist", "musician", "dancer", "chef", "explorer", "phoenix", "owl",
    "wolf", "bear", "eagle", "lighthouse", "windmill", "train", "motorcycle",
]

ADJECTIVES = [
    "majestic", "ancient", "futuristic", "mysterious", "glowing", "ethereal",
    "vibrant", "serene", "dramatic", "whimsical", "elegant", "rustic",
    "crystalline", "mechanical", "organic", "luminous", "shadowy", "golden",
]

SETTINGS = [
    "enchanted forest", "cyberpunk city", "underwater kingdom", "mountain peak",
    "desert oasis", "space station", "medieval village", "tropical beach",
    "snowy tundra", "volcanic island", "floating islands", "ancient ruins",
    "neon-lit street", "misty valley", "crystal cave", "steampunk workshop",
]

STYLES = [
    "impressionist", "art nouveau", "minimalist", "baroque", "surrealist",
    "watercolor", "oil painting", "digital art", "anime", "photorealistic",
    "concept art", "comic book", "vintage poster", "renaissance", "abstract",
]

TIMES = [
    "golden hour", "blue hour", "midnight", "dawn", "sunset", "noon",
    "twilight", "overcast day", "stormy afternoon", "starlit night",
]

LIGHTING = [
    "soft", "dramatic", "natural", "studio", "rim", "volumetric",
    "cinematic", "neon", "candlelit", "moonlit", "harsh", "diffused",
]

MOODS = [
    "peaceful", "epic", "mysterious", "joyful", "melancholic", "intense",
    "dreamlike", "nostalgic", "adventurous", "romantic", "eerie", "hopeful",
]

ACTIONS = [
    "standing majestically", "walking slowly", "flying through the air",
    "resting peacefully", "gazing into the distance", "emerging from shadows",
    "dancing gracefully", "meditating quietly", "exploring curiously",
]

DETAILS = [
    "intricate patterns", "glowing eyes", "ornate armor", "flowing robes",
    "mechanical parts", "crystal formations", "ancient symbols", "vine patterns",
    "metallic sheen", "ethereal glow", "scattered petals", "sparkling dust",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Z-Image-Turbo image generation performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-id", "-m",
        type=str,
        default="Tongyi-MAI/Z-Image-Turbo",
        help="Hugging Face model ID"
    )
    parser.add_argument(
        "--num-images", "-n",
        type=int,
        default=100,
        help="Number of measured image generations (excluding warmups), default: 100"
    )
    parser.add_argument(
        "--warmups", "-w",
        type=int,
        default=10,
        help="Number of warmup runs to ignore"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=9,
        help="Number of inference steps per image (9 steps = 8 DiT forwards for Turbo)"
    )
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        nargs=2,
        default=[1024, 1024],
        metavar=("WIDTH", "HEIGHT"),
        help="Image resolution [width, height]"
    )
    parser.add_argument(
        "--guidance-scale", "-g",
        type=float,
        default=0.0,
        help="Guidance scale for generation (must be 0.0 for Turbo models)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Data type for model weights"
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        choices=["sdpa", "flash", "flash3"],
        default="sdpa",
        help="Attention backend: sdpa (default), flash (Flash-Attention-2), flash3 (Flash-Attention-3)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save generated images (for verification)"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save every N images when --save-images is enabled"
    )

    return parser.parse_args()


def get_gpu_info() -> dict:
    """Detect GPU name and VRAM, sanitize for filenames."""
    gpu_info = {
        "name": "Unknown_GPU",
        "name_sanitized": "Unknown_GPU",
        "vram_gb": 0,
    }

    try:
        # Get GPU name
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        raw_name = result.stdout.strip().split("\n")[0]
        gpu_info["name"] = raw_name
        sanitized = re.sub(r'[^a-zA-Z0-9]', '_', raw_name)
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
        gpu_info["name_sanitized"] = sanitized if sanitized else "Unknown_GPU"

        # Get VRAM
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        vram_mb = int(result.stdout.strip().split("\n")[0])
        gpu_info["vram_gb"] = round(vram_mb / 1024, 1)

    except Exception as e:
        print(f"Warning: Could not detect GPU info: {e}")

    return gpu_info


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return dtype_map[dtype_str]


def generate_prompt(seed: int) -> str:
    """Generate a random prompt from templates."""
    rng = random.Random(seed)

    template = rng.choice(PROMPT_TEMPLATES)

    # Fill in template placeholders
    prompt = template.format(
        subject=rng.choice(SUBJECTS),
        adj=rng.choice(ADJECTIVES),
        setting=rng.choice(SETTINGS),
        style=rng.choice(STYLES),
        time=rng.choice(TIMES),
        lighting=rng.choice(LIGHTING),
        mood=rng.choice(MOODS),
        action=rng.choice(ACTIONS),
        detail=rng.choice(DETAILS),
    )

    return prompt


def load_pipeline(args):
    """Load the Z-Image-Turbo pipeline."""
    from diffusers import ZImagePipeline

    print(f"Loading model: {args.model_id}")
    print(f"Data type: {args.dtype}")

    dtype = get_dtype(args.dtype)

    pipe = ZImagePipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    ).to("cuda")

    print(f"[INFO] Model on device: {pipe.device}")

    # Set attention backend
    attention_backend_map = {
        "sdpa": None,  # Default, no change needed
        "flash": "flash",
        "flash3": "_flash_3",
    }
    if args.attention_backend != "sdpa":
        backend = attention_backend_map[args.attention_backend]
        print(f"Setting attention backend to: {args.attention_backend}")
        try:
            pipe.transformer.set_attention_backend(backend)
        except Exception as e:
            print(f"Warning: Could not set attention backend: {e}")

    return pipe


def run_single_generation(
    pipe,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
) -> tuple[float, "PIL.Image.Image"]:
    """Run a single image generation and return timing and image."""

    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Sync CUDA before timing
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    # Sync CUDA after generation
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    elapsed = end_time - start_time

    return elapsed, image


def save_csv(args, results: list, gpu_info: dict, timestamp: str) -> str:
    """Save results to CSV in standard naming format."""

    measured = [r for r in results if not r["is_warmup"]]

    total_images = len(measured)
    total_time = sum(r["elapsed_sec"] for r in measured)
    avg_time = total_time / total_images if total_images > 0 else 0
    images_per_sec = total_images / total_time if total_time > 0 else 0

    # Per-image statistics
    times = [r["elapsed_sec"] for r in measured]
    min_time = min(times) if times else 0
    max_time = max(times) if times else 0
    median_time = sorted(times)[len(times) // 2] if times else 0

    # Calculate steps/sec (note: for Z-Image-Turbo, 9 steps = 8 DiT forwards)
    total_steps = sum(r["steps"] for r in measured)
    steps_per_sec = total_steps / total_time if total_time > 0 else 0

    # Calculate DiT forwards per second (steps - 1 for Turbo models)
    dit_forwards = sum(r["steps"] - 1 for r in measured)
    dit_forwards_per_sec = dit_forwards / total_time if total_time > 0 else 0

    output_dir = f"{args.output_dir}/{gpu_info['name_sanitized']}"
    os.makedirs(output_dir, exist_ok=True)

    model_short = args.model_id.split("/")[-1][:30]
    model_sanitized = re.sub(r'[^a-zA-Z0-9]', '_', model_short)
    model_sanitized = re.sub(r'_+', '_', model_sanitized).strip('_')

    csv_file = f"{output_dir}/zimage_bench_{gpu_info['name_sanitized']}_{model_sanitized}_n{args.num_images}_{timestamp}.csv"

    with open(csv_file, "w") as f:
        f.write("metric,value,unit\n")
        f.write(f"device,{gpu_info['name']},\n")
        f.write(f"device_sanitized,{gpu_info['name_sanitized']},\n")
        f.write(f"vram_gb,{gpu_info['vram_gb']},GB\n")
        f.write(f"model_id,{args.model_id},\n")
        f.write(f"timestamp,{timestamp},\n")
        f.write(f"test_type,image_generation,\n")
        f.write(f"num_images,{args.num_images},\n")
        f.write(f"warmup_runs,{args.warmups},\n")
        f.write(f"inference_steps,{args.steps},\n")
        f.write(f"dit_forwards_per_image,{args.steps - 1},\n")
        f.write(f"resolution,{args.resolution[0]}x{args.resolution[1]},pixels\n")
        f.write(f"guidance_scale,{args.guidance_scale},\n")
        f.write(f"dtype,{args.dtype},\n")
        f.write(f"attention_backend,{args.attention_backend},\n")
        f.write(f"seed,{args.seed},\n")
        f.write(f"total_images,{total_images},images\n")
        f.write(f"total_time,{total_time:.2f},seconds\n")
        f.write(f"images_per_second,{images_per_sec:.4f},img/s\n")
        f.write(f"seconds_per_image,{avg_time:.4f},s/img\n")
        f.write(f"min_time,{min_time:.4f},seconds\n")
        f.write(f"max_time,{max_time:.4f},seconds\n")
        f.write(f"median_time,{median_time:.4f},seconds\n")
        f.write(f"total_steps,{total_steps},steps\n")
        f.write(f"steps_per_second,{steps_per_sec:.2f},steps/s\n")
        f.write(f"total_dit_forwards,{dit_forwards},forwards\n")
        f.write(f"dit_forwards_per_second,{dit_forwards_per_sec:.2f},forwards/s\n")

    print(f"\nCSV saved to: {csv_file}")
    return csv_file


def main():
    args = parse_args()

    # Warn if guidance_scale is not 0 for Turbo model
    if "Turbo" in args.model_id and args.guidance_scale != 0.0:
        print(f"WARNING: guidance_scale should be 0.0 for Turbo models, got {args.guidance_scale}")

    random.seed(args.seed)

    gpu_info = get_gpu_info()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_runs = args.num_images + args.warmups
    width, height = args.resolution

    print("=" * 60)
    print("Z-IMAGE-TURBO IMAGE GENERATION BENCHMARK")
    print("=" * 60)
    print(f"GPU: {gpu_info['name']} ({gpu_info['vram_gb']} GB VRAM)")
    print(f"Model: {args.model_id}")
    print(f"Resolution: {width}x{height}")
    print(f"Inference steps: {args.steps} ({args.steps - 1} DiT forwards)")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Data type: {args.dtype}")
    print(f"Attention backend: {args.attention_backend}")
    print(f"Runs: {total_runs} ({args.warmups} warmups, {args.num_images} measured)")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # Load pipeline
    pipe = load_pipeline(args)

    # Create image output directory if saving
    if args.save_images:
        img_dir = f"{args.output_dir}/{gpu_info['name_sanitized']}/images_{timestamp}"
        os.makedirs(img_dir, exist_ok=True)
        print(f"Saving images to: {img_dir}")

    results = []
    pbar = tqdm(range(total_runs), desc="Generating", unit="img")

    for i in pbar:
        is_warmup = i < args.warmups

        # Generate unique seed for this image
        img_seed = args.seed + i
        prompt = generate_prompt(img_seed)

        try:
            elapsed, image = run_single_generation(
                pipe=pipe,
                prompt=prompt,
                width=width,
                height=height,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=img_seed,
            )
        except Exception as e:
            pbar.write(f"ERROR on run {i + 1}: {e}")
            continue

        result = {
            "run_idx": i,
            "elapsed_sec": elapsed,
            "steps": args.steps,
            "is_warmup": is_warmup,
            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
        }
        results.append(result)

        # Save image if requested
        if args.save_images and not is_warmup and (i - args.warmups) % args.save_every == 0:
            img_path = f"{img_dir}/img_{i:04d}.png"
            image.save(img_path)

        label = "WRM" if is_warmup else "RUN"
        img_per_sec = 1.0 / elapsed if elapsed > 0 else 0
        pbar.set_postfix({
            "type": label,
            "time": f"{elapsed:.2f}s",
            "img/s": f"{img_per_sec:.3f}",
        })

    pbar.close()

    # Clear CUDA cache
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # === RESULTS (excluding warmups) ===
    measured = [r for r in results if not r["is_warmup"]]

    if not measured:
        print("ERROR: No successful measured runs!")
        return

    total_images = len(measured)
    total_time = sum(r["elapsed_sec"] for r in measured)
    avg_time = total_time / total_images
    images_per_sec = total_images / total_time

    times = [r["elapsed_sec"] for r in measured]
    min_time = min(times)
    max_time = max(times)
    median_time = sorted(times)[len(times) // 2]

    total_steps = sum(r["steps"] for r in measured)
    steps_per_sec = total_steps / total_time

    # DiT forwards (steps - 1 for Turbo)
    dit_forwards = sum(r["steps"] - 1 for r in measured)
    dit_forwards_per_sec = dit_forwards / total_time

    print("\n" + "=" * 60)
    print("RESULTS (excluding warmups)")
    print("=" * 60)
    print(f"Model: {args.model_id}")
    print(f"Resolution: {width}x{height}")
    print(f"Inference steps: {args.steps} ({args.steps - 1} DiT forwards)")
    print(f"Total images generated:   {total_images}")
    print(f"Total time:               {total_time:.2f} seconds")
    print("-" * 40)
    print(f"Images per second:        {images_per_sec:.4f} img/s")
    print(f"Seconds per image:        {avg_time:.4f} s/img")
    print(f"Min time:                 {min_time:.4f} s")
    print(f"Max time:                 {max_time:.4f} s")
    print(f"Median time:              {median_time:.4f} s")
    print("-" * 40)
    print(f"Total inference steps:    {total_steps}")
    print(f"Steps per second:         {steps_per_sec:.2f} steps/s")
    print(f"Total DiT forwards:       {dit_forwards}")
    print(f"DiT forwards per second:  {dit_forwards_per_sec:.2f} forwards/s")

    save_csv(args, results, gpu_info, timestamp)


if __name__ == "__main__":
    main()
