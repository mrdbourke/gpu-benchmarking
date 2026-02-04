#!/usr/bin/env python3
"""Shared helpers for reading and summarizing benchmark CSV results."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

KNOWN_FAMILIES: List[str] = [
    "vLLM",
    "llama_server",
    "flux_bench",
    "zimage_bench",
    "object_detection_bench",
    "llm_finetune_bench",
]

CONFIG_FIELDS: Dict[str, List[str]] = {
    "vLLM": [
        "model",
        "test_type",
        "num_prompts",
        "max_concurrency",
        "input_len",
        "output_len",
        "images_per_request",
        "image_bucket",
        "range_ratio",
        "thinking_disabled",
    ],
    "llama_server": [
        "model_name",
        "test_type",
        "prompt_type",
        "num_prompts",
        "input_token_range",
        "output_token_range",
        "avg_images_per_request",
        "mm_bucket_config",
    ],
    "flux_bench": [
        "model_id",
        "num_images",
        "inference_steps",
        "resolution",
        "guidance_scale",
        "dtype",
    ],
    "zimage_bench": [
        "model_id",
        "num_images",
        "inference_steps",
        "dit_forwards_per_image",
        "resolution",
        "guidance_scale",
        "dtype",
        "attention_backend",
    ],
    "object_detection_bench": [
        "model_name",
        "dataset_name",
        "num_epochs",
        "batch_size",
        "image_size",
        "fp16",
    ],
    "llm_finetune_bench": [
        "model_name",
        "dataset_name",
        "num_epochs",
        "batch_size",
        "max_length",
        "gradient_checkpointing",
    ],
}

COMPARE_METRICS: Dict[str, List[str]] = {
    "vLLM": [
        "output_token_throughput",
        "request_throughput",
        "total_token_throughput",
        "ttft_mean",
        "tpot_mean",
        "itl_mean",
        "benchmark_duration",
    ],
    "llama_server": [
        "true_prompt_throughput",
        "true_gen_throughput",
        "avg_prompt_speed",
        "avg_gen_speed",
        "total_runtime",
    ],
    "flux_bench": [
        "images_per_second",
        "seconds_per_image",
        "total_time",
        "steps_per_second",
    ],
    "zimage_bench": [
        "images_per_second",
        "seconds_per_image",
        "total_time",
        "steps_per_second",
        "dit_forwards_per_second",
    ],
    "object_detection_bench": [
        "samples_per_second",
        "total_train_time",
        "total_train_time_minutes",
    ],
    "llm_finetune_bench": [
        "total_train_time",
        "total_train_time_minutes",
        "inference_samples_per_second",
        "inference_tokens_per_second",
    ],
}

FALLBACK_METRIC_KEYWORDS = (
    "throughput",
    "per_second",
    "seconds",
    "time",
    "duration",
    "latency",
    "ttft",
    "tpot",
    "itl",
)

TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})$")


@dataclass
class BenchmarkResult:
    path: Path
    gpu_name: str
    family: str
    benchmark_id: str
    metrics: Dict[str, str]
    units: Dict[str, str]
    timestamp: str


def parse_csv_metrics(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    metrics: Dict[str, str] = {}
    units: Dict[str, str] = {}
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if not row[0]:
                continue
            if row[0].strip() == "metric":
                continue
            if row[0].strip().startswith("#"):
                continue
            metric = row[0].strip()
            if len(row) < 2:
                continue
            value = row[1].strip()
            unit = row[2].strip() if len(row) > 2 else ""
            metrics[metric] = value
            units[metric] = unit
    return metrics, units


def infer_benchmark_id(path: Path, gpu_name: str) -> str:
    stem = path.stem
    stem = TIMESTAMP_RE.sub("", stem)
    stem = stem.replace(f"_{gpu_name}_", "_")
    stem = stem.replace(f"_{gpu_name}", "")
    stem = stem.replace(f"{gpu_name}_", "")
    stem = re.sub(r"__+", "_", stem)
    return stem.strip("_")


def infer_family(benchmark_id: str) -> str:
    for prefix in KNOWN_FAMILIES:
        if benchmark_id == prefix or benchmark_id.startswith(prefix + "_"):
            return prefix
    return benchmark_id.split("_")[0]


def parse_timestamp(metrics: Dict[str, str], path: Path) -> str:
    ts = metrics.get("timestamp", "")
    if ts:
        return ts
    match = TIMESTAMP_RE.search(path.stem)
    return match.group(1) if match else ""


def load_results(results_dir: Path) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    for path in results_dir.rglob("*.csv"):
        gpu_name = path.parent.name
        metrics, units = parse_csv_metrics(path)
        benchmark_id = infer_benchmark_id(path, gpu_name)
        family = infer_family(benchmark_id)
        timestamp = parse_timestamp(metrics, path)
        results.append(
            BenchmarkResult(
                path=path,
                gpu_name=gpu_name,
                family=family,
                benchmark_id=benchmark_id,
                metrics=metrics,
                units=units,
                timestamp=timestamp,
            )
        )
    return results


def build_config_summary(metrics: Dict[str, str], family: str) -> List[Tuple[str, str]]:
    fields = CONFIG_FIELDS.get(family, [])
    summary: List[Tuple[str, str]] = []
    for field in fields:
        value = metrics.get(field)
        if value:
            summary.append((field, value))
    return summary


def select_metrics(
    metrics: Dict[str, str],
    family: str,
    override: Optional[List[str]] = None,
) -> List[str]:
    if override:
        return [m for m in override if m in metrics]
    selected = COMPARE_METRICS.get(family)
    if selected:
        return [m for m in selected if m in metrics]
    candidates: List[str] = []
    for key in metrics:
        lower = key.lower()
        if any(token in lower for token in FALLBACK_METRIC_KEYWORDS):
            candidates.append(key)
    return sorted(set(candidates))


def latest_per_gpu(results: Iterable[BenchmarkResult]) -> Dict[Tuple[str, str], BenchmarkResult]:
    latest: Dict[Tuple[str, str], BenchmarkResult] = {}
    for result in results:
        key = (result.benchmark_id, result.gpu_name)
        if key not in latest:
            latest[key] = result
            continue
        if result.timestamp and result.timestamp > latest[key].timestamp:
            latest[key] = result
    return latest


def format_metric_value(value: Optional[str]) -> str:
    if value is None or value == "":
        return "-"
    return value


def natural_sort_key(text: str) -> List[object]:
    parts = re.split(r"(\d+)", text)
    key: List[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key
