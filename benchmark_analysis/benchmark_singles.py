#!/usr/bin/env python3
"""Summarize benchmark results for a single GPU."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

try:
    from benchmark_analysis.benchmark_utils import (
        BenchmarkResult,
        build_config_summary,
        format_metric_value,
        latest_per_gpu,
        load_results,
        natural_sort_key,
        select_metrics,
    )
except ImportError:  # Allow running from within benchmark_analysis
    from benchmark_utils import (  # type: ignore
        BenchmarkResult,
        build_config_summary,
        format_metric_value,
        latest_per_gpu,
        load_results,
        natural_sort_key,
        select_metrics,
    )

DEFAULT_RESULTS_DIR = (Path(__file__).resolve().parent.parent / "benchmark_results").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collate benchmark results for a single GPU.")
    parser.add_argument(
        "--gpu-name",
        required=True,
        help="GPU name (folder under benchmark_results).",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Root directory containing benchmark results.",
    )
    parser.add_argument(
        "--family",
        action="append",
        help="Filter to a benchmark family (repeatable).",
    )
    parser.add_argument(
        "--metrics",
        help="Comma-separated metrics to include (overrides defaults).",
    )
    return parser.parse_args()


def group_results(results: List[BenchmarkResult]) -> Dict[str, BenchmarkResult]:
    grouped: Dict[str, BenchmarkResult] = {}
    latest = latest_per_gpu(results)
    for (_, _), result in latest.items():
        grouped[result.benchmark_id] = result
    return grouped


def render_markdown(
    grouped: Dict[str, BenchmarkResult],
    gpu_name: str,
    override: List[str] | None,
) -> str:
    lines: List[str] = []
    lines.append(f"GPU: {gpu_name}")
    if not grouped:
        lines.append("No benchmark results found for this GPU.")
        return "\n".join(lines) + "\n"

    for benchmark_id in sorted(grouped, key=natural_sort_key):
        result = grouped[benchmark_id]
        config = build_config_summary(result.metrics, result.family)
        metrics = override or select_metrics(result.metrics, result.family)

        lines.append("")
        lines.append(f"## {benchmark_id}")
        lines.append(f"Family: {result.family}")
        if config:
            config_text = ", ".join(f"{key}={value}" for key, value in config)
            lines.append(f"Config: {config_text}")

        lines.append("| Metric | Value | Unit |")
        lines.append("| --- | --- | --- |")
        for metric in metrics:
            value = format_metric_value(result.metrics.get(metric))
            unit = result.units.get(metric, "")
            lines.append(f"| {metric} | {value} | {unit} |")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results = load_results(results_dir)
    results = [result for result in results if result.gpu_name == args.gpu_name]
    if args.family:
        families = set(args.family)
        results = [result for result in results if result.family in families]
    override = args.metrics.split(",") if args.metrics else None
    grouped = group_results(results)
    output = render_markdown(grouped, args.gpu_name, override)
    print(output, end="")


if __name__ == "__main__":
    main()
