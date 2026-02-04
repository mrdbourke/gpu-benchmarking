#!/usr/bin/env python3
"""Compare GPU benchmark results across similar benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

try:
    from benchmark_analysis.benchmark_utils import (
        COMPARE_METRICS,
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
        COMPARE_METRICS,
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
        description="Compare GPU models across similar benchmark runs.")
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
        "--gpu",
        action="append",
        help="Filter to a GPU name (repeatable, uses folder name).",
    )
    parser.add_argument(
        "--metrics",
        help="Comma-separated metrics to compare (overrides defaults).",
    )
    parser.add_argument(
        "--include-single",
        action="store_true",
        help="Include benchmarks that only have one GPU result.",
    )
    return parser.parse_args()


def group_results(results: List[BenchmarkResult]) -> Dict[str, Dict[str, BenchmarkResult]]:
    grouped: Dict[str, Dict[str, BenchmarkResult]] = {}
    latest = latest_per_gpu(results)
    for (_, _), result in latest.items():
        grouped.setdefault(result.benchmark_id, {})[result.gpu_name] = result
    return grouped


def metric_list_for_group(
    family: str,
    results: Dict[str, BenchmarkResult],
    override: List[str] | None,
) -> List[str]:
    if override:
        return override
    if family in COMPARE_METRICS:
        metrics = [
            key
            for key in COMPARE_METRICS[family]
            if any(key in result.metrics for result in results.values())
        ]
        if metrics:
            return metrics
    union: List[str] = []
    for result in results.values():
        union.extend(select_metrics(result.metrics, family))
    return sorted(set(union), key=natural_sort_key)


def render_markdown(grouped: Dict[str, Dict[str, BenchmarkResult]], override: List[str] | None, include_single: bool) -> str:
    lines: List[str] = []
    for benchmark_id in sorted(grouped, key=natural_sort_key):
        results = grouped[benchmark_id]
        if not include_single and len(results) < 2:
            continue
        sample = next(iter(results.values()))
        family = sample.family
        config = build_config_summary(sample.metrics, family)
        metrics = metric_list_for_group(family, results, override)

        lines.append(f"## {benchmark_id}")
        lines.append(f"Family: {family}")
        if config:
            config_text = ", ".join(f"{key}={value}" for key, value in config)
            lines.append(f"Config: {config_text}")

        gpu_names = sorted(results.keys(), key=natural_sort_key)
        header = "| Metric | Unit | " + " | ".join(gpu_names) + " |"
        separator = "| --- | --- | " + " | ".join(["---"] * len(gpu_names)) + " |"
        lines.append(header)
        lines.append(separator)
        for metric in metrics:
            unit = ""
            for result in results.values():
                unit = result.units.get(metric, "")
                if unit:
                    break
            row_values = []
            for gpu in gpu_names:
                value = results[gpu].metrics.get(metric)
                row_values.append(format_metric_value(value))
            row = "| " + " | ".join([metric, unit] + row_values) + " |"
            lines.append(row)
        lines.append("")
    if not lines:
        return "No matching benchmark comparisons found."
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results = load_results(results_dir)
    if args.family:
        families = set(args.family)
        results = [result for result in results if result.family in families]
    if args.gpu:
        gpus = set(args.gpu)
        results = [result for result in results if result.gpu_name in gpus]
    override = args.metrics.split(",") if args.metrics else None
    grouped = group_results(results)
    output = render_markdown(grouped, override, args.include_single)
    print(output, end="")


if __name__ == "__main__":
    main()
