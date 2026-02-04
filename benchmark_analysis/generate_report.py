#!/usr/bin/env python3
"""Generate README.md and benchmark_report.html from benchmark_results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from benchmark_analysis.benchmark_utils import (
        COMPARE_METRICS,
        BenchmarkResult,
        build_config_summary,
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
        latest_per_gpu,
        load_results,
        natural_sort_key,
        select_metrics,
    )

SECTION_BY_FAMILY = {
    "vLLM": "LLM/VLM Inference",
    "llama_server": "LLM/VLM Inference",
    "llm_finetune_bench": "LLM Training",
    "object_detection_bench": "Object Detection",
    "flux_bench": "Image Generation",
    "zimage_bench": "Image Generation",
}

LOWER_BETTER_TOKENS = (
    "time",
    "duration",
    "latency",
    "ttft",
    "tpot",
    "itl",
    "seconds_per_image",
    "runtime",
)

DEFAULT_RESULTS_DIR = (Path(__file__).resolve().parent.parent / "benchmark_results").resolve()
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark analysis reports.")
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing benchmark results.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write README.md and benchmark_report.html.",
    )
    return parser.parse_args()


def is_lower_better(metric: str) -> bool:
    lower = metric.lower()
    return any(token in lower for token in LOWER_BETTER_TOKENS)


def parse_number(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.upper() in {"NA", "N/A"}:
        return None
    try:
        return float(stripped.replace(",", ""))
    except ValueError:
        return None


def metric_list_for_group(family: str, results: Dict[str, BenchmarkResult]) -> List[str]:
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


def build_report_data(results: List[BenchmarkResult]) -> Dict[str, object]:
    latest = latest_per_gpu(results)

    by_benchmark: Dict[str, Dict[str, BenchmarkResult]] = {}
    for (_, _), result in latest.items():
        by_benchmark.setdefault(result.benchmark_id, {})[result.gpu_name] = result

    all_gpus = sorted({result.gpu_name for result in latest.values()}, key=natural_sort_key)

    sections: Dict[str, List[Dict[str, object]]] = {
        "LLM/VLM Inference": [],
        "LLM Training": [],
        "Image Generation": [],
        "Object Detection": [],
        "Other": [],
    }

    for benchmark_id in sorted(by_benchmark, key=natural_sort_key):
        results_map = by_benchmark[benchmark_id]
        sample = next(iter(results_map.values()))
        family = sample.family
        section = SECTION_BY_FAMILY.get(family, "Other")
        config = build_config_summary(sample.metrics, family)
        config_text = ", ".join(f"{key}={value}" for key, value in config) if config else ""

        metrics = []
        for metric in metric_list_for_group(family, results_map):
            unit = ""
            for result in results_map.values():
                unit = result.units.get(metric, "")
                if unit:
                    break
            values: Dict[str, str] = {}
            numbers: Dict[str, Optional[float]] = {}
            for gpu in all_gpus:
                value = results_map.get(gpu).metrics.get(metric) if results_map.get(gpu) else "NA"
                values[gpu] = value if value not in {None, ""} else "NA"
                numbers[gpu] = parse_number(values[gpu])
            metrics.append(
                {
                    "name": metric,
                    "unit": unit,
                    "values": values,
                    "numbers": numbers,
                    "lower_better": is_lower_better(metric),
                }
            )

        sections[section].append(
            {
                "id": benchmark_id,
                "family": family,
                "config": config_text,
                "metrics": metrics,
            }
        )

    return {
        "meta": {"generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        "gpus": all_gpus,
        "sections": sections,
    }


def build_markdown(data: Dict[str, object]) -> str:
    gpus: List[str] = data["gpus"]  # type: ignore[assignment]
    sections: Dict[str, List[Dict[str, object]]] = data["sections"]  # type: ignore[assignment]
    generated: str = data["meta"]["generated"]  # type: ignore[index]

    lines: List[str] = []
    lines.append("# GPU Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {generated}")
    lines.append("")
    lines.append(
        "Note: Missing results are shown as NA and assumed to be unavailable because the model did not fit or could not be run on that GPU."
    )

    ordered_sections = [
        "LLM/VLM Inference",
        "LLM Training",
        "Image Generation",
        "Object Detection",
        "Other",
    ]

    for section_name in ordered_sections:
        benchmarks = sections.get(section_name, [])
        if not benchmarks:
            continue
        lines.append("")
        lines.append(f"# {section_name}")
        lines.append("")
        for benchmark in benchmarks:
            lines.append(f"## {benchmark['id']}")
            lines.append(f"Family: {benchmark['family']}")
            config_text = benchmark.get("config")
            if config_text:
                lines.append(f"Config: {config_text}")

            header = "| Metric | Unit | " + " | ".join(gpus) + " |"
            separator = "| --- | --- | " + " | ".join(["---"] * len(gpus)) + " |"
            lines.append(header)
            lines.append(separator)

            for metric in benchmark["metrics"]:  # type: ignore[index]
                unit = metric["unit"]
                values: Dict[str, str] = metric["values"]
                numbers: Dict[str, Optional[float]] = metric["numbers"]
                lower_better = bool(metric["lower_better"])

                best_value: Optional[float] = None
                for value in numbers.values():
                    if value is None:
                        continue
                    if best_value is None:
                        best_value = value
                    elif lower_better and value < best_value:
                        best_value = value
                    elif not lower_better and value > best_value:
                        best_value = value

                row_values: List[str] = []
                for gpu in gpus:
                    value = values.get(gpu, "NA")
                    numeric = numbers.get(gpu)
                    if value in {"NA", None}:
                        row_values.append("NA")
                        continue
                    formatted = value
                    if best_value is not None and numeric is not None and numeric == best_value:
                        formatted = f"**{formatted}**"
                    row_values.append(formatted)

                lines.append("| " + " | ".join([metric["name"], unit] + row_values) + " |")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_html(data: Dict[str, object]) -> str:
    generated: str = data["meta"]["generated"]  # type: ignore[index]
    payload = {
        "meta": data["meta"],
        "gpus": data["gpus"],
        "sections": [],
    }

    for section_name, benchmarks in data["sections"].items():  # type: ignore[union-attr]
        if not benchmarks:
            continue
        payload["sections"].append(
            {
                "name": section_name,
                "benchmarks": benchmarks,
            }
        )

    html_template = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>GPU Benchmark Report</title>
<style>
  :root {
    --bg: #f7f7f7;
    --panel: #ffffff;
    --text: #111111;
    --muted: #5a5a5a;
    --grid: #e4e4e4;
    --accent: #1f77b4;
    --shadow: rgba(0,0,0,0.08);
    --border: #d8d8d8;
  }
  body {
    margin: 0;
    font-family: "Georgia", "Times New Roman", serif;
    background: var(--bg);
    color: var(--text);
  }
  header {
    padding: 28px 32px 10px 32px;
    background: linear-gradient(120deg, #fcfcfc, #f1f1f1);
    border-bottom: 1px solid var(--border);
  }
  header h1 {
    margin: 0 0 8px 0;
    font-size: 28px;
    letter-spacing: 0.5px;
  }
  header p {
    margin: 4px 0;
    color: var(--muted);
    font-size: 14px;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 12px;
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
  }
  .legend-swatch {
    width: 16px;
    height: 10px;
    border-radius: 3px;
  }
  main {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px 32px 60px 32px;
  }
  section {
    margin: 32px 0;
  }
  section h2 {
    margin: 0 0 18px 0;
    font-size: 22px;
    border-bottom: 2px solid var(--border);
    padding-bottom: 6px;
  }
  .benchmark-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px 16px 20px;
    box-shadow: 0 10px 30px var(--shadow);
    margin-bottom: 22px;
  }
  .benchmark-card h3 {
    margin: 0 0 6px 0;
    font-size: 18px;
  }
  .benchmark-meta {
    margin: 0;
    font-size: 13px;
    color: var(--muted);
  }
  .benchmark-note {
    margin: 8px 0 0 0;
    font-size: 12px;
    color: var(--muted);
  }
  .benchmark-note span {
    font-weight: 700;
    color: var(--text);
  }
  .metric-row {
    display: grid;
    grid-template-columns: 220px 1fr;
    gap: 12px;
    margin-top: 14px;
  }
  .metric-label {
    font-size: 13px;
    color: var(--text);
  }
  .metric-label span {
    display: block;
    color: var(--muted);
    font-size: 12px;
  }
  .chart {
    position: relative;
    padding: 8px 12px 10px 12px;
    background-image: repeating-linear-gradient(
      to right,
      transparent,
      transparent 54px,
      var(--grid) 54px,
      var(--grid) 55px
    );
    border-radius: 8px;
    border: 1px solid #e8e8e8;
  }
  .bars {
    display: grid;
    gap: 8px;
  }
  .bar {
    position: relative;
    height: 24px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    padding-left: 8px;
    color: #fff;
    font-size: 12px;
    letter-spacing: 0.2px;
    min-width: 30px;
  }
  .bar.best {
    outline: 2px solid rgba(0,0,0,0.35);
  }
  .bar-label {
    position: absolute;
    right: 6px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 11px;
    color: rgba(0,0,0,0.7);
  }
  .missing-note {
    margin-top: 6px;
    font-size: 11px;
    color: var(--muted);
  }
  @media (max-width: 900px) {
    .metric-row {
      grid-template-columns: 1fr;
    }
  }
</style>
</head>
<body>
<header>
  <h1>GPU Benchmark Report</h1>
  <p>Generated: __GENERATED__</p>
  <p>Missing results are shown as NA and assumed to be unavailable because the model did not fit or could not be run on that GPU.</p>
  <div class="legend" id="legend"></div>
</header>
<main id="content"></main>
<script>
const data = __DATA__;

const colors = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
];

function createLegend(gpus) {
  const legend = document.getElementById('legend');
  gpus.forEach((gpu, idx) => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    const swatch = document.createElement('span');
    swatch.className = 'legend-swatch';
    swatch.style.background = colors[idx % colors.length];
    const label = document.createElement('span');
    label.textContent = gpu;
    item.appendChild(swatch);
    item.appendChild(label);
    legend.appendChild(item);
  });
}

function render() {
  const container = document.getElementById('content');
  createLegend(data.gpus);

  data.sections.forEach(section => {
    if (!section.benchmarks || section.benchmarks.length === 0) {
      return;
    }
    const sectionEl = document.createElement('section');
    const heading = document.createElement('h2');
    heading.textContent = section.name;
    sectionEl.appendChild(heading);

    section.benchmarks.forEach(benchmark => {
      const card = document.createElement('div');
      card.className = 'benchmark-card';
      const title = document.createElement('h3');
      title.textContent = benchmark.id;
      card.appendChild(title);

      if (benchmark.family) {
        const family = document.createElement('p');
        family.className = 'benchmark-meta';
        family.textContent = `Family: ${benchmark.family}`;
        card.appendChild(family);
      }

      if (benchmark.config) {
        const config = document.createElement('p');
        config.className = 'benchmark-meta';
        config.textContent = `Config: ${benchmark.config}`;
        card.appendChild(config);
      }

      const note = document.createElement('p');
      note.className = 'benchmark-note';
      note.innerHTML = 'Higher is better (<span>&uarr;</span>) for throughput/rate metrics. Lower is better (<span>&darr;</span>) for time/latency metrics.';
      card.appendChild(note);

      benchmark.metrics.forEach(metric => {
        const row = document.createElement('div');
        row.className = 'metric-row';

        const label = document.createElement('div');
        label.className = 'metric-label';
        label.textContent = metric.name;
        const unit = document.createElement('span');
        unit.textContent = metric.unit ? `Unit: ${metric.unit}` : 'Unit: n/a';
        label.appendChild(unit);
        row.appendChild(label);

        const chart = document.createElement('div');
        chart.className = 'chart';
        const bars = document.createElement('div');
        bars.className = 'bars';

        const values = data.gpus.map(gpu => metric.numbers[gpu]);
        const maxValue = Math.max(...values.filter(v => v !== null), 0);
        const lowerBetter = metric.lower_better;
        let bestValue = null;
        values.forEach(value => {
          if (value === null) return;
          if (bestValue === null) {
            bestValue = value;
          } else if (lowerBetter && value < bestValue) {
            bestValue = value;
          } else if (!lowerBetter && value > bestValue) {
            bestValue = value;
          }
        });

        const missing = [];

        data.gpus.forEach((gpu, idx) => {
          const value = metric.values[gpu] ?? 'NA';
          const numeric = metric.numbers[gpu];
          if (numeric === null) {
            missing.push(gpu);
            return;
          }

          const bar = document.createElement('div');
          bar.className = 'bar';
          const width = maxValue === 0 ? 0 : Math.max((numeric / maxValue) * 100, 4);
          bar.style.width = `${width}%`;
          bar.style.background = colors[idx % colors.length];
          bar.title = `${gpu}: ${value}${metric.unit ? ' ' + metric.unit : ''}`;

          bar.textContent = `${gpu}`;
          const valueLabel = document.createElement('span');
          valueLabel.className = 'bar-label';
          valueLabel.textContent = value;
          bar.appendChild(valueLabel);

          if (bestValue !== null && numeric === bestValue) {
            bar.classList.add('best');
          }

          bars.appendChild(bar);
        });

        chart.appendChild(bars);
        if (missing.length) {
          const missingNote = document.createElement('div');
          missingNote.className = 'missing-note';
          missingNote.textContent = `Missing: ${missing.join(', ')}`;
          chart.appendChild(missingNote);
        }
        row.appendChild(chart);
        card.appendChild(row);
      });

      sectionEl.appendChild(card);
    });

    container.appendChild(sectionEl);
  });
}

render();
</script>
</body>
</html>
"""

    return (
        html_template.replace("__DATA__", json.dumps(payload))
        .replace("__GENERATED__", generated)
        .rstrip()
        + "\n"
    )


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)
    data = build_report_data(results)

    readme_path = output_dir / "README.md"
    html_path = output_dir / "benchmark_report.html"

    readme_path.write_text(build_markdown(data))
    html_path.write_text(build_html(data))

    print(f"Wrote {readme_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
