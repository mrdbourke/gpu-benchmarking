"""
llama.cpp vs vLLM Prompt/Input Throughput Comparison

Compares prompt/input token throughput between llama.cpp and vLLM
for the same models and GPUs.

Command: manim -pql llama_cpp_vs_vllm_input_manim.py LlamaCppVsVLLMInputThroughput
Custom flags:
    --fadeout       Enable fade out at end (default: disabled)
"""

from manim import *
import os

Text.set_default(font="Inter")

CUSTOM_CONFIG = {
    "fadeout": os.environ.get("MANIM_FADEOUT", "0") == "1",
}


class CanvasLayout:
    def __init__(self, config, margin=0.5):
        self.frame_width = config.frame_width
        self.frame_height = config.frame_height
        self.margin = margin

        self.safe_left = -self.frame_width / 2 + self.margin
        self.safe_right = self.frame_width / 2 - self.margin
        self.safe_top = self.frame_height / 2 - self.margin
        self.safe_bottom = -self.frame_height / 2 + self.margin

        self.safe_width = self.safe_right - self.safe_left
        self.safe_height = self.safe_top - self.safe_bottom


def ease_out_cubic(t):
    return 1 - pow(1 - t, 3)


def format_value(value):
    if value is None:
        return "NA"
    formatted = f"{value:.2f}"
    if formatted.endswith(".00"):
        return formatted[:-3]
    if formatted.endswith("0"):
        return formatted[:-1]
    return formatted


class LlamaCppVsVLLMInputThroughput(Scene):
    def construct(self):
        models = [
            "gpt-oss-20b (text)",
            "Qwen3-VL-8B (text, 0 img)",
            "Qwen3-VL-8B (1 img)",
        ]

        series = [
            {
                "label": "GB10 · llama.cpp",
                "color": "#1f77b4",
                "values": [2703.36, 2478.36, 799.44],
            },
            {
                "label": "GB10 · vLLM",
                "color": "#6baed6",
                "values": [1538.38, 610.73, 451.77],
            },
            {
                "label": "RTX 4090 · llama.cpp",
                "color": "#ff7f0e",
                "values": [6035.78, 8414.23, 2459.23],
            },
            {
                "label": "RTX 4090 · vLLM",
                "color": "#ffb570",
                "values": [4848.70, 1575.83, 752.26],
            },
        ]

        canvas = CanvasLayout(config, margin=0.5)

        title = Text("llama.cpp vs vLLM: Prompt/Input Throughput", font_size=40, weight=BOLD)
        if title.width > canvas.safe_width:
            title.scale(canvas.safe_width / title.width * 0.95)
        title.move_to([0, canvas.safe_top - title.height / 2, 0])

        subtitle = Text("Tokens per Second (tok/s) · ↑ Higher is better", font_size=24, color=GRAY)
        if subtitle.width > canvas.safe_width:
            subtitle.scale(canvas.safe_width / subtitle.width * 0.95)
        subtitle.next_to(title, DOWN, buff=0.2)

        metadata = VGroup(
            Text("llama.cpp: true_prompt_throughput · vLLM: input_token_throughput", font_size=18, color=GRAY),
            Text("Same models, same GPUs; text-only vs multimodal noted in labels", font_size=18, color=GRAY),
        ).arrange(DOWN, buff=0.12)
        metadata.next_to(subtitle, DOWN, buff=0.25)

        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.6)
        self.play(FadeIn(subtitle, shift=DOWN * 0.2), run_time=0.4)
        self.play(FadeIn(metadata, shift=DOWN * 0.2), run_time=0.4)
        self.wait(0.3)

        legend_items = []
        for item in series:
            legend_items.append(
                VGroup(
                    Rectangle(width=0.4, height=0.25, fill_color=item["color"], fill_opacity=0.9, stroke_width=1),
                    Text(item["label"], font_size=18),
                ).arrange(RIGHT, buff=0.15)
            )
        legend = VGroup(*legend_items).arrange(RIGHT, buff=0.4)
        if legend.width > canvas.safe_width:
            legend.scale(canvas.safe_width / legend.width * 0.95)
        legend.next_to(metadata, DOWN, buff=0.25)
        self.play(FadeIn(legend), run_time=0.4)
        self.wait(0.2)

        num_models = len(models)
        series_count = len(series)
        bar_height = 0.26
        bar_gap = 0.08
        model_gap = 0.55

        max_value = max(
            value
            for item in series
            for value in item["values"]
            if value is not None
        )

        group_height = series_count * bar_height + (series_count - 1) * bar_gap
        total_chart_height = num_models * group_height + (num_models - 1) * model_gap

        chart_top = legend.get_bottom()[1] - 0.35
        chart_bottom = canvas.safe_bottom + 0.3
        available_height = chart_top - chart_bottom
        if total_chart_height > available_height:
            scale_factor = available_height / total_chart_height * 0.9
            bar_height *= scale_factor
            model_gap *= scale_factor
            bar_gap *= scale_factor
            group_height = series_count * bar_height + (series_count - 1) * bar_gap
            total_chart_height = num_models * group_height + (num_models - 1) * model_gap

        start_y = chart_top - (available_height - total_chart_height) / 2

        label_right_x = canvas.safe_left + 3.8
        bar_start_x = label_right_x + 0.35
        max_bar_width = canvas.safe_right - bar_start_x - 1.2

        model_labels = VGroup()
        series_bars = [VGroup() for _ in series]
        series_values_text = [VGroup() for _ in series]

        for i, model in enumerate(models):
            group_top_y = start_y - i * (group_height + model_gap)
            label_center_y = group_top_y - group_height / 2

            label = Text(model, font_size=18)
            label.move_to([label_right_x - label.width / 2, label_center_y, 0])
            model_labels.add(label)

            for s_index, item in enumerate(series):
                value = item["values"][i]
                width = (value / max_value) * max_bar_width if value is not None else 0.01
                bar_y = group_top_y - bar_height / 2 - s_index * (bar_height + bar_gap)
                bar = Rectangle(
                    width=0.01,
                    height=bar_height,
                    fill_color=item["color"],
                    fill_opacity=0.9,
                    stroke_color=WHITE,
                    stroke_width=1,
                )
                bar.move_to([bar_start_x + 0.005, bar_y, 0])
                bar.align_to([bar_start_x, 0, 0], LEFT)
                bar.target_width = width
                bar.value = value
                series_bars[s_index].add(bar)

                value_text = Text(format_value(value), font_size=14)
                value_text.set_opacity(0)
                series_values_text[s_index].add(value_text)

        self.play(
            LaggedStart(
                *[FadeIn(label, shift=RIGHT * 0.3) for label in model_labels],
                lag_ratio=0.08,
            ),
            run_time=1.2,
        )

        for group in series_bars:
            self.add(*group)

        bar_animations = []
        for group in series_bars:
            for bar in group:
                target = bar.copy()
                target.stretch_to_fit_width(bar.target_width)
                target.align_to(bar, LEFT)
                bar_animations.append(Transform(bar, target, rate_func=ease_out_cubic))
        self.play(LaggedStart(*bar_animations, lag_ratio=0.03), run_time=2.0)

        for group, value_group in zip(series_bars, series_values_text):
            for bar, value_text in zip(group, value_group):
                value_text.next_to(bar, RIGHT, buff=0.12)
                if value_text.get_right()[0] > canvas.safe_right:
                    value_text.move_to(bar.get_center())
                    value_text.set_color(WHITE)

        self.play(
            LaggedStart(
                *[v.animate.set_opacity(1) for group in series_values_text for v in group],
                lag_ratio=0.02,
            ),
            run_time=0.8,
        )

        footer = Text("All values in tok/s.", font_size=14, color=GRAY)
        footer.move_to([0, canvas.safe_bottom, 0])
        self.play(FadeIn(footer), run_time=0.4)

        self.wait(3)

        if CUSTOM_CONFIG.get("fadeout", False):
            all_elements = VGroup(
                title,
                subtitle,
                metadata,
                legend,
                model_labels,
                *series_bars,
                *series_values_text,
                footer,
            )
            self.play(FadeOut(all_elements, shift=DOWN * 0.5), run_time=0.8)


if __name__ == "__main__":
    import subprocess
    import argparse

    parser = argparse.ArgumentParser(description="llama.cpp vs vLLM Input Throughput Animation")
    parser.add_argument("--fadeout", action="store_true", default=False, help="Enable fade out")
    parser.add_argument("-ql", action="store_true", help="Low quality (480p)")
    parser.add_argument("-qm", action="store_true", help="Medium quality (720p)")
    parser.add_argument("-qh", action="store_true", help="High quality (1080p)")
    parser.add_argument("-qk", action="store_true", help="4K quality")
    parser.add_argument("-p", "--preview", action="store_true", help="Preview after render")

    args = parser.parse_args()

    env = os.environ.copy()
    env["MANIM_FADEOUT"] = "1" if args.fadeout else "0"

    quality = "-ql"
    if args.qm:
        quality = "-qm"
    elif args.qh:
        quality = "-qh"
    elif args.qk:
        quality = "-qk"

    cmd = ["manim", quality]
    if args.preview:
        cmd.append("-p")
    cmd.extend([__file__, "LlamaCppVsVLLMInputThroughput"])

    print(f"Running: {' '.join(cmd)}")
    print(f"Fadeout: {args.fadeout}")
    subprocess.run(cmd, env=env)