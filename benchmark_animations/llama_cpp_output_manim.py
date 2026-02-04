"""
llama.cpp Generation Throughput Animated Comparison

This Manim script generates an animated bar chart comparing generation token throughput
(true_gen_throughput) across models for NVIDIA GB10 and RTX 4090 GPUs.

Command: manim -pql llama_cpp_output_manim.py LlamaCppOutputThroughput
Custom flags:
    --fadeout       Enable fade out at end (default: disabled)
"""

from manim import *
import os

# Set global default font for better kerning
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


class LlamaCppOutputThroughput(Scene):
    def construct(self):
        models = [
            "gpt-oss-20b (text)",
            "Nemotron-3-Nano-30B (text)",
            "Qwen3-VL-8B (multimodal)",
            "Qwen3-VL-8B (text)",
        ]

        gb10_values = [88.04, 47.56, 33.07, 35.48]
        rtx_values = [242.54, None, 122.83, 132.22]

        rtx_color = "#ff7f0e"
        gb10_color = "#1f77b4"

        canvas = CanvasLayout(config, margin=0.5)

        title = Text("llama.cpp Inference: Generation Throughput", font_size=42, weight=BOLD)
        if title.width > canvas.safe_width:
            title.scale(canvas.safe_width / title.width * 0.95)
        title.move_to([0, canvas.safe_top - title.height / 2, 0])

        subtitle = Text("Generated Tokens per Second (tok/s) · ↑ Higher is better", font_size=24, color=GRAY)
        if subtitle.width > canvas.safe_width:
            subtitle.scale(canvas.safe_width / subtitle.width * 0.95)
        subtitle.next_to(title, DOWN, buff=0.2)

        metadata = VGroup(
            Text("Metric: true_gen_throughput (aggregate generated tokens / total gen time)", font_size=18, color=GRAY),
            Text("100 prompts · output range 50-1000 tokens", font_size=18, color=GRAY),
        ).arrange(DOWN, buff=0.12)
        metadata.next_to(subtitle, DOWN, buff=0.25)

        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.6)
        self.play(FadeIn(subtitle, shift=DOWN * 0.2), run_time=0.4)
        self.play(FadeIn(metadata, shift=DOWN * 0.2), run_time=0.4)
        self.wait(0.3)

        legend_rtx = VGroup(
            Rectangle(width=0.4, height=0.25, fill_color=rtx_color, fill_opacity=0.9, stroke_width=1),
            Text("RTX 4090", font_size=20),
        ).arrange(RIGHT, buff=0.15)

        legend_gb10 = VGroup(
            Rectangle(width=0.4, height=0.25, fill_color=gb10_color, fill_opacity=0.9, stroke_width=1),
            Text("GB10", font_size=20),
        ).arrange(RIGHT, buff=0.15)

        legend = VGroup(legend_rtx, legend_gb10).arrange(RIGHT, buff=0.6)
        legend.next_to(metadata, DOWN, buff=0.25)
        self.play(FadeIn(legend), run_time=0.4)
        self.wait(0.2)

        num_models = len(models)
        bar_height = 0.35
        bar_gap = 0.08
        model_gap = 0.5
        max_value = max([v for v in gb10_values + [v or 0 for v in rtx_values]])

        total_chart_height = num_models * (2 * bar_height + bar_gap) + (num_models - 1) * model_gap
        chart_top = legend.get_bottom()[1] - 0.35
        chart_bottom = canvas.safe_bottom + 0.3
        available_height = chart_top - chart_bottom
        if total_chart_height > available_height:
            scale_factor = available_height / total_chart_height * 0.9
            bar_height *= scale_factor
            model_gap *= scale_factor

        total_chart_height = num_models * (2 * bar_height + bar_gap) + (num_models - 1) * model_gap
        start_y = chart_top - (available_height - total_chart_height) / 2

        label_right_x = canvas.safe_left + 3.5
        bar_start_x = label_right_x + 0.3
        max_bar_width = canvas.safe_right - bar_start_x - 1.2

        model_labels = VGroup()
        rtx_bars = VGroup()
        gb10_bars = VGroup()
        rtx_values_text = VGroup()
        gb10_values_text = VGroup()

        for i, model in enumerate(models):
            group_top_y = start_y - i * (2 * bar_height + bar_gap + model_gap)
            rtx_y = group_top_y - bar_height / 2
            gb10_y = rtx_y - bar_height - bar_gap

            label = Text(model, font_size=18)
            label_center_y = (rtx_y + gb10_y) / 2
            label.move_to([label_right_x - label.width / 2, label_center_y, 0])
            model_labels.add(label)

            rtx_val = rtx_values[i]
            rtx_width = (rtx_val / max_value) * max_bar_width if rtx_val else 0.01
            rtx_bar = Rectangle(
                width=0.01,
                height=bar_height,
                fill_color=rtx_color,
                fill_opacity=0.9,
                stroke_color=WHITE,
                stroke_width=1,
            )
            rtx_bar.move_to([bar_start_x + 0.005, rtx_y, 0])
            rtx_bar.align_to([bar_start_x, 0, 0], LEFT)
            rtx_bar.target_width = rtx_width
            rtx_bar.value = rtx_val
            rtx_bars.add(rtx_bar)

            rtx_text = Text(format_value(rtx_val), font_size=14)
            rtx_text.set_opacity(0)
            rtx_values_text.add(rtx_text)

            gb10_val = gb10_values[i]
            gb10_width = (gb10_val / max_value) * max_bar_width
            gb10_bar = Rectangle(
                width=0.01,
                height=bar_height,
                fill_color=gb10_color,
                fill_opacity=0.9,
                stroke_color=WHITE,
                stroke_width=1,
            )
            gb10_bar.move_to([bar_start_x + 0.005, gb10_y, 0])
            gb10_bar.align_to([bar_start_x, 0, 0], LEFT)
            gb10_bar.target_width = gb10_width
            gb10_bar.value = gb10_val
            gb10_bars.add(gb10_bar)

            gb10_text = Text(format_value(gb10_val), font_size=14)
            gb10_text.set_opacity(0)
            gb10_values_text.add(gb10_text)

        self.play(
            LaggedStart(
                *[FadeIn(label, shift=RIGHT * 0.3) for label in model_labels],
                lag_ratio=0.08,
            ),
            run_time=1.2,
        )

        self.add(*rtx_bars, *gb10_bars)
        bar_animations = []
        for bar in list(rtx_bars) + list(gb10_bars):
            target = bar.copy()
            target.stretch_to_fit_width(bar.target_width)
            target.align_to(bar, LEFT)
            bar_animations.append(Transform(bar, target, rate_func=ease_out_cubic))
        self.play(LaggedStart(*bar_animations, lag_ratio=0.05), run_time=2.0)

        for bar, value_text in zip(rtx_bars, rtx_values_text):
            value_text.next_to(bar, RIGHT, buff=0.15)
            if value_text.get_right()[0] > canvas.safe_right:
                value_text.move_to(bar.get_center())
                value_text.set_color(WHITE)

        for bar, value_text in zip(gb10_bars, gb10_values_text):
            value_text.next_to(bar, RIGHT, buff=0.15)
            if value_text.get_right()[0] > canvas.safe_right:
                value_text.move_to(bar.get_center())
                value_text.set_color(WHITE)

        self.play(
            LaggedStart(
                *[v.animate.set_opacity(1) for v in list(rtx_values_text) + list(gb10_values_text)],
                lag_ratio=0.03,
            ),
            run_time=0.8,
        )

        footer = Text("NA = Model did not fit on GPU memory.", font_size=14, color=GRAY)

        footer_quant = Text(
            "All models use GGUF format see repo for exact model codes.",
            font_size=14,
            color=GRAY
        )
        
        # Stack footer notes
        footer = VGroup(footer, footer_quant).arrange(DOWN, buff=0.15)
        footer.move_to([0, canvas.safe_bottom-0.1, 0])
        self.play(FadeIn(footer), run_time=0.4)

        self.wait(3)

        if CUSTOM_CONFIG.get("fadeout", False):
            all_elements = VGroup(
                title,
                subtitle,
                metadata,
                legend,
                model_labels,
                rtx_bars,
                gb10_bars,
                rtx_values_text,
                gb10_values_text,
                footer,
            )
            self.play(FadeOut(all_elements, shift=DOWN * 0.5), run_time=0.8)


if __name__ == "__main__":
    import subprocess
    import argparse

    parser = argparse.ArgumentParser(description="llama.cpp Generation Throughput Animation")
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
    cmd.extend([__file__, "LlamaCppOutputThroughput"])

    print(f"Running: {' '.join(cmd)}")
    print(f"Fadeout: {args.fadeout}")
    subprocess.run(cmd, env=env)
