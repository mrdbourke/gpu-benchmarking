"""
vLLM Token Input Throughput Animated Comparison

This Manim script generates an animated bar chart comparing token input (prompt processing)
per second across different models for NVIDIA GB10 and RTX 4090 GPUs.

Run from: /home/claude/ (or wherever this file is saved)
Command: manim -pql vllm_input_manim.py VLLMInputThroughput
    -p: preview (open video after rendering)
    -ql: quality low (faster rendering, 480p)
    -qm: quality medium (720p)
    -qh: quality high (1080p)
    -qk: quality 4k

Custom flags (use via python wrapper):
    --fadeout       Enable fade out at end (default: disabled)

Examples:
    # No fadeout (default) - direct manim
    manim -pql vllm_input_manim.py VLLMInputThroughput
    
    # No fadeout (default) - via python wrapper
    python vllm_input_manim.py -p -ql
    
    # With fadeout - via python wrapper
    python vllm_input_manim.py -p -ql --fadeout
    
Output: media/videos/vllm_input_manim/480p15/VLLMInputThroughput.mp4
"""

from manim import *
import os

# Set global default font for better kerning
Text.set_default(font="Inter")

# Read config from environment variables (set by __main__ wrapper)
CUSTOM_CONFIG = {
    "fadeout": os.environ.get("MANIM_FADEOUT", "0") == "1",
}

class CanvasLayout:
    """
    Helper class to calculate safe layout bounds for Manim scenes.

    Manim's default frame (16:9 aspect ratio):
    - Width: ~14.22 units (frame_width)
    - Height: 8 units (frame_height)
    - Center: (0, 0)
    - X range: approximately -7.11 to +7.11
    - Y range: -4 to +4
    """

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

    def clamp_x_to_safe(self, content_left_x, total_content_width):
        min_left = self.safe_left
        max_left = self.safe_right - total_content_width
        if max_left < min_left:
            return min_left
        return max(min_left, min(content_left_x, max_left))


def ease_out_cubic(t):
    return 1 - pow(1 - t, 3)


class VLLMInputThroughput(Scene):
    def construct(self):
        # Data
        models = [
            "gpt-oss-20b",
            "gpt-oss-120b",
            "Nemotron-3-Nano-30B",
            "Qwen3-VL-8B (0 img)",
            "Qwen3-VL-8B (1 img)",
            "Qwen3-VL-30B-A3B (0 img)",
            "Qwen3-VL-30B-A3B (1 img)",
            "Qwen3-VL-32B (0 img)",
            "Qwen3-VL-32B (1 img)",
        ]
        
        # GB10 input throughput (tok/s) = total_token_throughput - output_token_throughput
        gb10_input = [
            1538,   # gpt-oss-20b: 1768.38 - 230.00
            628,    # gpt-oss-120b: 744.40 - 116.00
            413,    # Nemotron-3-Nano-30B: 643.34 - 230.00
            611,    # Qwen3-VL-8B (0 img): 830.73 - 220.00
            452,    # Qwen3-VL-8B (1 img): 642.77 - 191.00
            370,    # Qwen3-VL-30B-A3B (0 img): 519.84 - 150.00
            316,    # Qwen3-VL-30B-A3B (1 img): 446.02 - 130.00
            166,    # Qwen3-VL-32B (0 img): 231.20 - 65.00
            134,    # Qwen3-VL-32B (1 img): 196.18 - 62.00
        ]
        
        # RTX 4090 input throughput (tok/s), 0 for NA values
        rtx4090_input = [
            4849,   # gpt-oss-20b: 5432.70 - 584.00
            0,      # gpt-oss-120b: NA
            0,      # Nemotron-3-Nano-30B: NA
            1576,   # Qwen3-VL-8B (0 img): 2155.83 - 580.00
            752,    # Qwen3-VL-8B (1 img): 1144.26 - 392.00
            0,      # Qwen3-VL-30B-A3B (0 img): NA
            0,      # Qwen3-VL-30B-A3B (1 img): NA
            0,      # Qwen3-VL-32B (0 img): NA
            0,      # Qwen3-VL-32B (1 img): NA
        ]
        
        # Colors (matching reference style)
        rtx_color = "#ff7f0e"  # Orange
        gb10_color = "#1f77b4"  # Blue
        
        # Setup canvas layout
        canvas = CanvasLayout(config, margin=0.5)
        
        # === Title Section ===
        title = Text("vLLM Inference: Token Input Throughput", font_size=42, weight=BOLD)
        if title.width > canvas.safe_width:
            title.scale(canvas.safe_width / title.width * 0.95)
        title.move_to([0, canvas.safe_top - title.height / 2, 0])
        
        subtitle = Text("Input Tokens per Second (tok/s) · ↑ Higher is better", font_size=24, color=GRAY)
        if subtitle.width > canvas.safe_width:
            subtitle.scale(canvas.safe_width / subtitle.width * 0.95)
        subtitle.next_to(title, DOWN, buff=0.2)
        
        # Animate title entrance
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.6)
        self.play(FadeIn(subtitle, shift=DOWN * 0.2), run_time=0.4)
        self.wait(0.3)
        
        # === Legend ===
        legend_rtx = VGroup(
            Rectangle(width=0.4, height=0.25, fill_color=rtx_color, fill_opacity=0.9, stroke_width=1),
            Text("RTX 4090", font_size=20)
        ).arrange(RIGHT, buff=0.15)
        
        legend_gb10 = VGroup(
            Rectangle(width=0.4, height=0.25, fill_color=gb10_color, fill_opacity=0.9, stroke_width=1),
            Text("GB10", font_size=20)
        ).arrange(RIGHT, buff=0.15)
        
        legend = VGroup(legend_rtx, legend_gb10).arrange(RIGHT, buff=0.6)
        legend.next_to(subtitle, DOWN, buff=0.3)
        
        self.play(FadeIn(legend), run_time=0.4)
        self.wait(0.2)
        
        # === Chart Parameters ===
        num_models = len(models)
        bar_height = 0.35
        bar_gap = 0.08  # Gap between RTX and GB10 bars for same model
        model_gap = 0.5  # Gap between model groups
        max_value = max(max(gb10_input), max(rtx4090_input))
        
        # Calculate vertical space needed
        total_chart_height = num_models * (2 * bar_height + bar_gap) + (num_models - 1) * model_gap
        
        # Chart positioning
        chart_top = legend.get_bottom()[1] - 0.4
        chart_bottom = canvas.safe_bottom + 0.3
        available_height = chart_top - chart_bottom
        
        # Scale bar heights if needed
        if total_chart_height > available_height:
            scale_factor = available_height / total_chart_height * 0.9
            bar_height *= scale_factor
            model_gap *= scale_factor
        
        # Recalculate total height
        total_chart_height = num_models * (2 * bar_height + bar_gap) + (num_models - 1) * model_gap
        
        # Starting Y position (top of chart area, centered)
        start_y = chart_top - (available_height - total_chart_height) / 2
        
        # X positions
        label_right_x = canvas.safe_left + 3.5  # Right edge of labels
        bar_start_x = label_right_x + 0.3
        max_bar_width = canvas.safe_right - bar_start_x - 1.2  # Leave room for values
        
        # === Build Chart ===
        model_labels = VGroup()
        rtx_bars = VGroup()
        gb10_bars = VGroup()
        rtx_values = VGroup()
        gb10_values = VGroup()
        
        for i, model in enumerate(models):
            # Y position for this model group
            group_top_y = start_y - i * (2 * bar_height + bar_gap + model_gap)
            rtx_y = group_top_y - bar_height / 2
            gb10_y = rtx_y - bar_height - bar_gap
            
            # Model label (right-aligned)
            label = Text(model, font_size=18)
            label_center_y = (rtx_y + gb10_y) / 2
            label.move_to([label_right_x - label.width / 2, label_center_y, 0])
            model_labels.add(label)
            
            # RTX 4090 bar
            rtx_val = rtx4090_input[i]
            if rtx_val > 0:
                rtx_bar_width = (rtx_val / max_value) * max_bar_width
            else:
                rtx_bar_width = 0.01  # Minimal width for NA
            
            rtx_bar = Rectangle(
                width=0.01,  # Start small for animation
                height=bar_height,
                fill_color=rtx_color,
                fill_opacity=0.9,
                stroke_color=WHITE,
                stroke_width=1,
            )
            rtx_bar.move_to([bar_start_x + 0.005, rtx_y, 0])
            rtx_bar.align_to([bar_start_x, 0, 0], LEFT)
            rtx_bar.target_width = rtx_bar_width
            rtx_bar.value = rtx_val
            rtx_bars.add(rtx_bar)
            
            # RTX value text
            rtx_text = Text(f"{rtx_val}" if rtx_val > 0 else "NA", font_size=14)
            rtx_text.set_opacity(0)
            rtx_values.add(rtx_text)
            
            # GB10 bar
            gb10_val = gb10_input[i]
            gb10_bar_width = (gb10_val / max_value) * max_bar_width
            
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
            gb10_bar.target_width = gb10_bar_width
            gb10_bar.value = gb10_val
            gb10_bars.add(gb10_bar)
            
            # GB10 value text
            gb10_text = Text(f"{gb10_val}", font_size=14)
            gb10_text.set_opacity(0)
            gb10_values.add(gb10_text)
        
        # === Animate Labels ===
        self.play(
            LaggedStart(
                *[FadeIn(label, shift=RIGHT * 0.3) for label in model_labels],
                lag_ratio=0.08
            ),
            run_time=1.2,
        )
        
        # === Animate Bars Growing ===
        self.add(*rtx_bars, *gb10_bars)
        
        bar_animations = []
        for bar in list(rtx_bars) + list(gb10_bars):
            target = bar.copy()
            target.stretch_to_fit_width(bar.target_width)
            target.align_to(bar, LEFT)
            bar_animations.append(Transform(bar, target, rate_func=ease_out_cubic))
        
        self.play(LaggedStart(*bar_animations, lag_ratio=0.05), run_time=2.0)
        
        # === Position and Animate Values ===
        for bar, value_text in zip(rtx_bars, rtx_values):
            value_text.next_to(bar, RIGHT, buff=0.15)
            if value_text.get_right()[0] > canvas.safe_right:
                value_text.move_to(bar.get_center())
                value_text.set_color(WHITE)
        
        for bar, value_text in zip(gb10_bars, gb10_values):
            value_text.next_to(bar, RIGHT, buff=0.15)
            if value_text.get_right()[0] > canvas.safe_right:
                value_text.move_to(bar.get_center())
                value_text.set_color(WHITE)
        
        self.play(
            LaggedStart(
                *[v.animate.set_opacity(1) for v in list(rtx_values) + list(gb10_values)],
                lag_ratio=0.03
            ),
            run_time=0.8,
        )
        
        # === Footer Notes ===
        footer_na = Text(
            "NA = Model did not fit on GPU memory.",
            font_size=14,
            color=GRAY
        )
        footer_quant = Text(
            "All models use FP8 quantization except for gpt-oss models which use MXFP4.",
            font_size=14,
            color=GRAY
        )
        
        # Stack footer notes
        footer = VGroup(footer_na, footer_quant).arrange(DOWN, buff=0.15)
        footer.move_to([0, canvas.safe_bottom, 0])
        
        self.play(FadeIn(footer), run_time=0.4)
        
        # Hold final frame
        self.wait(3)
        
        # === Fade Out (optional) ===
        if CUSTOM_CONFIG.get("fadeout", False):
            all_elements = VGroup(
                title, subtitle, legend, model_labels,
                rtx_bars, gb10_bars, rtx_values, gb10_values, footer
            )
            self.play(FadeOut(all_elements, shift=DOWN * 0.5), run_time=0.8)


if __name__ == "__main__":
    import subprocess
    import argparse
    
    # Parse custom arguments
    parser = argparse.ArgumentParser(description="vLLM Input Throughput Animation")
    parser.add_argument("--fadeout", action="store_true", default=False,
                        help="Enable fade out at end of animation (default: disabled)")
    parser.add_argument("-ql", action="store_true", help="Low quality (480p)")
    parser.add_argument("-qm", action="store_true", help="Medium quality (720p)")
    parser.add_argument("-qh", action="store_true", help="High quality (1080p)")
    parser.add_argument("-qk", action="store_true", help="4K quality")
    parser.add_argument("-p", "--preview", action="store_true", help="Preview after render")
    
    args = parser.parse_args()
    
    # Set environment variable for subprocess
    env = os.environ.copy()
    env["MANIM_FADEOUT"] = "1" if args.fadeout else "0"
    
    # Build manim command
    quality = "-ql"  # default
    if args.qm:
        quality = "-qm"
    elif args.qh:
        quality = "-qh"
    elif args.qk:
        quality = "-qk"
    
    cmd = ["manim", quality]
    if args.preview:
        cmd.append("-p")
    cmd.extend([__file__, "VLLMInputThroughput"])
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Fadeout: {args.fadeout}")
    subprocess.run(cmd, env=env)