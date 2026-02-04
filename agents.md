Commands for agents go in here.

## Plots we want

There are four benchmarks (LLM/VLM inference, LLM training, image generation, object detection) and we want to visualize the results easily.

Plots we want:

* ✅ vLLM input (see example: benchmark_animations/vllm_input_manim.py)
* ✅ vLLM output (see example: benchmark_animations/vllm_ouput_manim.py)
* llama.cpp input
* llama.cpp output
* llama.cpp vs vLLM input on the same models and same GPU
* llama.cpp vs vLLM output on the same models and same GPU
* LLM fine-tuning comparison of different training times on different GPUs
* Image generation comparison of different models on different GPUs
* Object detection comparison of different training times on different GPUs

Metrics to reference: 

* Our metrics are located in `benchmark_analysis/README.md`, these are a collation of what's inside `benchmark_results/`.

## Plot overviews

### vLLM input: Comparison of models token input per second across GPUs for vLLM inference

Inference engine: vLLM
Metric to plot: token input per second
X axis: tokens per second
y axis: model name 
Key: RTX_4090 (orange), GB_10 (blue), matplotlib colours
Instructions:
    - For GPUs such as RTX_4090 which don't have results for certain models, leave this value as 0

### vLLM output: Comparison of models token generation per second across GPUs for vLLM inference

Inference engine: vLLM
Metric to plot: token generation per second
X axis: tokens per second
y axis: model name 
Key: RTX_4090 (orange), GB_10 (blue), matplotlib colours
Instructions:
    - For GPUs such as RTX_4090 which don't have results for certain models, leave this value as 0

### llama.cpp input: Comparison of models token input per second across GPUs for vLLM inference

Inference engine: llama.cpp
Metric to plot: token input per second
X axis: tokens per second
y axis: model name 
Key: RTX_4090 (orange), GB_10 (blue), matplotlib colours
Instructions:
    - For GPUs such as RTX_4090 which don't have results for certain models, leave this value as 0

### llama.cpp output: Comparison of models token input per second across GPUs for vLLM inference

Inference engine: llama.cpp
Metric to plot: token input per second
X axis: tokens per second
y axis: model name 
Key: RTX_4090 (orange), GB_10 (blue), matplotlib colours
Instructions:
    - For GPUs such as RTX_4090 which don't have results for certain models, leave this value as 0

### llama.cpp vs vLLM input on the same models on RTX 4090 

Inference engines: llama.cpp and vLLM
Output file: comparison of llama.cpp and vLLM inference engines on RTX 4090 on same models
X axis: tokens per second
y axis: model name

### llama.cpp vs vLLM input on the same models on DGX Spark

Inference engines: llama.cpp and vLLM
Output file: comparison of llama.cpp and vLLM inference engines on DGX Spark on same models
X axis: tokens per second
y axis: model name

### LLM fine-tuning comparison of different training times on different GPUs

Training library: Hugging Face Transformers
Output file: comparison of DGX Spark and RTX 4090 fine-tuning a small language model on 1000 samples for 3 epochs
X axis: total training time
y axis: GPU name

### Image generation comparison of different models on different GPUs

Inference library: Diffusers and Transformers
Output file: comparison of DGX Spark and RTX 4090 image generation on two different models
X axis: generation time for 100 images (lower is better)
y axis: model name
Key: RTX_4090 (orange), GB_10 (blue), matplotlib colours

### Object detection comparison of different training times on different GPUs

Training library: Hugging Face Transformers
Output file: comparison of DGX Spark and RTX 4090 fine-tuning an object detection model on 1000 images for 10 epochs
X axis: total training time
y axis: GPU name
Key: RTX_4090 (orange), GB_10 (blue), matplotlib colours