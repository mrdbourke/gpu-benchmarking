# Comapre GPUs on different tasks

The main goal of this repo is to benchmark the NVIDIA DGX Spark against the NVIDIA RTX 4090 on a series of ML/AI tasks.

Why these two GPUs? 

They are the two GPUs I own.

But the tests could easily be extended to other GPUs.

Most of the analysis in `benchmark_analysis/` was performed by GPT-Codex simply to compare the results in `benchmark_results/` (this holds the ground truth for a series of tests in `.csv` files). 

These benchmarks are mostly targeted at single user usage rather than multiple users at the same time. For example, a single developer/researcher working on their own machine.

## Summary

**One line:** RTX 4090 faster on training and inference by ~3-4x but cannot run larger models like the DGX Spark (24GB VRAM vs 128GB VRAM).

### Performance

The NVIDIA DGX Spark is a like a minivan.

Whereas the NVIDIA RTX 4090 is like Ferrari.

What I mean by this is that for any compute intensive task, the RTX 4090 is generally *faster* (on training and inference) than the DGX Spark on an order of 3-4x.

However, there are several benchmarks (e.g. running `gpt-oss-120b`) where the RTX 4090 simply can't because it doesn't have enough space (VRAM).

So the DGX Spark while generally slower than the RTX 4090 can run much larger models with a much larger context.

### Getting started and machine footprint

DGX Spark is one plug and play (it is a full system in one box). Go from unboxing to running models in ~30 mins. Especially with the [NVIDIA DGX Spark website tutorials](https://build.nvidia.com/spark).

RTX 4090 is a GPU card only and requires a custom built PC to house it (this took me and my friend ~3-4 hours) as well as software setup (~2-3 hours with the help of AIs).

The DGX Spark also has a small footprint, similar to the size of a small textbook.

The RTX 4090, depending on your PC build will have a much larger footprint.

TK image - compare the footprints of the two machines 

### Which one?

* You: Want quickest possible setup to go from 0 to running AI models on my desktop = DGX Spark.
* You: Want fastest GPU for fine-tuning/running smaller models and don't mind building a PC = RTX 4090.
* You: Want to run large models (e.g. >8B parameters) and want lots of context tokens = DGX Spark.
* You: Want to create large amounts of synthetic data with large models + distill synthetic data into smaller models via fine-tuing = Get both (DGX Spark for synthetic data generation -> RTX 4090 for small model fine-tuning).

## Benchmark overview

* LLM/VLM inference - Compare LLM/VLM inference speed with `llama.cpp` and `vLLM` inference engines across various models.
* LLM training - Compare LLM full fine-tuning training speed with a relatively small LLM ([Gemma-3-270M](https://huggingface.co/google/gemma-3-270m-it)).
* Image generation - Compare image generation speeds with [Flux.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) and [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo).
* Object detection - Fine-tune an object detection model.

## Requirements

* Install `llama.cpp` - https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md 
     * Setting up llama server - https://github.com/ggml-org/llama.cpp/discussions/16938 
* Install `vLLM` - https://docs.vllm.ai/en/stable/getting_started/installation/gpu/ 
* PyTorch, Transformers, Datasets, Diffusers.

Tests:

* LLM inference (tok/s)
    * 1000x formatting samples and see what each of them look like
    * Frameworks: llama.cpp + vLLM + transformers
    * Models:
        * GPT-OSS-20B / GPT-OSS-120B?
        * GLM 4.7 Flash? - https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/blob/main/GLM-4.7-Flash-UD-Q4_K_XL.gguf 
        * Nemotron v3 nano via vLLM - https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 
        * Nemotron v3 via llama.cpp - https://build.nvidia.com/spark/nemotron/instructions 
            * What's the input/output spec? Fast for now, looks like 50 tok/s outputs
            * Measure this across 1k samples and see what happens
        * Qwen3 series 
            * Qwen3-VL-32B-Instruct - https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8 
            * Qwen3-VL-30B-A3B-FP8 - https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
            * Qwen3-VL-8B-Instruct-FP8 - https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8
            * Can run via unsloth/GGUF/llama.cpp - https://unsloth.ai/docs/models/qwen3-vl-how-to-run-and-fine-tune 
* VLM inference (tok/s | samples/s | total time)
    * Frameworks: llama.cpp + vLLM + transformers 
        * Qwen3-VL-8B, Qwen3-VL-30B-A3B, Qwen3-VL-32B
    * VLM benchmark here: https://docs.vllm.ai/en/stable/benchmarking/cli/ 
* LLM fine-tuning (tok/s | samples/s | total time)
    * Gemma-3-270M-Base, how long does this take? 
    * Start with a small dataset: FoodExtract-1k, compare the results? 
* Image generation (e.g. 1000x images on different prompts, how long per image?)
    * Models:
        * Z-Image-Turbo - https://huggingface.co/Tongyi-MAI/Z-Image-Turbo 
        * Flux.2 collection (all weights of models here):
            * Flux.2-Flein-9B (flux-non-commercial-license) - https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
            * Flux.2-Klein-4B (Apache 2.0) - https://huggingface.co/black-forest-labs/FLUX.2-klein-4B 
                * Flux.2-Klein-4B-NVFP4 - https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-nvfp4 
                    * Note: Requires ComfyUI, perhaps someone could test that?
* Object detection model training 
    * See learnhuggingface.com object detection notebook for more

## LLM/VLM Inference

The main goal here is to test token/s throughput and output.

### vLLM benchmarking (text only and image + text)

Steps:

* Install Docker
* Setup vLLM container on GPU via NVIDIA: https://build.nvidia.com/spark/vllm/instructions 
* Run benchmarks `.sh` scripts, these are slightly modified versions of the vLLM multimodal benchmarking examples: https://docs.vllm.ai/en/stable/benchmarking/cli/#multi-modal-benchmark 

> [!NOTE]
> All tests done with NVIDIA vLLM container version: [25.12.post1-py3](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm?version=25.12.post1-py3), this requires a [docker installation](https://docs.docker.com/engine/install/). 
>
> So you can use: `export LATEST_VLLM_VERSION=25.12.post1-py3`

Example command for starting a vLLM server: 

```
docker run -it --gpus all -p 8000:8000 nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION} vllm serve "MODEL_NAME" --gpu-memory-utilization 0.8 --max_model_len 32000
```

Then you can run the tests, these are executed via a bash script `vllm_image_and_text_testing.sh` or `vllm_text_only_testing.sh` which runs a set of tests:
    * 0 images + random range of input/output tokens
    * 1 images + random range of input/output tokens (note: images are of various sizes randomly sampled to enable real-world test-like conditions)
        * These tests are designed to mimic image + text throughput as well as text only throughput

100 prompts each (~1000 input, ~300 output) with 10 warmup prompts (these are not measured in final output performance).

**Nemotron3-Nano-FP8 - 0 images (model is text only)**  

Source: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8

**Note:** Requires `nano_v3_reasoning_parser.py`. 

Download `nano_v3_reasoning_parser.py` (required for using Nemotron v3 versions):

```
wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8/resolve/main/nano_v3_reasoning_parser.py
```

Start server:

```
docker run -it --gpus all \
  -p 8000:8000 \
  -v $(pwd)/nano_v3_reasoning_parser.py:/workspace/nano_v3_reasoning_parser.py \
  nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION} vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 32000 \
  --port 8000 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3 \
  --kv-cache-dtype fp8
```

Run benchmark:

```
./vllm_text_only_testing.sh nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
```

**Qwen3-VL-32B-Instruct-FP8**

Note: You could only run this on the NVIDIA DGX Spark (too large for the RTX 4090).

Tests done: 0 images and 1 image input.

Start vLLM server:

```
docker run -it --gpus all -p 8000:8000 nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION} vllm serve "Qwen/Qwen3-VL-32B-Instruct-FP8" --gpu-memory-utilization 0.8 --max_model_len 32000
```

Benchmark:

```
./vllm_image_and_text_testing.sh Qwen/Qwen3-VL-32B-Instruct-FP8
```

* ✅ Qwen3-VL-30B-A3B-Instruct-FP8
    * 0 images 
    * 1 image

Start server: 

```
docker run -it --gpus all -p 8000:8000 nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION} vllm serve "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8" --gpu-memory-utilization 0.8 --max_model_len 32000
```

Benchmark:

```
./vllm_image_and_text_testing.sh Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
```

* ✅ Qwen3-VL-8B-Instruct-FP8 
    * 0 images 
    * 1 image

Start server: 

```
docker run -it --gpus all -p 8000:8000 nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION} vllm serve "Qwen/Qwen3-VL-8B-Instruct-FP8" --gpu-memory-utilization 0.8 --max_model_len 8000
```

Benchmark:

```
./vllm_image_and_text_testing.sh Qwen/Qwen3-VL-8B-Instruct-FP8
```

* openai/gpt-oss-20b (RTX 4090 & DGX Spark)
    * 0 images

```
docker run -it --gpus all -p 8000:8000 nvcr.io/nvidia/vllm:25.12.post1-py3 \
bash -c "pip install openai-harmony && vllm serve 'openai/gpt-oss-20b' --gpu-memory-utilization 0.8 --max_model_len 8000"
```

Benchmark:

```
./vllm_text_only_testing.sh openai/gpt-oss-20b
```

* openai/gpt-oss-120b
    * 0 images
    
**Note:** `gpt-oss-120b` requires the OpenAI Harmony package: https://github.com/openai/harmony     

```
docker run -it --gpus all -p 8000:8000 nvcr.io/nvidia/vllm:25.12.post1-py3 \
bash -c "pip install openai-harmony && vllm serve 'openai/gpt-oss-120b' --gpu-memory-utilization 0.8 --max_model_len 32000"
```

Benchmark:

```
./vllm_text_only_testing.sh openai/gpt-oss-120b
```

## Commands 

* Serving from a docker container: 

```
docker run -it --gpus all -p 8000:8000 nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION} vllm serve "Qwen/Qwen2.5-Math-1.5B-Instruct" --gpu-memory-utilization 0.8
```

* For setting up a server with Nemotron-Nanov3 (after following the steps in: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8#use-it-with-vllm):

```
docker run -it --gpus all \
  -p 8000:8000 \
  -v $(pwd)/nano_v3_reasoning_parser.py:/workspace/nano_v3_reasoning_parser.py \
  nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION} vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --max-num-seqs 8 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --port 8000 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3 \
  --kv-cache-dtype fp8
```

* Running a benchmark with different settings (or use `vllm_image_and_text_testing.sh`):

```
docker run -it --network host \
  nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION} \
  vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --endpoint /v1/chat/completions \
  --base-url http://localhost:8000 \
  --dataset-name random-mm \
  --num-prompts 500 \
  --max-concurrency 10 \
  --num-warmups 10 \
  --random-prefix-len 0 \
  --random-input-len 180 \
  --random-output-len 220 \
  --random-range-ratio 0.55 \
  --random-mm-base-items-per-request 1 \
  --random-mm-num-mm-items-range-ratio 0.5 \
  --random-mm-limit-mm-per-prompt '{"image": 2, "video": 0}' \
  --random-mm-bucket-config '{(512, 512, 1): 0.2, (720, 1280, 1): 0.3, (1024, 1024, 1): 0.2, (1080, 1920, 1): 0.3}' \
  --request-rate inf \
  --ignore-eos \
  --seed 42
```

## LLM inference

* Next: implement inference with pure transformers and compare with vLLM (see: https://build.nvidia.com/spark/vllm)
    * Make 3 comparisons of the same model: llama.cpp + transformers native + vLLM and see how they all fair
* Can perform inference with llama.cpp
    * Can perform multimodal inference with llama.cpp as long as we download the multi modal projector, see example: https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF/blob/main/mmproj-BF16.gguf 
    * See docs:
        * Unsloth for Qwen3-VL - https://unsloth.ai/docs/models/qwen3-vl-how-to-run-and-fine-tune 
        * GGUF format - https://github.com/ggml-org/ggml/blob/master/docs/gguf.md 
        * GGUF format (HF docs) - https://huggingface.co/docs/hub/en/gguf  
        * llama-server for running LLMs locally - https://github.com/ggml-org/llama.cpp/discussions/16938 


### llamma.cpp running in a server

**Setup:**

* See the guide for creating a llama.cpp server on DGX Spark: https://build.nvidia.com/spark/nemotron/instructions 
* See the docs for llama.cpp HTTP Server: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md 

**gpt-oss-20b**

Link: https://huggingface.co/unsloth/gpt-oss-20b-GGUF/blob/main/gpt-oss-20b-UD-Q4_K_XL.gguf 

Download model:

```
hf download unsloth/gpt-oss-20b-GGUF gpt-oss-20b-UD-Q4_K_XL.gguf --local-dir ~/models/gpt-oss-20b-gguf/
```

Run a server:

```
./bin/llama-server \
  --model ~/models/gpt-oss-20b-gguf/gpt-oss-20b-UD-Q4_K_XL.gguf \
  --host 0.0.0.0 \
  --port 30000 \
  --n-gpu-layers 99 \
  --ctx-size 16384 \
  --threads 8 \
  --jinja \
  --temp 1.0 \
  --top-p 1.0 \
  --top-k 0
```

Run the text only benchmark:

```
python bench_llama_cpp_text_only.py
```

**Qwen3-VL-8B-GGUF**

Source: https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF/blob/main/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf 

Download a model:

```
hf download unsloth/Qwen3-VL-8B-Instruct-GGUF \Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf --local-dir ~/models/
qwen3-vl-8b-gguf/
```

For multimodal, download the multimodal projector:

```
hf download unsloth/Qwen3-VL-8B-Instruct-GGUF \mmproj-BF16.gguf --local-dir ~/models/
qwen3-vl-8b-gguf/
```

Run a server (with multimodal capabilities):

```
./bin/llama-server \
    --model ~/models/qwen3-vl-8b-gguf/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf \
    --mmproj ~/models/qwen3-vl-8b-gguf/mmproj-BF16.gguf \
    --n-gpu-layers 99 \
    --jinja \
    --top-p 0.8 \
    --top-k 20 \
    --temp 0.7 \
    --min-p 0.0 \
    --presence-penalty 1.5 \
    --ctx-size 8192 \
    --host 0.0.0.0 \
    --port 30000
```

With flash attention on:

```
./bin/llama-server \
    --model ~/models/qwen3-vl-8b-gguf/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf \
    --mmproj ~/models/qwen3-vl-8b-gguf/mmproj-BF16.gguf \
    --n-gpu-layers 99 \
    --jinja \
    --top-p 0.8 \
    --top-k 20 \
    --temp 0.7 \
    --min-p 0.0 \
    --flash-attn auto \
    --presence-penalty 1.5 \
    --ctx-size 8192 \
    --host 0.0.0.0 \
    --port 30000
```

Run the text only benchmark:

```
python bench_llama_cpp_text_only.py
```

Run the multimodal (text + image) benchmark (requires a multimodal model):

```
python bench_llama_cpp_text_and_image.py
```

**Nemotron-3-Nano**

Source: https://huggingface.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF/blob/main/Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf 

> **Note:** Requires 40GB download for Q8 model.

Download a model:

```
hf download unsloth/Nemotron-3-Nano-30B-A3B-GGUF \
  Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf \
  --local-dir ~/models/nemotron3-gguf
```

Run a server:

```
./bin/llama-server \
  --model ~/models/nemotron3-gguf/Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf \
  --host 0.0.0.0 \
  --port 30000 \
  --n-gpu-layers 99 \
  --ctx-size 8192 \
  --threads 8
```

Run the text only benchmark:

```
python bench_llama_cpp_text_only.py
```

## LLM fune-tuning

* Overview: Fully fine-tune a small language model (Gemma-3-270m) on ~1k samples for 3 epochs.
* Source: https://www.learnhuggingface.com/notebooks/hugging_face_llm_full_fine_tune_tutorial (converted notebook into a Python script)
* See: `bench_train_llm_fine_tune.py` for settings.

Run the benchmark:

```
python bench_train_llm_fine_tune.py
```

## Image generation

* Overview: Generate ~100 images with random input prompts time how long each takes to generate.
* Source: [`black-forest-labs/FLUX.2-klein-4B`](black-forest-labs/FLUX.2-klein-4B) & [`Tongyi-MAI/Z-Image-Turbo`](Tongyi-MAI/Z-Image-Turbo).
* See: `bench_z_image_turbo.py` & `bench_flux.py` for settings.

Run the Flux.2 benchmark:

```
python `bench_flux.py`
```

Run the Z-Image-Turbo benchmark:

```
python bench_z_image_turbo.py
```

## Training an object detection model 

* Overview: Fine-tune an object detection model (RT-DETRv2) on a custom dataset of ~1.1k images for 10 epochs.
* Source: https://www.learnhuggingface.com/notebooks/hugging_face_object_detection_tutorial (converted notebook into a Python script)
* See: `bench_train_object_detection.py` for settings. 

```
python bench_train_object_detection.py
```

## Notes and potential extensions

* Defaults where possible: I tried to use the default settings on all available repos if they were available. In the future, benchmarks would likely take into account different specific settings for different scenarios (e.g. long context input/long context output etc).
* Extension: Upgrade the amount of tokens used in context to see how the devices perform under higher load (e.g. large codebase in the context window).
* Extension: Use `llama-bench` for benchmarking `llama.cpp` servers: https://blog.steelph0enix.dev/posts/llama-cpp-guide/#llama-bench
* Extension: Get some numbers on NVFP4 (in my *brief* research, it's still not as fleshed out as I'd like), see example: https://build.nvidia.com/spark/nvfp4-quantization/instructions 
* Analogy for DGX Spark vs RTX 4090 = RTX 4090 is a Ferrari, fast but not much storage. DGX Spark is like a minivan, plenty of storage, not that fast.
* Does Docker slow down inference? (e.g. using NVIDIA's approaved/signed Docker container but does this make things slower?), hat tip: @ibrahimadiallo7444
    * Using this container: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm?version=25.12.post1-py3 
* Check these docs for possible improvements: https://docs.vllm.ai/en/stable/configuration/optimization/#configuration 
* For multimodal, is it best for image/text first?
* For large inference runs, if you wanted to use the same prompt over and over again on different samples, what is the best way to cache the prompt? 
* For different image sizes in vLLM, getting different processing speeds, this was done intentionally to reflect the different image sizes in the wild
    * In practice, if you wanted to maintain the highest throughput, you would likely normalize all the image sizes (make them the same sizes) 
* Why not GLM 4.7 Flash? At the time of recording (late January 2026, vLLM support GLM 4.7 Flash on their main branches but I'm using NVIDIA's vLLM container)
    * "vLLM and SGLang only support GLM-4.7-Flash on their main branches."
    * I'd expect the inference to be similar speeds to Qwen3-VL-30B-A3B-Instruct-FP8 
    * Unsloth makes GLM Flash easier - https://unsloth.ai/docs/models/glm-4.7-flash 
* Could I increase maximum request concurrency? Currently set to default of 10 but perhaps this could be higher/better? 
* If you have any tips of what settings I could use to improve vLLM througput, please let me know.
* Try another inference engine? SGLang? How does this go?
    * See here: https://build.nvidia.com/spark/sglang/overview 
* First time running NVIDIA-Nemotron3-Nano, thinking was enabled so it output 10x more output tokens total compared to GPT-OSS-120B.
* Better flash attention settings = better results? 
    * Flash attention 2 seems questionable on the DGX Spark (as of January 2026)
    * Depending on your platform/task, look into the best settings for that task 
* Leave a comment for which model you'd like to see benchmarked next (optional: and benchmarked what on), I'll add it to the repo 