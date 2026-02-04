# Comapre GPUs on different tasks

The main goal of this repo is to benchmark the NVIDIA DGX Spark against the NVIDIA RTX 4090 on a series of ML/AI tasks.

Why these two GPUs? 

They are the two GPUs I own.

But the tests could easily be extended to other GPUs.

Most of the analysis in `benchmark_analysis/` was performed by GPT-Codex simply to compare the results in `benchmark_results/` (this holds the ground truth for a series of tests in `.csv` files). 

## Summary

**One line:** RTX 4090 faster on training and inference by ~3-4x but cannot run larger models like the DGX Spark (24GB VRAM vs 128GB VRAM).

### Performance

The NVIDIA DGX Spark is a like a minivan.

Whereas the NVIDIA RTX 4090 is like Ferrari.

What I mean by this is that for any compute intensive task, the RTX 4090 is generally *faster* (on training and inference) than the DGX Spark on an order of 3-4x.

However, there are several benchmarks (e.g. running `gpt-oss-120b`) where the RTX 4090 simply can't because it doesn't have enough space (VRAM).

So the DGX Spark while generally slower than the RTX 4090 can run much larger models.

### Getting started and machine footprint

DGX Spark is one plug and play. Go from unboxing to running models in ~30 mins. Especially with the [NVIDIA DGX Spark website tutorials](https://build.nvidia.com/spark).

RTX 4090 requires a custom built PC (this took me and my friend ~3-4 hours) as well as software setup (~2-3 hours with the help of AIs).

The DGX Spark also has a small footprint, similar to the size of a small textbook.

The RTX 4090, depending on your PC build will have a much larger footprint.

TK image - compare the footprints of the two machines 


Goal: Run several ML tasks and measure performance on each, then collate the results into a report.

## Requirements

* Install `llama.cpp` - https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md 
* Install vLLM - https://docs.vllm.ai/en/stable/getting_started/installation/gpu/ 

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

### Stats

### vLLM benchmarking (text only and image + text)

Steps:

* Setup vLLM container on GPU via NVIDIA: https://build.nvidia.com/spark/vllm/instructions
* Run benchmarks via vLLM: https://docs.vllm.ai/en/stable/benchmarking/cli/#multi-modal-benchmark 

> [!NOTE]
> All tests done with NVIDIA vLLM container version: [25.12.post1-py3](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm?version=25.12.post1-py3), this requires a [docker installation](https://docs.docker.com/engine/install/). 
>
> So you can use: `export LATEST_VLLM_VERSION=25.12.post1-py3`

```
docker run -it --gpus all -p 8000:8000 nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION} vllm serve "MODEL_NAME" --gpu-memory-utilization 0.8 --max_model_len 32000
```

* Then you run the tests, these are executed via a bash script `vllm_image_and_text_testing.sh` which runs a set of tests:
    * 0 images + random range of input/output tokens
    * 1 images + random range of input/output tokens (note: images are of various sizes randomly sampled to enable real-world test-like conditions)
        * These tests are designed to mimic image + text throughput as well as text only throughput

100 prompts each (~1000 input, ~300 output): 

* ✅ Nemotron3-Nano-FP8 - 0 images (model is text only) (requires `nano_v3_reasoning_parser.py`)

Download `nano_v3_reasoning_parser.py`:

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

* ✅ Qwen3-VL-32B-Instruct-FP8, note: you could only run this on the NVIDIA DGX Spark (too large for the RTX 4090)
    * 0 images
    * 1 image

Start server:

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

### Llamma.cpp running in a server

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

**Qwen3-VL-8B-GGUF**

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

**Nemotron-3-Nano**

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

**Qwen3-VL-8B-Instruct** 

Capable of doing images + text.



Run a server:

(requires downloading the `mmproj-F16.gguf` file from hf)

```
./bin/llama-server \
  --model ~/models/qwen-3-vl-gguf/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf \
  --mmproj ~/models/qwen-3-vl-gguf/mmproj-BF16.gguf \
  --host 0.0.0.0 \
  --port 30000 \
  --n-gpu-layers 99 \
  --jinja \
  --top-p 0.8 \
  --top-k 20 \
  --temp 0.7 \
  --min-p 0.0 \
  --presence-penalty 1.5 \
  --ctx-size 8192
```

Run the text only benchmark:

```
python bench_llama_cpp_text_only.py
```

Run the multimodal (text + image) benchmark (requires a multimodal model):

```
python bench_llama_cpp_text_and_image.py
```



NVIDIA DGX Spark + llama.cpp
TODO: Model: GPT-OSS-120B? 

TODO: Model: Nemotronv3-Nano 

Model: Qwen3-VL-30B-A3B-Instruct-UD-Q4_K_XL.gguf
Reading - 1222 tokens - 1.27s - 961.00 tokens/s
Generation - 1,016 tokens - 13.44s - 75.58 tokens/s

Model: Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf
Reading - 1219 tokens - 1.47s - 831.42 tokens/s
Output - 125 tokens - 3.23s - 38.65 tokens/s

## Image generation tests

* Create a list of ~1000 image generation prompts, save them to file, iterate through each one and see how long it takes to create an image based on each prompt.
    * Potentially could go: recipe -> generate a prompt based on the recipe -> generate an image

## Training and object detection model 

See: `train_object_detection_model.py`

## Notes

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
    * Flash attention 2 seems questionable on the DGX Spark
    * Depending on your platform/task, look into the best settings for that task 
* Defaults where possible: I tried to use the default settings on all available repos if they were available
* Leave a comment for which model you'd like to see benchmarked next (optional: and benchmarked what on), I'll add it to the repo 

```
Nemotronv3-Nano Thinking=On
============ Serving Benchmark Result ============
Successful requests:                     100       
Failed requests:                         0         
Maximum request concurrency:             10        
Benchmark duration (s):                  232.07    
Total input tokens:                      101603    
Total generated tokens:                  29842     
Request throughput (req/s):              0.43      
Output token throughput (tok/s):         128.59    
Peak output token throughput (tok/s):    184.00    
Peak concurrent requests:                13.00     
Total Token throughput (tok/s):          566.41    
---------------Time to First Token----------------
Mean TTFT (ms):                          4817.19   
Median TTFT (ms):                        4713.98   
P99 TTFT (ms):                           12820.38  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          58.98     
Median TPOT (ms):                        59.81     
P99 TPOT (ms):                           62.41     
---------------Inter-token Latency----------------
Mean ITL (ms):                           58.86     
Median ITL (ms):                         52.52     
P99 ITL (ms):                            366.10    
==================================================
```

## Log

* Next:
    * Finish off series of tests on the DGX Spark
        * GPT-OSS-20B? 
        * Transformers? (this could be a simple one of just a script with multiple models running with simulated tokens in/out, just copy the settings from the vLLM setup)
        * llama.cpp with the model suite
    * Then: image generation, object detection model fine-tuning, LLM fine-tuning 
    * Commit repo to GitHub for easy cloning + updating
    * Run the tests on RTX 4090 and see how it compares