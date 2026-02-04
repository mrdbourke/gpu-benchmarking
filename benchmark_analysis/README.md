# GPU Benchmark Report

Generated: 2026-02-04 10:25:57

Note: Missing results are shown as NA and assumed to be unavailable because the model did not fit or could not be run on that GPU.

# LLM/VLM Inference

## llama_server_gpt_oss_20b_UD_Q4_K_XL_gguf_samples_100
Family: llama_server
Config: model_name=gpt-oss-20b-UD-Q4_K_XL.gguf, test_type=text_only, num_prompts=100, input_token_range=50-1000, output_token_range=50-1000
| Metric | Unit | NVIDIA_GB10 | NVIDIA_GeForce_RTX_4090 |
| --- | --- | --- | --- |
| true_prompt_throughput | tok/s | 2703.36 | **6035.78** |
| true_gen_throughput | tok/s | 88.04 | **242.54** |
| avg_prompt_speed | tok/s | 2515.66 | **5609.64** |
| avg_gen_speed | tok/s | 88.19 | **243.19** |
| total_runtime | seconds | 586.25 | **215.43** |

## llama_server_Nemotron_3_Nano_30B_A3B_UD_Q8_K_XL_gguf_samples_100
Family: llama_server
Config: model_name=Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf, test_type=text_only, num_prompts=100, input_token_range=50-1000, output_token_range=50-1000
| Metric | Unit | NVIDIA_GB10 | NVIDIA_GeForce_RTX_4090 |
| --- | --- | --- | --- |
| true_prompt_throughput | tok/s | **1037.78** | NA |
| true_gen_throughput | tok/s | **47.56** | NA |
| avg_prompt_speed | tok/s | **963.99** | NA |
| avg_gen_speed | tok/s | **47.66** | NA |

## llama_server_Qwen3_VL_8B_Instruct_UD_Q4_K_XL_gguf_multimodal_samples_100
Family: llama_server
Config: model_name=Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf, test_type=multimodal, prompt_type=mixed, num_prompts=100, output_token_range=50-1000, avg_images_per_request=1.00, mm_bucket_config={(2556; 1179; 1): 0.25; (1080; 1920; 1): 0.25; (1080; 1080; 1): 0.25; (4032; 3024; 1): 0.25}
| Metric | Unit | NVIDIA_GB10 | NVIDIA_GeForce_RTX_4090 |
| --- | --- | --- | --- |
| true_prompt_throughput | tok/s | 799.44 | **2459.23** |
| true_gen_throughput | tok/s | 33.07 | **122.83** |
| avg_prompt_speed | tok/s | 561.41 | **1702.90** |
| avg_gen_speed | tok/s | 20.94 | **78.08** |
| total_runtime | seconds | 389.80 | **115.33** |

## llama_server_Qwen3_VL_8B_Instruct_UD_Q4_K_XL_gguf_samples_100
Family: llama_server
Config: model_name=Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf, test_type=text_only, num_prompts=100, input_token_range=50-1000, output_token_range=50-1000
| Metric | Unit | NVIDIA_GB10 | NVIDIA_GeForce_RTX_4090 |
| --- | --- | --- | --- |
| true_prompt_throughput | tok/s | 2478.36 | **8414.23** |
| true_gen_throughput | tok/s | 35.48 | **132.22** |
| avg_prompt_speed | tok/s | 2373.85 | **7911.93** |
| avg_gen_speed | tok/s | 35.57 | **132.46** |
| total_runtime | seconds | 1388.77 | **383.06** |

## vLLM_gpt_oss_20b_text_only_samples_100
Family: vLLM
Config: model=openai/gpt-oss-20b, test_type=text_only, num_prompts=100, max_concurrency=10, input_len=1000, output_len=300, range_ratio=0.3, thinking_disabled=false
| Metric | Unit | NVIDIA_GB10 | NVIDIA_GeForce_RTX_4090 |
| --- | --- | --- | --- |
| output_token_throughput | tok/s | 230.00 | **584.00** |
| request_throughput | req/s | 1.69 | **5.18** |
| total_token_throughput | tok/s | 1768.38 | **5432.70** |
| ttft_mean | ms | 282.53 | **113.54** |
| tpot_mean | ms | 183.54 | **59.23** |
| itl_mean | ms | 55.56 | **17.95** |
| benchmark_duration | s | 59.28 | **19.29** |

## vLLM_gpt_oss_120b_text_only_samples_100
Family: vLLM
Config: model=openai/gpt-oss-120b, test_type=text_only, num_prompts=100, max_concurrency=10, input_len=1000, output_len=300, range_ratio=0.3
| Metric | Unit | NVIDIA_GB10 | NVIDIA_GeForce_RTX_4090 |
| --- | --- | --- | --- |
| output_token_throughput | tok/s | **116.00** | NA |
| request_throughput | req/s | **0.71** | NA |
| total_token_throughput | tok/s | **744.40** | NA |
| ttft_mean | ms | **577.98** | NA |
| tpot_mean | ms | **459.91** | NA |
| itl_mean | ms | **124.75** | NA |
| benchmark_duration | s | **140.81** | NA |

## vLLM_NVIDIA_Nemotron_3_Nano_30B_A3B_FP8_text_only_samples_100
Family: vLLM
Config: model=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8, test_type=text_only, num_prompts=100, max_concurrency=10, input_len=1000, output_len=300, range_ratio=0.3, thinking_disabled=true
| Metric | Unit | NVIDIA_GB10 | NVIDIA_GeForce_RTX_4090 |
| --- | --- | --- | --- |
| output_token_throughput | tok/s | **230.00** | NA |
| request_throughput | req/s | **0.49** | NA |
| total_token_throughput | tok/s | **643.34** | NA |
| ttft_mean | ms | **610.81** | NA |
| tpot_mean | ms | **65.30** | NA |
| itl_mean | ms | **65.06** | NA |
| benchmark_duration | s | **204.32** | NA |

## vLLM_Qwen3_VL_8B_Instruct_FP8_samples_100_images_0
Family: vLLM
Config: model=Qwen/Qwen3-VL-8B-Instruct-FP8, num_prompts=100, max_concurrency=10, input_len=1000, output_len=300, images_per_request=0, image_bucket=512x512(20%) 720x1280(30%) 1024x1024(20%) 1080x1920(30%), range_ratio=0.3
| Metric | Unit | NVIDIA_GB10 | NVIDIA_GeForce_RTX_4090 |
| --- | --- | --- | --- |
| output_token_throughput | tok/s | 220.00 | **580.00** |
| request_throughput | req/s | 0.63 | **1.64** |
| total_token_throughput | tok/s | 830.73 | **2155.83** |
| ttft_mean | ms | 273.10 | **124.11** |
| tpot_mean | ms | 50.01 | **19.15** |
| itl_mean | ms | 49.85 | **19.09** |
| benchmark_duration | s | 158.23 | **60.97** |

## vLLM_Qwen3_VL_8B_Instruct_FP8_samples_100_images_1
Family: vLLM
Config: model=Qwen/Qwen3-VL-8B-Instruct-FP8, num_prompts=100, max_concurrency=10, input_len=1000, output_len=300, images_per_request=1, image_bucket=512x512(20%) 720x1280(30%) 1024x1024(20%) 1080x1920(30%), range_ratio=0.3
| Metric | Unit | NVIDIA_GB10 | NVIDIA_GeForce_RTX_4090 |
| --- | --- | --- | --- |
| output_token_throughput | tok/s | 191.00 | **392.00** |
| request_throughput | req/s | 0.49 | **0.87** |
| total_token_throughput | tok/s | 642.77 | **1144.26** |
| ttft_mean | ms | **714.11** | 4644.34 |
| tpot_mean | ms | 64.00 | **21.13** |
| itl_mean | ms | 63.90 | **21.10** |
| benchmark_duration | s | 204.50 | **114.87** |

## vLLM_Qwen3_VL_30B_A3B_Instruct_FP8_samples_100_images_0
Family: vLLM
Config: model=Qwen/Qwen3-VL-30B-A3B-Instruct-FP8, num_prompts=100, max_concurrency=10, input_len=1000, output_len=300, images_per_request=0, image_bucket=512x512(20%) 720x1280(30%) 1024x1024(20%) 1080x1920(30%), range_ratio=0.3
| Metric | Unit | NVIDIA_GB10 | NVIDIA_GeForce_RTX_4090 |
| --- | --- | --- | --- |
| output_token_throughput | tok/s | **150.00** | NA |
| request_throughput | req/s | **0.40** | NA |
| total_token_throughput | tok/s | **519.84** | NA |
| ttft_mean | ms | **338.31** | NA |
| tpot_mean | ms | **82.34** | NA |
| itl_mean | ms | **82.00** | NA |
| benchmark_duration | s | **252.86** | NA |

## vLLM_Qwen3_VL_30B_A3B_Instruct_FP8_samples_100_images_1
Family: vLLM
Config: model=Qwen/Qwen3-VL-30B-A3B-Instruct-FP8, num_prompts=100, max_concurrency=10, input_len=1000, output_len=300, images_per_request=1, image_bucket=512x512(20%) 720x1280(30%) 1024x1024(20%) 1080x1920(30%), range_ratio=0.3
| Metric | Unit | NVIDIA_GB10 | NVIDIA_GeForce_RTX_4090 |
| --- | --- | --- | --- |
| output_token_throughput | tok/s | **130.00** | NA |
| request_throughput | req/s | **0.34** | NA |
| total_token_throughput | tok/s | **446.02** | NA |
| ttft_mean | ms | **766.03** | NA |
| tpot_mean | ms | **94.84** | NA |
| itl_mean | ms | **94.51** | NA |
| benchmark_duration | s | **294.71** | NA |

## vLLM_Qwen3_VL_32B_Instruct_FP8_samples_100_images_0
Family: vLLM
Config: model=Qwen/Qwen3-VL-32B-Instruct-FP8, num_prompts=100, max_concurrency=10, input_len=1000, output_len=300, images_per_request=0, image_bucket=512x512(20%) 720x1280(30%) 1024x1024(20%) 1080x1920(30%), range_ratio=0.3
| Metric | Unit | NVIDIA_GB10 | NVIDIA_GeForce_RTX_4090 |
| --- | --- | --- | --- |
| output_token_throughput | tok/s | **65.00** | NA |
| request_throughput | req/s | **0.18** | NA |
| total_token_throughput | tok/s | **231.20** | NA |
| ttft_mean | ms | **1036.98** | NA |
| tpot_mean | ms | **179.14** | NA |
| itl_mean | ms | **178.57** | NA |
| benchmark_duration | s | **568.53** | NA |

## vLLM_Qwen3_VL_32B_Instruct_FP8_samples_100_images_1
Family: vLLM
Config: model=Qwen/Qwen3-VL-32B-Instruct-FP8, num_prompts=100, max_concurrency=10, input_len=1000, output_len=300, images_per_request=1, image_bucket=512x512(20%) 720x1280(30%) 1024x1024(20%) 1080x1920(30%), range_ratio=0.3
| Metric | Unit | NVIDIA_GB10 | NVIDIA_GeForce_RTX_4090 |
| --- | --- | --- | --- |
| output_token_throughput | tok/s | **62.00** | NA |
| request_throughput | req/s | **0.15** | NA |
| total_token_throughput | tok/s | **196.18** | NA |
| ttft_mean | ms | **1877.67** | NA |
| tpot_mean | ms | **210.39** | NA |
| itl_mean | ms | **209.71** | NA |
| benchmark_duration | s | **670.01** | NA |


# LLM Training

## llm_finetune_bench_gemma_3_270m_it_epochs_3
Family: llm_finetune_bench
Config: model_name=google/gemma-3-270m-it, dataset_name=mrdbourke/FoodExtract-1k, num_epochs=3, batch_size=8, max_length=512, gradient_checkpointing=False
| Metric | Unit | NVIDIA_GB10 | NVIDIA_GeForce_RTX_4090 |
| --- | --- | --- | --- |
| total_train_time | seconds | 343.20 | **81.59** |
| total_train_time_minutes | minutes | 5.72 | **1.36** |
| inference_samples_per_second | samples/s | 1.87 | **3.39** |
| inference_tokens_per_second | tokens/s | 67.79 | **106.15** |


# Image Generation

## flux_bench_FLUX_2_klein_4B_n100
Family: flux_bench
Config: model_id=black-forest-labs/FLUX.2-klein-4B, num_images=100, inference_steps=4, resolution=1024x1024, guidance_scale=1.0, dtype=bf16
| Metric | Unit | NVIDIA_GB10 | NVIDIA_GeForce_RTX_4090 |
| --- | --- | --- | --- |
| images_per_second | img/s | 0.2380 | **0.7764** |
| seconds_per_image | s/img | 4.2012 | **1.2880** |
| total_time | seconds | 420.12 | **128.80** |
| steps_per_second | steps/s | 0.95 | **3.11** |

## zimage_bench_Z_Image_Turbo_n100
Family: zimage_bench
Config: model_id=Tongyi-MAI/Z-Image-Turbo, num_images=100, inference_steps=9, dit_forwards_per_image=8, resolution=1024x1024, guidance_scale=0.0, dtype=bf16, attention_backend=sdpa
| Metric | Unit | NVIDIA_GB10 | NVIDIA_GeForce_RTX_4090 |
| --- | --- | --- | --- |
| images_per_second | img/s | 0.0598 | **0.2053** |
| seconds_per_image | s/img | 16.7194 | **4.8713** |
| total_time | seconds | 1671.94 | **487.13** |
| steps_per_second | steps/s | 0.54 | **1.85** |
| dit_forwards_per_second | forwards/s | 0.48 | **1.64** |


# Object Detection

## object_detection_bench_rtdetr_v2_r50vd_epochs_10
Family: object_detection_bench
Config: model_name=PekingU/rtdetr_v2_r50vd, dataset_name=mrdbourke/trashify_manual_labelled_images, num_epochs=10, batch_size=8, image_size=640, fp16=False
| Metric | Unit | NVIDIA_GB10 | NVIDIA_GeForce_RTX_4090 |
| --- | --- | --- | --- |
| samples_per_second | samples/s | 0.08 | **0.29** |
| total_train_time | seconds | 807.05 | **222.36** |
| total_train_time_minutes | minutes | 13.45 | **3.71** |
