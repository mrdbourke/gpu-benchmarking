#!/usr/bin/env python3
"""
LLM Fine-tuning Training Script with Timing Metrics

This script fine-tunes a Small Language Model (SLM) on the FoodExtract dataset
and outputs training/evaluation time metrics to a CSV file.

Based on: https://www.learnhuggingface.com/notebooks/hugging_face_llm_full_fine_tune_tutorial

Usage:
    Run on your local machine (requires GPU for efficient training):
    
    python train_llm_fine_tune.py --num_epochs 3 --batch_size 8 --output_dir ./benchmark_results
    
    For a quick test run:
    python train_llm_fine_tune.py --num_epochs 1 --batch_size 4 --output_dir ./benchmark_results

Required packages:
    pip install torch transformers datasets trl tqdm
"""

import argparse
import csv
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from trl import SFTTrainer, SFTConfig


# ============================================================================
# Configuration and Data Classes
# ============================================================================

@dataclass
class TimingMetrics:
    """Stores timing metrics for training and evaluation."""
    epoch: int
    train_time_seconds: float
    eval_time_seconds: Optional[float] = None
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    samples_per_second: Optional[float] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class InferenceMetrics:
    """Stores metrics for inference benchmarking."""
    batch_size: int
    total_samples: int
    total_time_seconds: float
    samples_per_second: float
    tokens_generated: int = 0
    tokens_per_second: float = 0.0


# ============================================================================
# Device Detection Functions
# ============================================================================

def get_gpu_name() -> str:
    """Detect GPU name, sanitize for filenames."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        raw_name = result.stdout.strip().split("\n")[0]
        sanitized = re.sub(r'[^a-zA-Z0-9]', '_', raw_name)
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
        return sanitized if sanitized else "Unknown_GPU"
    except Exception:
        return "Unknown_GPU"


def get_device_info() -> Tuple[torch.device, str, str]:
    """
    Detect and return device information.
    
    Returns:
        Tuple of (device, device_info_string, sanitized_device_name)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_info = f"CUDA - {torch.cuda.get_device_name(0)}"
        device_sanitized = get_gpu_name()
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_info = "Apple MPS"
        device_sanitized = "Apple_MPS"
    else:
        device = torch.device("cpu")
        device_info = "CPU"
        device_sanitized = "CPU"
    
    return device, device_info, device_sanitized


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information if available."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        free_memory = total_memory - reserved_memory
        
        return {
            "total_memory_gb": total_memory / 1e9,
            "allocated_memory_gb": allocated_memory / 1e9,
            "reserved_memory_gb": reserved_memory / 1e9,
            "free_memory_gb": free_memory / 1e9,
        }
    return {}


# ============================================================================
# Data Preprocessing Functions
# ============================================================================

def sample_to_conversation(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a dataset sample to conversation format for SFT training.
    
    Args:
        sample: A single sample from the dataset
    
    Returns:
        Dictionary with 'messages' key containing the conversation
    """
    return {
        "messages": [
            {"role": "user", "content": sample["sequence"]},
            {"role": "assistant", "content": sample["gpt-oss-120b-label-condensed"]}
        ]
    }


def prepare_dataset(
    dataset_name: str,
    test_split: float = 0.2,
    seed: int = 42
) -> DatasetDict:
    """
    Load and prepare the dataset for training.
    
    Args:
        dataset_name: Name of the Hugging Face dataset
        test_split: Fraction of data to use for testing
        seed: Random seed for reproducibility
    
    Returns:
        DatasetDict with train and test splits
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(path=dataset_name)
    
    # Map samples to conversation format
    dataset = dataset.map(sample_to_conversation, batched=False)
    
    # Create train/test split
    dataset_split = dataset["train"].train_test_split(
        test_size=test_split,
        shuffle=True,
        seed=seed
    )
    
    return dataset_split


# ============================================================================
# Model Creation
# ============================================================================

def create_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    attn_implementation: str = "eager"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Creates and returns an LLM model and tokenizer.
    
    Args:
        model_name: The name or path of the pretrained model
        device: Device to move the model to
        attn_implementation: Attention implementation to use
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        dtype="auto",
        attn_implementation=attn_implementation
    )

    model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def get_model_num_params(model: AutoModelForCausalLM) -> Dict[str, int]:
    """
    Returns the number of trainable, non-trainable and total parameters.
    
    Args:
        model: The PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params
    
    return {
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "total_params": total_params
    }


# ============================================================================
# Custom Trainer with Timing
# ============================================================================

class TimedSFTTrainer(SFTTrainer):
    """SFTTrainer subclass that tracks timing metrics."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timing_metrics: List[TimingMetrics] = []
        self.epoch_start_time: Optional[float] = None
        self.current_epoch: int = 0
        self.train_start_time: Optional[float] = None
        self.total_train_time: float = 0.0
    
    def train(self, *args, **kwargs):
        """Override train to track overall training time."""
        self.train_start_time = time.time()
        result = super().train(*args, **kwargs)
        self.total_train_time = time.time() - self.train_start_time
        return result
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Track training step timing."""
        if self.epoch_start_time is None:
            self.epoch_start_time = time.time()
        return super().training_step(model, inputs, num_items_in_batch)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Record epoch timing metrics."""
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            
            # Get training loss from state
            train_loss = None
            eval_loss = None
            if state.log_history:
                for log in reversed(state.log_history):
                    if "loss" in log and train_loss is None:
                        train_loss = log["loss"]
                    if "eval_loss" in log and eval_loss is None:
                        eval_loss = log["eval_loss"]
                    if train_loss is not None and eval_loss is not None:
                        break
            
            metrics = TimingMetrics(
                epoch=self.current_epoch + 1,
                train_time_seconds=epoch_time,
                train_loss=train_loss,
                eval_loss=eval_loss
            )
            self.timing_metrics.append(metrics)
            self.current_epoch += 1
            self.epoch_start_time = None
        
        return super().on_epoch_end(args, state, control, **kwargs)


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_model_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_dataset,
    device: torch.device,
    batch_size: int = 1,
    max_new_tokens: int = 256,
    num_samples: Optional[int] = None
) -> InferenceMetrics:
    """
    Evaluates the model inference speed.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        test_dataset: Test dataset
        device: Device to run evaluation on
        batch_size: Batch size for inference
        max_new_tokens: Maximum new tokens to generate
        num_samples: Number of samples to evaluate (None for all)
    
    Returns:
        InferenceMetrics containing timing information
    """
    model.eval()
    
    # Create pipeline for inference
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device if device.type != "cuda" else 0
    )
    
    # Prepare prompts
    if num_samples is not None:
        samples = test_dataset.select(range(min(num_samples, len(test_dataset))))
    else:
        samples = test_dataset
    
    prompts = [
        tokenizer.apply_chat_template(
            item["messages"][:1],
            tokenize=False,
            add_generation_prompt=True
        )
        for item in samples
    ]
    
    total_tokens = 0
    start_time = time.time()
    
    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating inference", unit="batch"):
        batch_prompts = prompts[i:i + batch_size]
        outputs = pipe(
            batch_prompts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            return_full_text=False
        )
        
        # Count generated tokens
        for output in outputs:
            generated_text = output[0]["generated_text"] if isinstance(output, list) else output["generated_text"]
            total_tokens += len(tokenizer.encode(generated_text))
    
    total_time = time.time() - start_time
    
    return InferenceMetrics(
        batch_size=batch_size,
        total_samples=len(prompts),
        total_time_seconds=total_time,
        samples_per_second=len(prompts) / total_time if total_time > 0 else 0,
        tokens_generated=total_tokens,
        tokens_per_second=total_tokens / total_time if total_time > 0 else 0
    )


# ============================================================================
# CSV Output Functions
# ============================================================================

def save_metrics_to_csv(
    metrics: List[TimingMetrics],
    output_path: str,
    total_train_time: float,
    device_info: str,
    device_sanitized: str,
    args: argparse.Namespace,
    model_params: Dict[str, int],
    inference_metrics: Optional[InferenceMetrics] = None
):
    """
    Saves timing metrics to a CSV file in metric,value,unit format.
    
    Args:
        metrics: List of TimingMetrics objects
        output_path: Path to the output CSV file
        total_train_time: Total training time in seconds
        device_info: Information about the training device
        device_sanitized: Sanitized device name for filenames
        args: Command line arguments used for training
        model_params: Dictionary with model parameter counts
        inference_metrics: Optional inference metrics
    """
    # Calculate summary statistics
    train_times = [m.train_time_seconds for m in metrics if m.train_time_seconds]
    train_losses = [m.train_loss for m in metrics if m.train_loss is not None]
    eval_losses = [m.eval_loss for m in metrics if m.eval_loss is not None]
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["metric", "value", "unit"])
        
        # Device and model info
        writer.writerow(["device", device_info, ""])
        writer.writerow(["device_sanitized", device_sanitized, ""])
        writer.writerow(["model_name", args.model_name, ""])
        writer.writerow(["dataset_name", args.dataset_name, ""])
        writer.writerow(["timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"), ""])
        writer.writerow(["test_type", "llm_fine_tuning", ""])
        
        # Model parameters
        writer.writerow(["total_parameters", f"{model_params['total_params']:,}", ""])
        writer.writerow(["trainable_parameters", f"{model_params['trainable_params']:,}", ""])
        writer.writerow(["total_parameters_millions", f"{model_params['total_params'] / 1e6:.2f}", "M"])
        
        # Training configuration
        writer.writerow(["num_epochs", args.num_epochs, ""])
        writer.writerow(["batch_size", args.batch_size, ""])
        writer.writerow(["learning_rate", args.learning_rate, ""])
        writer.writerow(["weight_decay", args.weight_decay, ""])
        writer.writerow(["warmup_ratio", args.warmup_ratio, ""])
        writer.writerow(["max_length", args.max_length, "tokens"])
        writer.writerow(["test_split", args.test_split, ""])
        writer.writerow(["gradient_checkpointing", args.gradient_checkpointing, ""])
        writer.writerow(["seed", args.seed, ""])
        
        # GPU memory info if available
        gpu_mem = get_gpu_memory_info()
        if gpu_mem:
            writer.writerow(["gpu_total_memory", f"{gpu_mem['total_memory_gb']:.2f}", "GB"])
        
        # Timing metrics
        writer.writerow(["total_train_time", f"{total_train_time:.2f}", "seconds"])
        writer.writerow(["total_train_time_minutes", f"{total_train_time / 60:.2f}", "minutes"])
        
        # Per-epoch training time stats
        if train_times:
            writer.writerow(["avg_epoch_train_time", f"{np.mean(train_times):.2f}", "seconds"])
            writer.writerow(["min_epoch_train_time", f"{np.min(train_times):.2f}", "seconds"])
            writer.writerow(["max_epoch_train_time", f"{np.max(train_times):.2f}", "seconds"])
            writer.writerow(["std_epoch_train_time", f"{np.std(train_times):.2f}", "seconds"])
        
        # Loss metrics
        if train_losses:
            writer.writerow(["initial_train_loss", f"{train_losses[0]:.4f}", ""])
            writer.writerow(["final_train_loss", f"{train_losses[-1]:.4f}", ""])
            writer.writerow(["min_train_loss", f"{np.min(train_losses):.4f}", ""])
        
        if eval_losses:
            writer.writerow(["initial_eval_loss", f"{eval_losses[0]:.4f}", ""])
            writer.writerow(["final_eval_loss", f"{eval_losses[-1]:.4f}", ""])
            writer.writerow(["min_eval_loss", f"{np.min(eval_losses):.4f}", ""])
        
        # Inference metrics
        if inference_metrics:
            writer.writerow(["inference_batch_size", inference_metrics.batch_size, ""])
            writer.writerow(["inference_total_samples", inference_metrics.total_samples, ""])
            writer.writerow(["inference_total_time", f"{inference_metrics.total_time_seconds:.2f}", "seconds"])
            writer.writerow(["inference_samples_per_second", f"{inference_metrics.samples_per_second:.2f}", "samples/s"])
            writer.writerow(["inference_tokens_generated", inference_metrics.tokens_generated, "tokens"])
            writer.writerow(["inference_tokens_per_second", f"{inference_metrics.tokens_per_second:.2f}", "tokens/s"])
        
        # Per-epoch details
        for i, metric in enumerate(metrics):
            epoch_num = i + 1
            writer.writerow([f"epoch_{epoch_num}_train_time", f"{metric.train_time_seconds:.2f}", "seconds"])
            if metric.train_loss is not None:
                writer.writerow([f"epoch_{epoch_num}_train_loss", f"{metric.train_loss:.4f}", ""])
            if metric.eval_loss is not None:
                writer.writerow([f"epoch_{epoch_num}_eval_loss", f"{metric.eval_loss:.4f}", ""])
    
    print(f"\nCSV saved to: {output_path}")


def print_summary(
    metrics: List[TimingMetrics],
    total_train_time: float,
    device_info: str,
    model_params: Dict[str, int],
    inference_metrics: Optional[InferenceMetrics] = None
):
    """Prints a summary of training metrics to console."""
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Device: {device_info}")
    print(f"Model Parameters: {model_params['total_params']:,} ({model_params['total_params'] / 1e6:.2f}M)")
    print(f"Trainable Parameters: {model_params['trainable_params']:,}")
    print(f"Total Training Time: {total_train_time:.2f} seconds ({total_train_time / 60:.2f} minutes)")
    print(f"Number of Epochs: {len(metrics)}")
    
    train_times = [m.train_time_seconds for m in metrics if m.train_time_seconds]
    if train_times:
        print(f"\nPer-Epoch Training Time:")
        print(f"  Average: {np.mean(train_times):.2f} seconds")
        print(f"  Min: {np.min(train_times):.2f} seconds")
        print(f"  Max: {np.max(train_times):.2f} seconds")
    
    train_losses = [m.train_loss for m in metrics if m.train_loss is not None]
    if train_losses:
        print(f"\nTraining Loss:")
        print(f"  Initial: {train_losses[0]:.4f}")
        print(f"  Final: {train_losses[-1]:.4f}")
    
    eval_losses = [m.eval_loss for m in metrics if m.eval_loss is not None]
    if eval_losses:
        print(f"\nEvaluation Loss:")
        print(f"  Initial: {eval_losses[0]:.4f}")
        print(f"  Final: {eval_losses[-1]:.4f}")
    
    if inference_metrics:
        print(f"\nInference Metrics (batch_size={inference_metrics.batch_size}):")
        print(f"  Samples: {inference_metrics.total_samples}")
        print(f"  Total Time: {inference_metrics.total_time_seconds:.2f} seconds")
        print(f"  Samples/Second: {inference_metrics.samples_per_second:.2f}")
        print(f"  Tokens/Second: {inference_metrics.tokens_per_second:.2f}")
    
    print("=" * 60)


# ============================================================================
# Main Training Function
# ============================================================================

def main(args: argparse.Namespace):
    """Main training function."""
    
    # -------------------------------------------------------------------------
    # Setup device
    # -------------------------------------------------------------------------
    device, device_info, device_sanitized = get_device_info()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'=' * 60}")
    print("LLM FINE-TUNING BENCHMARK SCRIPT")
    print(f"{'=' * 60}")
    print(f"Device: {device_info}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Max Sequence Length: {args.max_length}")
    print(f"Seed: {args.seed}")
    
    # Print GPU memory if available
    gpu_mem = get_gpu_memory_info()
    if gpu_mem:
        print(f"GPU Memory: {gpu_mem['total_memory_gb']:.2f} GB total, {gpu_mem['free_memory_gb']:.2f} GB free")
    print(f"{'=' * 60}\n")
    
    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    print("Loading and preparing dataset...")
    dataset = prepare_dataset(
        dataset_name=args.dataset_name,
        test_split=args.test_split,
        seed=args.seed
    )
    
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    # -------------------------------------------------------------------------
    # Load model and tokenizer
    # -------------------------------------------------------------------------
    print("\nLoading model and tokenizer...")
    model, tokenizer = create_model_and_tokenizer(
        model_name=args.model_name,
        device=device,
        attn_implementation=args.attn_implementation
    )
    
    # Get model parameters
    model_params = get_model_num_params(model)
    print(f"Total parameters: {model_params['total_params']:,}")
    print(f"Trainable parameters: {model_params['trainable_params']:,}")
    print(f"Model dtype: {model.dtype}")
    print(f"Model device: {model.device}")
    
    # -------------------------------------------------------------------------
    # Setup output directory
    # -------------------------------------------------------------------------
    output_dir = Path(args.output_dir) / device_sanitized
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    model_short = re.sub(r'[^a-zA-Z0-9]', '_', args.model_name.split('/')[-1])
    model_short = re.sub(r'_+', '_', model_short).strip('_')[:50]
    csv_filename = f"llm_finetune_bench_{device_sanitized}_{model_short}_epochs_{args.num_epochs}_{timestamp}.csv"
    csv_path = output_dir / csv_filename
    
    # -------------------------------------------------------------------------
    # Setup training configuration
    # -------------------------------------------------------------------------
    print("\nSetting up training configuration...")
    
    training_output_dir = output_dir / "training_output"
    training_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine precision settings
    use_fp16 = args.fp16 and torch.cuda.is_available()
    use_bf16 = args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    # If neither specified, auto-detect based on model dtype
    if not use_fp16 and not use_bf16:
        if model.dtype == torch.bfloat16:
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        elif model.dtype == torch.float16:
            use_fp16 = torch.cuda.is_available()
    
    sft_config = SFTConfig(
        output_dir=str(training_output_dir),
        max_length=args.max_length,
        packing=args.packing,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optimizer,
        logging_steps=args.logging_steps,
        save_strategy="epoch" if args.save_each_epoch else "no",
        eval_strategy="epoch" if args.eval_each_epoch else "no",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        fp16=use_fp16,
        bf16=use_bf16,
        save_total_limit=2,
        push_to_hub=False,
        report_to="none",
        seed=args.seed,
    )
    
    print(f"Using FP16: {use_fp16}")
    print(f"Using BF16: {use_bf16}")
    
    # -------------------------------------------------------------------------
    # Create trainer and train
    # -------------------------------------------------------------------------
    trainer = TimedSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if args.eval_each_epoch else None,
        processing_class=tokenizer,
    )
    
    print("\nStarting training...")
    print("-" * 40)
    
    train_result = trainer.train()
    
    print("-" * 40)
    print("Training complete!")
    
    # -------------------------------------------------------------------------
    # Run inference benchmark
    # -------------------------------------------------------------------------
    inference_metrics = None
    if args.run_inference_bench:
        print("\nRunning inference benchmark...")
        inference_metrics = evaluate_model_inference(
            model=model,
            tokenizer=tokenizer,
            test_dataset=dataset["test"],
            device=device,
            batch_size=args.inference_batch_size,
            max_new_tokens=args.max_new_tokens,
            num_samples=args.inference_samples
        )
        print(f"Inference complete: {inference_metrics.samples_per_second:.2f} samples/second")
    
    # -------------------------------------------------------------------------
    # Save metrics to CSV
    # -------------------------------------------------------------------------
    print(f"\nSaving metrics to {csv_path}...")
    save_metrics_to_csv(
        metrics=trainer.timing_metrics,
        output_path=str(csv_path),
        total_train_time=trainer.total_train_time,
        device_info=device_info,
        device_sanitized=device_sanitized,
        args=args,
        model_params=model_params,
        inference_metrics=inference_metrics
    )
    
    # Print summary
    print_summary(
        metrics=trainer.timing_metrics,
        total_train_time=trainer.total_train_time,
        device_info=device_info,
        model_params=model_params,
        inference_metrics=inference_metrics
    )
    
    # -------------------------------------------------------------------------
    # Save model (optional)
    # -------------------------------------------------------------------------
    if args.save_model:
        model_save_path = output_dir / "final_model"
        print(f"\nSaving model to: {model_save_path}")
        trainer.save_model(str(model_save_path))
        tokenizer.save_pretrained(str(model_save_path))
    
    return trainer.timing_metrics


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune an LLM and measure training time metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and dataset
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-270m-it",
        help="Name or path of the pretrained model (e.g., google/gemma-3-270m-it)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mrdbourke/FoodExtract-1k",
        help="Name of the Hugging Face dataset"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "flash_attention_2", "sdpa"],
        help="Attention implementation to use"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for learning rate scheduler"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["linear", "cosine", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw_torch_fused",
        help="Optimizer to use (adamw_torch_fused recommended for GPU)"
    )
    
    # Data processing
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing"
    )
    parser.add_argument(
        "--packing",
        action="store_true",
        help="Enable sequence packing for efficiency"
    )
    
    # Training options
    parser.add_argument(
        "--eval_each_epoch",
        action="store_true",
        default=True,
        help="Run evaluation after each epoch"
    )
    parser.add_argument(
        "--no_eval_each_epoch",
        action="store_false",
        dest="eval_each_epoch",
        help="Skip evaluation after each epoch"
    )
    parser.add_argument(
        "--save_each_epoch",
        action="store_true",
        help="Save model checkpoint after each epoch"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 mixed precision training (CUDA only)"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use BF16 mixed precision training (CUDA only, if supported)"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the final trained model"
    )
    
    # Inference benchmark options
    parser.add_argument(
        "--run_inference_bench",
        action="store_true",
        default=True,
        help="Run inference benchmark after training"
    )
    parser.add_argument(
        "--no_inference_bench",
        action="store_false",
        dest="run_inference_bench",
        help="Skip inference benchmark"
    )
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=1,
        help="Batch size for inference benchmark"
    )
    parser.add_argument(
        "--inference_samples",
        type=int,
        default=50,
        help="Number of samples for inference benchmark (None for all test samples)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate during inference"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./benchmark_results",
        help="Directory for training outputs and CSV results"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log metrics every N steps"
    )
    
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Suppress some warnings for cleaner output
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Run training
    metrics = main(args)
