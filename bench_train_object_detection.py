#!/usr/bin/env python3
"""
Object Detection Training Script with Timing Metrics

This script trains an RT-DETRv2 object detection model on the Trashify dataset
and outputs training/evaluation time metrics to a CSV file.

Based on: https://www.learnhuggingface.com/notebooks/hugging_face_object_detection_tutorial

Usage:
    Run on your local machine (requires GPU for efficient training):
    
    python train_object_detection.py --num_epochs 10 --batch_size 8 --output_dir ./benchmark_results
    
    For a quick test run:
    python train_object_detection.py --num_epochs 1 --batch_size 4 --output_dir ./benchmark_results

Required packages:
    pip install torch torchvision transformers datasets accelerate torchmetrics pycocotools tqdm
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
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
)


# ============================================================================
# Configuration and Data Classes
# ============================================================================

@dataclass
class COCOAnnotation:
    """Represents a single COCO format annotation."""
    image_id: int
    category_id: int
    bbox: List[float]  # [x, y, width, height] in absolute pixels
    area: float
    iscrowd: int = 0


@dataclass
class COCOImageAnnotations:
    """Represents all annotations for a single image in COCO format."""
    image_id: int
    annotations: List[COCOAnnotation]


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


# ============================================================================
# Dataset Configuration
# ============================================================================

# Label mappings for the Trashify dataset
CATEGORIES = ['bin', 'hand', 'not_bin', 'not_hand', 'not_trash', 'trash', 'trash_arm']
ID2LABEL = {i: class_name for i, class_name in enumerate(CATEGORIES)}
LABEL2ID = {value: key for key, value in ID2LABEL.items()}


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


def get_device_info() -> tuple[torch.device, str, str]:
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


# ============================================================================
# Data Preprocessing Functions
# ============================================================================

def format_annotations_as_coco(
    image_id: int,
    category_ids: List[int],
    bboxes: List[List[float]],
    areas: List[float],
    iscrowds: Optional[List[int]] = None
) -> COCOImageAnnotations:
    """
    Formats annotations into COCO format.
    
    Args:
        image_id: Unique identifier for the image
        category_ids: List of category IDs for each annotation
        bboxes: List of bounding boxes in [x, y, width, height] format
        areas: List of areas for each bounding box
        iscrowds: Optional list of iscrowd flags (defaults to 0)
    
    Returns:
        COCOImageAnnotations object containing all annotations for the image
    """
    if iscrowds is None:
        iscrowds = [0] * len(category_ids)
    
    annotations = []
    for cat_id, bbox, area, iscrowd in zip(category_ids, bboxes, areas, iscrowds):
        annotation = COCOAnnotation(
            image_id=image_id,
            category_id=cat_id,
            bbox=bbox,
            area=area,
            iscrowd=iscrowd
        )
        annotations.append(annotation)
    
    return COCOImageAnnotations(image_id=image_id, annotations=annotations)


def coco_annotations_to_dict(coco_annotations: COCOImageAnnotations) -> Dict[str, Any]:
    """Converts COCOImageAnnotations to dictionary format expected by the processor."""
    return {
        "image_id": coco_annotations.image_id,
        "annotations": [
            {
                "image_id": ann.image_id,
                "category_id": ann.category_id,
                "bbox": ann.bbox,
                "area": ann.area,
                "iscrowd": ann.iscrowd
            }
            for ann in coco_annotations.annotations
        ]
    }


def create_preprocessing_function(processor: AutoImageProcessor):
    """
    Creates a preprocessing function for the dataset.
    
    Args:
        processor: The image processor for the model
    
    Returns:
        A function that preprocesses batches of samples
    """
    def preprocess_batch(batch):
        """Preprocesses a batch of samples for training."""
        images = batch["image"]
        
        # Prepare annotations in COCO format
        annotations_list = []
        for i in range(len(images)):
            image_id = batch["image_id"][i]
            category_ids = batch["annotations"][i]["category_id"]
            bboxes = batch["annotations"][i]["bbox"]
            areas = batch["annotations"][i]["area"]
            iscrowds = batch["annotations"][i].get("iscrowd", [0] * len(category_ids))
            
            coco_annotations = format_annotations_as_coco(
                image_id=image_id,
                category_ids=category_ids,
                bboxes=bboxes,
                areas=areas,
                iscrowds=iscrowds
            )
            annotations_list.append(coco_annotations_to_dict(coco_annotations))
        
        # Process images and annotations
        processed = processor(
            images=images,
            annotations=annotations_list,
            return_tensors="pt"
        )
        
        return processed
    
    return preprocess_batch


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collation function for object detection batches.
    
    Args:
        batch: List of preprocessed samples
    
    Returns:
        Collated batch dictionary
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }


# ============================================================================
# Model Creation
# ============================================================================

def create_model(
    model_name: str = "PekingU/rtdetr_v2_r50vd",
    label2id: Dict[str, int] = LABEL2ID,
    id2label: Dict[int, str] = ID2LABEL
) -> AutoModelForObjectDetection:
    """
    Creates and returns an object detection model.
    
    Args:
        model_name: The name or path of the pretrained model
        label2id: Dictionary mapping labels to IDs
        id2label: Dictionary mapping IDs to labels
    
    Returns:
        Configured AutoModelForObjectDetection instance
    """
    model = AutoModelForObjectDetection.from_pretrained(
        pretrained_model_name_or_path=model_name,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    return model


# ============================================================================
# Custom Trainer with Timing
# ============================================================================

class TimedTrainer(Trainer):
    """Trainer subclass that tracks timing metrics."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timing_metrics: List[TimingMetrics] = []
        self.epoch_start_time: Optional[float] = None
        self.current_epoch: int = 0
    
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
            if state.log_history:
                for log in reversed(state.log_history):
                    if "loss" in log:
                        train_loss = log["loss"]
                        break
            
            metrics = TimingMetrics(
                epoch=self.current_epoch + 1,
                train_time_seconds=epoch_time,
                train_loss=train_loss
            )
            self.timing_metrics.append(metrics)
            self.current_epoch += 1
            self.epoch_start_time = None
        
        return super().on_epoch_end(args, state, control, **kwargs)


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_model(
    model: AutoModelForObjectDetection,
    eval_dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluates the model and returns timing metrics.
    
    Args:
        model: The trained model
        eval_dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
    
    Returns:
        Dictionary containing evaluation metrics and timing
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating", unit="batch"):
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in label.items()} for label in batch["labels"]]
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1
    
    eval_time = time.time() - start_time
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        "eval_time_seconds": eval_time,
        "eval_loss": avg_loss,
        "num_samples": len(eval_dataloader.dataset) if hasattr(eval_dataloader, 'dataset') else num_batches
    }


# ============================================================================
# CSV Output Functions
# ============================================================================

def save_metrics_to_csv(
    metrics: List[TimingMetrics],
    output_path: str,
    total_train_time: float,
    device_info: str,
    device_sanitized: str,
    args: argparse.Namespace
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
    """
    # Calculate summary statistics
    train_times = [m.train_time_seconds for m in metrics if m.train_time_seconds]
    eval_times = [m.eval_time_seconds for m in metrics if m.eval_time_seconds]
    train_losses = [m.train_loss for m in metrics if m.train_loss is not None]
    eval_losses = [m.eval_loss for m in metrics if m.eval_loss is not None]
    
    # Calculate samples per second if we have the data
    total_samples = args.batch_size * args.num_epochs * (1 - args.test_split)
    samples_per_second = total_samples / total_train_time if total_train_time > 0 else 0
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["metric", "value", "unit"])
        
        # Device and model info
        writer.writerow(["device", device_info, ""])
        writer.writerow(["device_sanitized", device_sanitized, ""])
        writer.writerow(["model_name", args.model_name, ""])
        writer.writerow(["dataset_name", args.dataset_name, ""])
        writer.writerow(["timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"), ""])
        writer.writerow(["test_type", "object_detection_training", ""])
        
        # Training configuration
        writer.writerow(["num_epochs", args.num_epochs, ""])
        writer.writerow(["batch_size", args.batch_size, ""])
        writer.writerow(["learning_rate", args.learning_rate, ""])
        writer.writerow(["weight_decay", args.weight_decay, ""])
        writer.writerow(["warmup_ratio", args.warmup_ratio, ""])
        writer.writerow(["image_size", args.image_size, "pixels"])
        writer.writerow(["test_split", args.test_split, ""])
        writer.writerow(["fp16", args.fp16, ""])
        writer.writerow(["seed", args.seed, ""])
        
        # Timing metrics
        writer.writerow(["total_train_time", f"{total_train_time:.2f}", "seconds"])
        writer.writerow(["total_train_time_minutes", f"{total_train_time / 60:.2f}", "minutes"])
        writer.writerow(["samples_per_second", f"{samples_per_second:.2f}", "samples/s"])
        
        # Per-epoch training time stats
        if train_times:
            writer.writerow(["avg_epoch_train_time", f"{np.mean(train_times):.2f}", "seconds"])
            writer.writerow(["min_epoch_train_time", f"{np.min(train_times):.2f}", "seconds"])
            writer.writerow(["max_epoch_train_time", f"{np.max(train_times):.2f}", "seconds"])
            writer.writerow(["std_epoch_train_time", f"{np.std(train_times):.2f}", "seconds"])
        
        # Per-epoch eval time stats
        if eval_times:
            writer.writerow(["avg_epoch_eval_time", f"{np.mean(eval_times):.2f}", "seconds"])
            writer.writerow(["min_epoch_eval_time", f"{np.min(eval_times):.2f}", "seconds"])
            writer.writerow(["max_epoch_eval_time", f"{np.max(eval_times):.2f}", "seconds"])
        
        # Loss metrics
        if train_losses:
            writer.writerow(["initial_train_loss", f"{train_losses[0]:.4f}", ""])
            writer.writerow(["final_train_loss", f"{train_losses[-1]:.4f}", ""])
            writer.writerow(["min_train_loss", f"{np.min(train_losses):.4f}", ""])
        
        if eval_losses:
            writer.writerow(["final_eval_loss", f"{eval_losses[-1]:.4f}", ""])
            writer.writerow(["min_eval_loss", f"{np.min(eval_losses):.4f}", ""])
        
        # Per-epoch details
        for i, metric in enumerate(metrics):
            epoch_num = i + 1
            writer.writerow([f"epoch_{epoch_num}_train_time", f"{metric.train_time_seconds:.2f}", "seconds"])
            if metric.train_loss is not None:
                writer.writerow([f"epoch_{epoch_num}_train_loss", f"{metric.train_loss:.4f}", ""])
            if metric.eval_time_seconds is not None:
                writer.writerow([f"epoch_{epoch_num}_eval_time", f"{metric.eval_time_seconds:.2f}", "seconds"])
            if metric.eval_loss is not None:
                writer.writerow([f"epoch_{epoch_num}_eval_loss", f"{metric.eval_loss:.4f}", ""])
    
    print(f"\nCSV saved to: {output_path}")


def print_summary(
    metrics: List[TimingMetrics],
    total_train_time: float,
    device_info: str
):
    """Prints a summary of training metrics to console."""
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Device: {device_info}")
    print(f"Total Training Time: {total_train_time:.2f} seconds ({total_train_time / 60:.2f} minutes)")
    print(f"Number of Epochs: {len(metrics)}")
    
    train_times = [m.train_time_seconds for m in metrics if m.train_time_seconds]
    if train_times:
        print(f"\nPer-Epoch Training Time:")
        print(f"  Average: {np.mean(train_times):.2f} seconds")
        print(f"  Min: {np.min(train_times):.2f} seconds")
        print(f"  Max: {np.max(train_times):.2f} seconds")
    
    eval_times = [m.eval_time_seconds for m in metrics if m.eval_time_seconds]
    if eval_times:
        print(f"\nPer-Epoch Evaluation Time:")
        print(f"  Average: {np.mean(eval_times):.2f} seconds")
        print(f"  Min: {np.min(eval_times):.2f} seconds")
        print(f"  Max: {np.max(eval_times):.2f} seconds")
    
    train_losses = [m.train_loss for m in metrics if m.train_loss is not None]
    if train_losses:
        print(f"\nTraining Loss:")
        print(f"  Initial: {train_losses[0]:.4f}")
        print(f"  Final: {train_losses[-1]:.4f}")
    
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
    print("OBJECT DETECTION TRAINING SCRIPT")
    print(f"{'=' * 60}")
    print(f"Device: {device_info}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Image Size: {args.image_size}")
    print(f"Seed: {args.seed}")
    print(f"{'=' * 60}\n")
    
    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    print("Loading dataset...")
    dataset = load_dataset(path=args.dataset_name)
    
    # Split dataset into train and test
    dataset_split = dataset["train"].train_test_split(
        test_size=args.test_split,
        seed=args.seed
    )
    train_dataset = dataset_split["train"]
    test_dataset = dataset_split["test"]
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # -------------------------------------------------------------------------
    # Load processor and model
    # -------------------------------------------------------------------------
    print("\nLoading processor and model...")
    processor = AutoImageProcessor.from_pretrained(
        args.model_name,
        do_resize=True,
        size={"height": args.image_size, "width": args.image_size}
    )
    
    model = create_model(
        model_name=args.model_name,
        label2id=LABEL2ID,
        id2label=ID2LABEL
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # -------------------------------------------------------------------------
    # Preprocess datasets
    # -------------------------------------------------------------------------
    print("\nPreprocessing datasets...")
    preprocess_fn = create_preprocessing_function(processor)
    
    # Apply preprocessing with caching disabled for memory efficiency
    train_dataset_processed = train_dataset.map(
        preprocess_fn,
        batched=True,
        batch_size=args.preprocessing_batch_size,
        remove_columns=train_dataset.column_names
    )
    
    test_dataset_processed = test_dataset.map(
        preprocess_fn,
        batched=True,
        batch_size=args.preprocessing_batch_size,
        remove_columns=test_dataset.column_names
    )
    
    # Set format to PyTorch tensors
    train_dataset_processed.set_format("torch")
    test_dataset_processed.set_format("torch")
    
    # -------------------------------------------------------------------------
    # Setup output directory
    # -------------------------------------------------------------------------
    output_dir = Path(args.output_dir) / device_sanitized
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    model_short = re.sub(r'[^a-zA-Z0-9]', '_', args.model_name.split('/')[-1])
    model_short = re.sub(r'_+', '_', model_short).strip('_')[:50]
    csv_filename = f"object_detection_bench_{device_sanitized}_{model_short}_epochs_{args.num_epochs}_{timestamp}.csv"
    csv_path = output_dir / csv_filename
    
    # -------------------------------------------------------------------------
    # Setup training arguments
    # -------------------------------------------------------------------------
    print("\nSetting up training...")
    
    training_output_dir = output_dir / "training_output"
    training_output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(training_output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="epoch" if args.eval_each_epoch else "no",
        save_strategy="epoch" if args.save_each_epoch else "no",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        fp16=args.fp16 and torch.cuda.is_available(),
        report_to="none",  # Disable wandb/tensorboard for cleaner output
        load_best_model_at_end=False,
    )
    
    # -------------------------------------------------------------------------
    # Create trainer and train
    # -------------------------------------------------------------------------
    trainer = TimedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_processed,
        eval_dataset=test_dataset_processed if args.eval_each_epoch else None,
        data_collator=collate_fn,
    )
    
    print("\nStarting training...")
    print("-" * 40)
    
    train_result = trainer.train()
    
    print("-" * 40)
    print("Training complete!")
    
    # -------------------------------------------------------------------------
    # Final evaluation
    # -------------------------------------------------------------------------
    if args.final_eval:
        print("\nRunning final evaluation...")
        eval_dataloader = DataLoader(
            test_dataset_processed,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=args.num_workers
        )
        
        eval_metrics = evaluate_model(model, eval_dataloader, device)
        
        # Update last timing metric with eval info
        if trainer.timing_metrics:
            trainer.timing_metrics[-1].eval_time_seconds = eval_metrics["eval_time_seconds"]
            trainer.timing_metrics[-1].eval_loss = eval_metrics["eval_loss"]
        
        print(f"Final Evaluation Loss: {eval_metrics['eval_loss']:.4f}")
        print(f"Evaluation Time: {eval_metrics['eval_time_seconds']:.2f} seconds")
    
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
        args=args
    )
    
    # Print summary
    print_summary(
        metrics=trainer.timing_metrics,
        total_train_time=trainer.total_train_time,
        device_info=device_info
    )
    
    # -------------------------------------------------------------------------
    # Save model (optional)
    # -------------------------------------------------------------------------
    if args.save_model:
        model_save_path = output_dir / "final_model"
        print(f"\nSaving model to: {model_save_path}")
        trainer.save_model(str(model_save_path))
        processor.save_pretrained(str(model_save_path))
    
    return trainer.timing_metrics


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train an object detection model and measure training time metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and dataset
    parser.add_argument(
        "--model_name",
        type=str,
        default="PekingU/rtdetr_v2_r50vd",
        help="Name or path of the pretrained model"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mrdbourke/trashify_manual_labelled_images",
        help="Name of the Hugging Face dataset"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
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
        default=1e-4,
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
        "--image_size",
        type=int,
        default=640,
        help="Input image size (height and width)"
    )
    
    # Data processing
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing"
    )
    parser.add_argument(
        "--preprocessing_batch_size",
        type=int,
        default=16,
        help="Batch size for dataset preprocessing"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    
    # Training options
    parser.add_argument(
        "--eval_each_epoch",
        action="store_true",
        help="Run evaluation after each epoch"
    )
    parser.add_argument(
        "--save_each_epoch",
        action="store_true",
        help="Save model checkpoint after each epoch"
    )
    parser.add_argument(
        "--final_eval",
        action="store_true",
        default=True,
        help="Run final evaluation after training"
    )
    parser.add_argument(
        "--no_final_eval",
        action="store_false",
        dest="final_eval",
        help="Skip final evaluation after training"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 mixed precision training (CUDA only)"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the final trained model"
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