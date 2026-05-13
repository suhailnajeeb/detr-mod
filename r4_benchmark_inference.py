#!/usr/bin/env python
"""R4 inference benchmark adapter for the RDD DETR fork."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision import transforms as TVT

from main import get_args_parser as get_model_args_parser
from models import build_model
from util.misc import nested_tensor_from_tensor_list


MODEL_NAME = "DETR"
MODEL_SLUG = "detr"
MANUSCRIPT_ROLE = "DETR-family comparison row"
REPO_PATH = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = REPO_PATH / "outputs/holdout_01/checkpoint.pth"
DEFAULT_COCO_ROOT = Path("/data/gpfs/projects/punim1800/RDD-2022/holdout/coco_holdout")
DEFAULT_ARTIFACT_ROOT = Path("/data/gpfs/projects/punim1800/rdd/output/r4_i4_inference_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R4 DETR inference benchmark")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--coco-root", default=str(DEFAULT_COCO_ROOT))
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--run-mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--timed-images", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-classes", type=int, default=5)
    return parser.parse_args()


def get_driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip().splitlines()[0]
    except Exception:
        return None


def build_loaded_model(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, dict]:
    model_parser = argparse.ArgumentParser(parents=[get_model_args_parser()])
    model_args = model_parser.parse_args(
        [
            "--dataset_file",
            "coco",
            "--num_classes",
            str(args.num_classes),
            "--device",
            str(device),
        ]
    )
    model, _, _ = build_model(model_args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    load_result = model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return model, {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "missing_keys": list(getattr(load_result, "missing_keys", [])),
        "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
    }


def load_coco_images(coco_root: Path, limit: int) -> list[dict]:
    ann_path = coco_root / "annotations" / "instances_test2017.json"
    with ann_path.open("r") as f:
        annotations = json.load(f)
    images = annotations["images"][:limit]
    if len(images) < limit:
        raise RuntimeError(f"Requested {limit} images but only found {len(images)} in {ann_path}")
    return images


def preprocess_image(image_path: Path, image_size: int, device: torch.device):
    transform = TVT.Compose(
        [
            TVT.Resize((image_size, image_size)),
            TVT.ToTensor(),
            TVT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB"))
    return nested_tensor_from_tensor_list([tensor]).to(device)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[int(idx)]
    return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)


def summarize(latencies: list[float]) -> dict:
    return {
        "latency_mean_ms": statistics.fmean(latencies),
        "latency_std_ms": statistics.pstdev(latencies) if len(latencies) > 1 else 0.0,
        "latency_median_ms": statistics.median(latencies),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_min_ms": min(latencies),
        "latency_max_ms": max(latencies),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_metadata(args: argparse.Namespace, device: torch.device, model: torch.nn.Module | None) -> dict:
    gpu_name = None
    gpu_memory_gb = None
    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        gpu_name = props.name
        gpu_memory_gb = round(props.total_memory / (1024**3), 2)
    total_params = None
    trainable_params = None
    if model is not None:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "model": MODEL_NAME,
        "model_slug": MODEL_SLUG,
        "manuscript_role": MANUSCRIPT_ROLE,
        "repository_path": str(REPO_PATH),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": "main.get_args_parser with --dataset_file coco --num_classes 5",
        "framework": "PyTorch DETR fork",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "driver_version": get_driver_version(),
        "hostname": socket.gethostname(),
        "gpu_node_type": "deeplearn",
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
        "parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        "input_size": f"{args.image_size}x{args.image_size}",
        "batch_size": 1,
        "precision": "FP32",
        "preprocess_included": False,
        "postprocess_included": False,
        "run_mode": args.run_mode,
    }


def run(args: argparse.Namespace) -> int:
    artifact_dir = Path(args.artifact_root) / MODEL_SLUG
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "command.sh").write_text(" ".join([sys.executable, *sys.argv]) + "\n")

    timed_images = args.timed_images
    warmup_iters = args.warmup_iters
    if args.run_mode == "smoke":
        timed_images = timed_images if timed_images is not None else 2
        warmup_iters = warmup_iters if warmup_iters is not None else 2
    else:
        timed_images = timed_images if timed_images is not None else 500
        warmup_iters = warmup_iters if warmup_iters is not None else 50

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    device = torch.device(args.device)
    model = None
    try:
        model, load_info = build_loaded_model(args, device)
        metadata = collect_metadata(args, device, model)
        metadata.update(
            {
                "warmup_iters": warmup_iters,
                "timed_images": timed_images,
                "status": "running",
                "notes": "Smoke mode is non-paper-facing" if args.run_mode == "smoke" else "",
                **load_info,
            }
        )

        images = load_coco_images(Path(args.coco_root), timed_images)
        if device.type == "cuda":
            torch.cuda.synchronize()

        for idx in range(warmup_iters):
            image_info = images[idx % len(images)]
            image_path = Path(args.coco_root) / "test2017" / image_info["file_name"]
            samples = preprocess_image(image_path, args.image_size, device)
            with torch.no_grad():
                _ = model(samples)
        if device.type == "cuda":
            torch.cuda.synchronize()

        raw_rows = []
        latencies = []
        for idx, image_info in enumerate(images):
            image_path = Path(args.coco_root) / "test2017" / image_info["file_name"]
            samples = preprocess_image(image_path, args.image_size, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(samples)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(latency_ms)
            raw_rows.append(
                {
                    "model": MODEL_NAME,
                    "model_slug": MODEL_SLUG,
                    "image_index": idx,
                    "image_id": image_info["id"],
                    "file_name": image_info["file_name"],
                    "latency_ms": latency_ms,
                    "run_mode": args.run_mode,
                    "status": "ok",
                }
            )

        summary = summarize(latencies)
        metadata.update({"status": "ok"})
        summary_row = {
            "model": MODEL_NAME,
            "manuscript_role": MANUSCRIPT_ROLE,
            "gpu_node_type": metadata["gpu_node_type"],
            "gpu_name": metadata["gpu_name"],
            "gpu_memory_gb": metadata["gpu_memory_gb"],
            "hostname": metadata["hostname"],
            "cuda_version": metadata["cuda_version"],
            "driver_version": metadata["driver_version"],
            "framework": metadata["framework"],
            "checkpoint": metadata["checkpoint"],
            "config": metadata["config"],
            "input_size": metadata["input_size"],
            "batch_size": metadata["batch_size"],
            "precision": metadata["precision"],
            "warmup_iters": warmup_iters,
            "timed_images": timed_images,
            **summary,
            "postprocess_included": metadata["postprocess_included"],
            "preprocess_included": metadata["preprocess_included"],
            "status": "ok",
            "notes": metadata["notes"],
        }

        raw_fields = ["model", "model_slug", "image_index", "image_id", "file_name", "latency_ms", "run_mode", "status"]
        summary_fields = list(summary_row.keys())
        write_csv(artifact_dir / "raw_timings.csv", raw_rows, raw_fields)
        write_csv(artifact_dir / "summary.csv", [summary_row], summary_fields)
        (artifact_dir / "summary.json").write_text(json.dumps(summary_row, indent=2) + "\n")
        (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return 0
    except Exception:
        metadata = collect_metadata(args, device if "device" in locals() else torch.device("cpu"), model)
        metadata.update({"status": "failed", "notes": "See failure.log"})
        (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (artifact_dir / "failure.log").write_text(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
