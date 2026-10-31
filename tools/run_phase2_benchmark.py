#!/usr/bin/env python3
"""Phase 2: Camera and YOLO Inference Optimization Benchmark Suite.

Executes controlled empirical optimization benchmarks across:
1. Supported camera resolutions (640x480 vs 1280x720) and AVFoundation property controls.
2. Preprocessing pipelines (Original vs CLAHE vs Gamma Correction).
3. YOLO inference tuning (Confidence thresholds, IoU thresholds, Image sizes, Max detections).
4. Apple Silicon device comparison (CPU vs MPS latency, warmup, initialization, memory).

Outputs structured CSV benchmark results to reports/phase2_benchmark.csv.
"""

import csv
import logging
import os
import resource
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.webcam import WebcamManager, get_macbook_builtin_camera_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Phase2Benchmark")


# ------------------------------------------------------------------------------
# 1. Image Preprocessing Algorithms
# ------------------------------------------------------------------------------

def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization on Luminance channel."""
    if img is None or img.size == 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def apply_gamma(img: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """Apply Gamma Correction contrast adjustment using a 256-element lookup table."""
    if img is None or img.size == 0 or gamma <= 0.0:
        return img
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def get_rss_mb() -> float:
    """Retrieve current process Resident Set Size (RSS) in megabytes."""
    # On macOS, ru_maxrss is in bytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


# ------------------------------------------------------------------------------
# 2. Camera Resolution & Hardware Property Control Benchmark
# ------------------------------------------------------------------------------

def run_camera_resolution_benchmark() -> List[Dict[str, Any]]:
    """Evaluate camera resolutions and macOS AVFoundation hardware property controllable limits."""
    logger.info("Evaluating camera resolutions and AVFoundation hardware control limits...")
    results = []
    idx = get_macbook_builtin_camera_index()

    test_resolutions = [(640, 480), (1280, 720)]

    for req_w, req_h in test_resolutions:
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            logger.warning("Could not open camera at index %d for %dx%d", idx, req_w, req_h)
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Warmup and convergence
        for _ in range(12):
            cap.read()

        read_times = []
        last_frame = None
        for _ in range(25):
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if ret and frame is not None:
                read_times.append((time.perf_counter() - t0) * 1000.0)
                last_frame = frame

        # Check property write support
        prop_tests = {
            "exposure": cap.set(cv2.CAP_PROP_EXPOSURE, -4.0),
            "brightness": cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5),
            "contrast": cap.set(cv2.CAP_PROP_CONTRAST, 0.5),
            "gain": cap.set(cv2.CAP_PROP_GAIN, 1.0),
            "focus": cap.set(cv2.CAP_PROP_FOCUS, 10.0),
            "fps": cap.set(cv2.CAP_PROP_FPS, 30.0),
        }
        cap.release()

        if last_frame is not None:
            real_h, real_w = last_frame.shape[:2]
            gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            mean_lum = float(np.mean(gray))
            contrast_std = float(np.std(gray))
            med_read = float(np.median(read_times))
            mean_read = float(np.mean(read_times))

            results.append({
                "benchmark_category": "Camera Resolution & Control",
                "test_case": f"{req_w}x{req_h}",
                "requested_resolution": f"{req_w}x{req_h}",
                "actual_resolution": f"{real_w}x{real_h}",
                "aspect_ratio": f"{real_w/real_h:.2f}:1",
                "median_capture_ms": round(med_read, 2),
                "mean_capture_ms": round(mean_read, 2),
                "mean_luminance": round(mean_lum, 2),
                "contrast_std": round(contrast_std, 2),
                "laplacian_variance": round(lap_var, 2),
                "exposure_controllable": prop_tests["exposure"],
                "brightness_controllable": prop_tests["brightness"],
                "gain_controllable": prop_tests["gain"],
                "focus_controllable": prop_tests["focus"],
                "fps_controllable": prop_tests["fps"],
                "details": "AVFoundation driver strictly enforces automatic firmware exposure/gain on macOS."
            })

    return results


# ------------------------------------------------------------------------------
# 3. Preprocessing Experiments Benchmark
# ------------------------------------------------------------------------------

def run_preprocessing_benchmark() -> List[Dict[str, Any]]:
    """Compare detection performance and latency across Original, CLAHE, and Gamma correction."""
    logger.info("Running preprocessing experiments (Original vs CLAHE vs Gamma)...")
    model = YOLO("yolo11n.pt")
    results = []

    test_images = [
        ("bus.jpg", "assets/test_images/bus.jpg", {"bus": 1, "person": 4}),
        ("zidane.jpg", "assets/test_images/zidane.jpg", {"person": 2, "tie": 2}),
        ("grace_hopper.jpg", "assets/test_images/grace_hopper.jpg", {"person": 1, "tie": 1}),
        ("webcam_live.jpg", "assets/test_images/webcam_live.jpg", {}),
    ]

    preprocessing_modes = [
        ("Original (None)", lambda img: (img, 0.0)),
        ("CLAHE (clip=2.0)", lambda img: (apply_clahe(img, 2.0), 0.0)),
        ("Gamma Correction (1.5)", lambda img: (apply_gamma(img, 1.5), 0.0)),
    ]

    for img_name, img_path, gt_dict in test_images:
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            continue

        for mode_name, prep_fn in preprocessing_modes:
            # Measure preprocessing latency over 15 runs
            prep_times = []
            for _ in range(15):
                t0 = time.perf_counter()
                processed_img, _ = prep_fn(raw_img)
                prep_times.append((time.perf_counter() - t0) * 1000.0)
            median_prep_ms = float(np.median(prep_times))

            # Measure inference latency on processed image over 15 runs
            inf_times = []
            boxes = None
            for _ in range(15):
                t0 = time.perf_counter()
                res = model(processed_img, conf=0.25, imgsz=640, device="cpu", verbose=False)
                inf_times.append((time.perf_counter() - t0) * 1000.0)
                boxes = res[0].boxes
            median_inf_ms = float(np.median(inf_times))

            total_pipeline_ms = median_prep_ms + median_inf_ms
            effective_fps = 1000.0 / total_pipeline_ms if total_pipeline_ms > 0 else 0.0

            classes_found = [model.names[int(b.cls[0])] for b in boxes] if boxes else []
            confs_found = [round(float(b.conf[0]), 3) for b in boxes] if boxes else []

            # Calculate Ground Truth TP, FP, FN
            pred_counts: Dict[str, int] = {}
            for c in classes_found:
                pred_counts[c] = pred_counts.get(c, 0) + 1

            total_gt = sum(gt_dict.values())
            tp = 0
            for c, cnt in pred_counts.items():
                gt_c = gt_dict.get(c, 0)
                tp += min(cnt, gt_c)
            fp = max(0, len(classes_found) - tp)
            fn = max(0, total_gt - tp)

            precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 1.0
            recall = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 1.0

            results.append({
                "benchmark_category": "Image Preprocessing",
                "test_case": f"{img_name} [{mode_name}]",
                "image_source": img_name,
                "preprocessing_mode": mode_name,
                "preprocessing_latency_ms": round(median_prep_ms, 2),
                "inference_latency_ms": round(median_inf_ms, 2),
                "total_pipeline_ms": round(total_pipeline_ms, 2),
                "effective_fps": round(effective_fps, 1),
                "detections_count": len(classes_found),
                "detected_classes": "|".join(classes_found),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
            })

    return results


# ------------------------------------------------------------------------------
# 4. YOLO Inference Tuning Benchmark (Conf, IoU, Imgsz, Max_det)
# ------------------------------------------------------------------------------

def run_inference_tuning_benchmark() -> List[Dict[str, Any]]:
    """Evaluate combinations of confidence thresholds, IoU thresholds, image sizes, and max_det."""
    logger.info("Running YOLO inference tuning experiments...")
    model = YOLO("yolo11n.pt")
    results = []

    bus_img = cv2.imread("assets/test_images/bus.jpg")
    zidane_img = cv2.imread("assets/test_images/zidane.jpg")
    gt_bus = {"bus": 1, "person": 4}
    gt_zidane = {"person": 2, "tie": 2}

    test_scenarios = [
        ("bus.jpg", bus_img, gt_bus),
        ("zidane.jpg", zidane_img, gt_zidane),
    ]

    conf_list = [0.15, 0.25, 0.35, 0.45]
    iou_list = [0.45, 0.60, 0.70]
    imgsz_list = [640, 800]
    max_det_list = [20, 50, 300]

    for img_name, img, gt_dict in test_scenarios:
        total_gt = sum(gt_dict.values())

        # 1. Sweep Confidence Thresholds (fixed iou=0.45, imgsz=640, max_det=50)
        for conf in conf_list:
            t0 = time.perf_counter()
            res = model(img, conf=conf, iou=0.45, imgsz=640, max_det=50, device="cpu", verbose=False)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            boxes = res[0].boxes
            classes = [model.names[int(b.cls[0])] for b in boxes] if boxes else []

            pred_counts: Dict[str, int] = {}
            for c in classes:
                pred_counts[c] = pred_counts.get(c, 0) + 1
            tp = sum(min(pred_counts.get(c, 0), gt_dict.get(c, 0)) for c in pred_counts)
            fp = max(0, len(classes) - tp)
            fn = max(0, total_gt - tp)

            results.append({
                "benchmark_category": "Inference Tuning: Confidence Threshold",
                "test_case": f"{img_name} [conf={conf:.2f}]",
                "image_source": img_name,
                "conf_threshold": conf,
                "iou_threshold": 0.45,
                "imgsz": 640,
                "max_det": 50,
                "detections_count": len(classes),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(tp / (tp + fp), 3) if (tp + fp) > 0 else 1.0,
                "recall": round(tp / (tp + fn), 3) if (tp + fn) > 0 else 1.0,
                "latency_ms": round(lat_ms, 2),
                "classes_found": "|".join(classes)
            })

        # 2. Sweep IoU Thresholds (fixed conf=0.25, imgsz=640, max_det=50)
        for iou in iou_list:
            t0 = time.perf_counter()
            res = model(img, conf=0.25, iou=iou, imgsz=640, max_det=50, device="cpu", verbose=False)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            boxes = res[0].boxes
            classes = [model.names[int(b.cls[0])] for b in boxes] if boxes else []

            pred_counts = {}
            for c in classes:
                pred_counts[c] = pred_counts.get(c, 0) + 1
            tp = sum(min(pred_counts.get(c, 0), gt_dict.get(c, 0)) for c in pred_counts)
            fp = max(0, len(classes) - tp)
            fn = max(0, total_gt - tp)

            results.append({
                "benchmark_category": "Inference Tuning: IoU Threshold",
                "test_case": f"{img_name} [iou={iou:.2f}]",
                "image_source": img_name,
                "conf_threshold": 0.25,
                "iou_threshold": iou,
                "imgsz": 640,
                "max_det": 50,
                "detections_count": len(classes),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(tp / (tp + fp), 3) if (tp + fp) > 0 else 1.0,
                "recall": round(tp / (tp + fn), 3) if (tp + fn) > 0 else 1.0,
                "latency_ms": round(lat_ms, 2),
                "classes_found": "|".join(classes)
            })

        # 3. Sweep Max Detections (fixed conf=0.25, iou=0.45, imgsz=640)
        for max_det in max_det_list:
            t0 = time.perf_counter()
            res = model(img, conf=0.25, iou=0.45, imgsz=640, max_det=max_det, device="cpu", verbose=False)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            boxes = res[0].boxes
            classes = [model.names[int(b.cls[0])] for b in boxes] if boxes else []

            results.append({
                "benchmark_category": "Inference Tuning: Max Detections",
                "test_case": f"{img_name} [max_det={max_det}]",
                "image_source": img_name,
                "conf_threshold": 0.25,
                "iou_threshold": 0.45,
                "imgsz": 640,
                "max_det": max_det,
                "detections_count": len(classes),
                "latency_ms": round(lat_ms, 2),
                "classes_found": "|".join(classes)
            })

    return results


# ------------------------------------------------------------------------------
# 5. Apple Silicon Hardware Acceleration Benchmark (CPU vs MPS)
# ------------------------------------------------------------------------------

def run_apple_silicon_benchmark() -> List[Dict[str, Any]]:
    """Compare CPU vs MPS on model initialization, warmup, steady-state latency, and memory."""
    logger.info("Running Apple Silicon hardware acceleration benchmark (CPU vs MPS)...")
    results = []
    img = cv2.imread("assets/test_images/bus.jpg")

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    for dev in devices:
        rss_before = get_rss_mb()
        t_init0 = time.perf_counter()
        model = YOLO("yolo11n.pt")
        init_ms = (time.perf_counter() - t_init0) * 1000.0

        # Warmup (first 3 passes)
        warmup_passes = []
        for _ in range(3):
            tw0 = time.perf_counter()
            _ = model(img, device=dev, imgsz=640, verbose=False)
            if dev == "mps":
                torch.mps.synchronize()
            warmup_passes.append((time.perf_counter() - tw0) * 1000.0)

        # Steady-state inference (25 passes)
        steady_latencies = []
        for _ in range(25):
            ts0 = time.perf_counter()
            _ = model(img, device=dev, imgsz=640, verbose=False)
            if dev == "mps":
                torch.mps.synchronize()
            steady_latencies.append((time.perf_counter() - ts0) * 1000.0)

        rss_after = get_rss_mb()
        mps_mem_mb = 0.0
        if dev == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            mps_mem_mb = torch.mps.current_allocated_memory() / (1024.0 * 1024.0)

        med_lat = float(np.median(steady_latencies))
        mean_lat = float(np.mean(steady_latencies))
        fps = 1000.0 / med_lat if med_lat > 0 else 0.0

        results.append({
            "benchmark_category": "Apple Silicon Hardware Comparison",
            "test_case": f"Device: {dev.upper()}",
            "device": dev.upper(),
            "model_init_time_ms": round(init_ms, 2),
            "warmup_pass_1_ms": round(warmup_passes[0], 2),
            "warmup_pass_2_ms": round(warmup_passes[1], 2),
            "warmup_pass_3_ms": round(warmup_passes[2], 2),
            "steady_state_median_ms": round(med_lat, 2),
            "steady_state_mean_ms": round(mean_lat, 2),
            "steady_state_min_ms": round(min(steady_latencies), 2),
            "steady_state_max_ms": round(max(steady_latencies), 2),
            "effective_fps": round(fps, 1),
            "process_rss_mb": round(rss_after, 1),
            "process_rss_delta_mb": round(rss_after - rss_before, 1),
            "torch_mps_allocated_mb": round(mps_mem_mb, 2)
        })

    return results


def main() -> None:
    """Execute all Phase 2 optimization benchmarks and export consolidated CSV."""
    print("=" * 76)
    print("PHASE 2: CAMERA AND YOLO INFERENCE OPTIMIZATION BENCHMARK")
    print("=" * 76)

    os.makedirs("reports", exist_ok=True)
    os.makedirs("benchmarks/results", exist_ok=True)

    cam_res = run_camera_resolution_benchmark()
    prep_res = run_preprocessing_benchmark()
    tune_res = run_inference_tuning_benchmark()
    silicon_res = run_apple_silicon_benchmark()

    # Combine all results into reports/phase2_benchmark.csv
    consolidated: List[Dict[str, Any]] = []
    consolidated.extend(cam_res)
    consolidated.extend(prep_res)
    consolidated.extend(tune_res)
    consolidated.extend(silicon_res)

    out_csv = "reports/phase2_benchmark.csv"
    if consolidated:
        # Determine all unique fieldnames
        keys = []
        for row in consolidated:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(consolidated)

        logger.info("Saved consolidated Phase 2 benchmark to %s (%d total records)", out_csv, len(consolidated))

    print(f"\nPhase 2 Benchmark complete! Exported to: {out_csv}")


if __name__ == "__main__":
    main()
