#!/usr/bin/env python3
"""Phase 1: YOLO Object Detection Diagnosis and Baseline Benchmark Suite.

Executes controlled empirical CV benchmarks to diagnose detection limitations,
characterize MacBook camera image quality, and measure inference latency across
configurations (confidence thresholds, image sizes, and devices).

Outputs structured CSV benchmark results to benchmarks/results/.
"""

import csv
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.object_detector import YOLOObjectDetector
from perception.webcam import WebcamManager, get_macbook_builtin_camera_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Phase1Benchmark")


def measure_image_quality(img: np.ndarray, source_name: str) -> Dict[str, Any]:
    """Calculate objective image quality metrics (luminance, contrast, sharpness)."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    mean_lum = float(np.mean(gray))
    std_contrast = float(np.std(gray))
    min_val = int(np.min(gray))
    max_val = int(np.max(gray))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    b_mean = float(np.mean(img[:, :, 0])) if len(img.shape) == 3 else mean_lum
    g_mean = float(np.mean(img[:, :, 1])) if len(img.shape) == 3 else mean_lum
    r_mean = float(np.mean(img[:, :, 2])) if len(img.shape) == 3 else mean_lum

    if mean_lum < 15.0:
        lighting_eval = "Severely Under-exposed (Near-Black Noise Floor)"
    elif mean_lum < 40.0:
        lighting_eval = "Under-exposed (Low Light)"
    elif mean_lum < 180.0:
        lighting_eval = "Normal (Good Illumination)"
    else:
        lighting_eval = "Over-exposed / Saturated"

    if lap_var < 50.0:
        sharpness_eval = "Severe Blur / Featureless"
    elif lap_var < 200.0:
        sharpness_eval = "Moderate Blur / Soft"
    else:
        sharpness_eval = "Sharp / High Texture"

    return {
        "source": source_name,
        "resolution": f"{w}x{h}",
        "aspect_ratio": f"{w/h:.2f}:1",
        "mean_luminance": round(mean_lum, 2),
        "contrast_std": round(std_contrast, 2),
        "min_val": min_val,
        "max_val": max_val,
        "blue_mean": round(b_mean, 2),
        "green_mean": round(g_mean, 2),
        "red_mean": round(r_mean, 2),
        "laplacian_variance": round(lap_var, 2),
        "lighting_evaluation": lighting_eval,
        "sharpness_evaluation": sharpness_eval,
    }


def run_camera_diagnostics(output_csv: str) -> List[Dict[str, Any]]:
    """Capture frames from MacBook webcam and evaluate against reference test images."""
    logger.info("Evaluating camera image quality and reference images...")
    results = []

    # 1. Evaluate local test images
    test_dir = "assets/test_images"
    image_files = [
        "webcam_dark_baseline.jpg",
        "webcam_live.jpg",
        "bus.jpg",
        "zidane.jpg",
        "grace_hopper.jpg",
    ]

    for fname in image_files:
        fpath = os.path.join(test_dir, fname)
        if os.path.exists(fpath):
            img = cv2.imread(fpath)
            if img is not None:
                results.append(measure_image_quality(img, fname))

    # 2. Capture a fresh live frame directly from WebcamManager
    webcam = WebcamManager()
    opened = webcam.start_preview()
    if opened:
        time.sleep(1.2)  # Allow sensor exposure to converge
        live_frame = None
        for _ in range(30):
            live_frame = webcam.get_current_frame()
            if live_frame is not None:
                break
            time.sleep(0.04)
        webcam.stop_preview()

        if live_frame is not None:
            metrics = measure_image_quality(live_frame, "live_camera_direct_capture")
            results.append(metrics)
            # Save fresh diagnostic frame
            cv2.imwrite("assets/test_images/webcam_live_fresh.jpg", live_frame)
    else:
        logger.warning("Could not open live camera for diagnostics.")

    # Write CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info("Saved camera diagnostics to %s (%d records)", output_csv, len(results))

    return results


def run_configuration_experiments(output_csv: str) -> List[Dict[str, Any]]:
    """Compare confidence thresholds, image sizes, and devices across images."""
    logger.info("Running controlled configuration experiments...")
    model_name = "yolo11n.pt"
    model = YOLO(model_name)

    conf_thresholds = [0.15, 0.25, 0.35, 0.45]
    imgsz_list = [640, 800]
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    test_images = [
        ("bus.jpg", "assets/test_images/bus.jpg"),
        ("zidane.jpg", "assets/test_images/zidane.jpg"),
        ("webcam_dark_baseline.jpg", "assets/test_images/webcam_dark_baseline.jpg"),
        ("webcam_live.jpg", "assets/test_images/webcam_live.jpg"),
    ]

    results = []

    for img_label, img_path in test_images:
        if not os.path.exists(img_path):
            continue
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        for dev in devices:
            for sz in imgsz_list:
                # Warmup run for this device and size
                _ = model(frame, imgsz=sz, device=dev, conf=0.25, verbose=False)
                if dev == "mps":
                    torch.mps.synchronize()

                for conf in conf_thresholds:
                    # Timed inference iterations
                    latencies = []
                    boxes = None
                    for _ in range(10):
                        t0 = time.perf_counter()
                        res = model(frame, imgsz=sz, device=dev, conf=conf, verbose=False)
                        if dev == "mps":
                            torch.mps.synchronize()
                        latencies.append((time.perf_counter() - t0) * 1000.0)
                        boxes = res[0].boxes

                    median_lat = float(np.median(latencies))
                    mean_lat = float(np.mean(latencies))
                    min_lat = float(np.min(latencies))
                    max_lat = float(np.max(latencies))
                    fps = 1000.0 / median_lat if median_lat > 0 else 0.0

                    det_count = len(boxes) if boxes is not None else 0
                    classes_found = []
                    conf_scores = []
                    if boxes is not None and det_count > 0:
                        classes_found = [model.names[int(b.cls[0])] for b in boxes]
                        conf_scores = [round(float(b.conf[0]), 3) for b in boxes]

                    mean_conf = float(np.mean(conf_scores)) if conf_scores else 0.0

                    results.append({
                        "image_source": img_label,
                        "device": dev,
                        "imgsz": sz,
                        "conf_threshold": conf,
                        "detections_count": det_count,
                        "detected_classes": "|".join(classes_found),
                        "mean_confidence": round(mean_conf, 3),
                        "median_latency_ms": round(median_lat, 2),
                        "mean_latency_ms": round(mean_lat, 2),
                        "min_latency_ms": round(min_lat, 2),
                        "max_latency_ms": round(max_lat, 2),
                        "effective_fps": round(fps, 1),
                    })
                    logger.debug(
                        "Config: img=%s dev=%s sz=%d conf=%.2f -> %d dets in %.1f ms",
                        img_label, dev, sz, conf, det_count, median_lat
                    )

    # Save to CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info("Saved configuration experiments to %s (%d records)", output_csv, len(results))

    return results


def run_ground_truth_benchmark(output_csv: str) -> List[Dict[str, Any]]:
    """Evaluate detection metrics against known, verified ground-truth labels."""
    logger.info("Evaluating detection accuracy on verified ground-truth data...")
    model = YOLO("yolo11n.pt")

    # Ground truth annotations for standard benchmark images
    ground_truth = {
        "bus.jpg": {"bus": 1, "person": 4},
        "zidane.jpg": {"person": 2, "tie": 2},
        "grace_hopper.jpg": {"person": 1, "tie": 1},
        "webcam_dark_baseline.jpg": {},
        "webcam_live.jpg": {},
    }

    # Configurations to evaluate: Baseline (Current production defaults) vs Recommended
    configs = [
        {"name": "Current Default (conf=0.35, imgsz=640, dev=cpu)", "conf": 0.35, "imgsz": 640, "device": "cpu"},
        {"name": "Balanced Recall (conf=0.25, imgsz=640, dev=cpu)", "conf": 0.25, "imgsz": 640, "device": "cpu"},
        {"name": "Balanced MPS (conf=0.25, imgsz=640, dev=mps)", "conf": 0.25, "imgsz": 640, "device": "mps" if torch.backends.mps.is_available() else "cpu"},
        {"name": "High Recall (conf=0.15, imgsz=640, dev=cpu)", "conf": 0.15, "imgsz": 640, "device": "cpu"},
        {"name": "Strict High Precision (conf=0.45, imgsz=640, dev=cpu)", "conf": 0.45, "imgsz": 640, "device": "cpu"},
    ]

    records = []

    for cfg in configs:
        dev = cfg["device"]
        sz = cfg["imgsz"]
        conf = cfg["conf"]
        cfg_name = cfg["name"]

        for img_label, gt_counts in ground_truth.items():
            img_path = os.path.join("assets/test_images", img_label)
            if not os.path.exists(img_path):
                continue
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            # Run inference
            t0 = time.perf_counter()
            res = model(frame, imgsz=sz, device=dev, conf=conf, verbose=False)
            if dev == "mps":
                torch.mps.synchronize()
            lat_ms = (time.perf_counter() - t0) * 1000.0

            boxes = res[0].boxes
            detected_classes = [model.names[int(b.cls[0])] for b in boxes] if boxes else []
            det_confs = [round(float(b.conf[0]), 3) for b in boxes] if boxes else []

            # Count occurrences per class
            pred_counts: Dict[str, int] = {}
            for cls_name in detected_classes:
                pred_counts[cls_name] = pred_counts.get(cls_name, 0) + 1

            total_gt = sum(gt_counts.values())
            total_pred = len(detected_classes)

            # Calculate True Positives, False Positives, False Negatives
            tp = 0
            for cls_name, count in pred_counts.items():
                gt_c = gt_counts.get(cls_name, 0)
                tp += min(count, gt_c)

            fp = max(0, total_pred - tp)
            fn = max(0, total_gt - tp)

            records.append({
                "config_name": cfg_name,
                "image_source": img_label,
                "device": dev,
                "conf_threshold": conf,
                "imgsz": sz,
                "gt_total_objects": total_gt,
                "detected_total": total_pred,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "missed_detections": fn,
                "detected_classes": "|".join(detected_classes),
                "confidence_scores": "|".join([str(c) for c in det_confs]),
                "inference_latency_ms": round(lat_ms, 2),
            })

    # Save to CSV
    if records:
        fieldnames = list(records[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        logger.info("Saved ground truth benchmark to %s (%d records)", output_csv, len(records))

    return records


def run_scale_and_blur_experiments(scale_csv: str, blur_csv: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate object detection robustness against simulated distance/scale and motion blur."""
    logger.info("Evaluating scale (distance) and blur degradation curves...")
    model = YOLO("yolo11n.pt")
    bus_path = "assets/test_images/bus.jpg"
    orig = cv2.imread(bus_path)
    if orig is None:
        return [], []

    h, w = orig.shape[:2]

    # 1. Scale / Distance Experiment
    scales = [1.0, 0.75, 0.50, 0.35, 0.25]
    scale_records = []

    for scale in scales:
        sw, sh = int(w * scale), int(h * scale)
        scaled_frame = cv2.resize(orig, (sw, sh), interpolation=cv2.INTER_AREA)

        for conf in [0.25, 0.35]:
            t0 = time.perf_counter()
            res = model(scaled_frame, imgsz=640, device="cpu", conf=conf, verbose=False)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            boxes = res[0].boxes
            detected = [model.names[int(b.cls[0])] for b in boxes] if boxes else []
            confs = [round(float(b.conf[0]), 3) for b in boxes] if boxes else []

            # In bus.jpg: 1 bus, 4 persons
            has_bus = "bus" in detected
            person_count = detected.count("person")

            scale_records.append({
                "simulated_distance_factor": f"{1.0/scale:.2f}x",
                "scale_factor": scale,
                "input_resolution": f"{sw}x{sh}",
                "conf_threshold": conf,
                "total_detections": len(detected),
                "detected_classes": "|".join(detected),
                "confs": "|".join(str(c) for c in confs),
                "bus_detected": has_bus,
                "persons_detected": person_count,
                "missed_persons": max(0, 4 - person_count),
                "latency_ms": round(lat_ms, 2)
            })

    if scale_records:
        with open(scale_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(scale_records[0].keys()))
            writer.writeheader()
            writer.writerows(scale_records)
        logger.info("Saved scale/distance benchmark to %s (%d records)", scale_csv, len(scale_records))

    # 2. Motion Blur Experiment
    blur_records = []
    blur_kernels = [1, 5, 11, 21, 31]

    for k in blur_kernels:
        if k == 1:
            blurred = orig
        else:
            kernel = np.zeros((k, k))
            kernel[int((k - 1) / 2), :] = np.ones(k)
            kernel /= k
            blurred = cv2.filter2D(orig, -1, kernel)

        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        for conf in [0.25, 0.35]:
            t0 = time.perf_counter()
            res = model(blurred, imgsz=640, device="cpu", conf=conf, verbose=False)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            boxes = res[0].boxes
            detected = [model.names[int(b.cls[0])] for b in boxes] if boxes else []
            confs = [round(float(b.conf[0]), 3) for b in boxes] if boxes else []

            blur_records.append({
                "blur_kernel_size": k,
                "laplacian_variance": round(lap_var, 2),
                "conf_threshold": conf,
                "total_detections": len(detected),
                "detected_classes": "|".join(detected),
                "confidence_scores": "|".join(str(c) for c in confs),
                "latency_ms": round(lat_ms, 2)
            })

    if blur_records:
        with open(blur_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(blur_records[0].keys()))
            writer.writeheader()
            writer.writerows(blur_records)
        logger.info("Saved motion blur benchmark to %s (%d records)", blur_csv, len(blur_records))

    return scale_records, blur_records


def main() -> None:
    """Run full Phase 1 benchmark suite."""
    print("=" * 72)
    print("PHASE 1: YOLO OBJECT DETECTION DIAGNOSIS & BASELINE BENCHMARK")
    print("=" * 72)

    os.makedirs("benchmarks/results", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    cam_csv = "benchmarks/results/phase1_camera_diagnostics.csv"
    cfg_csv = "benchmarks/results/phase1_configuration_experiments.csv"
    gt_csv = "benchmarks/results/phase1_ground_truth_benchmark.csv"
    scale_csv = "benchmarks/results/phase1_scale_distance_benchmark.csv"
    blur_csv = "benchmarks/results/phase1_blur_benchmark.csv"

    run_camera_diagnostics(cam_csv)
    run_configuration_experiments(cfg_csv)
    run_ground_truth_benchmark(gt_csv)
    run_scale_and_blur_experiments(scale_csv, blur_csv)

    print("\nBenchmark complete! Result files generated in benchmarks/results/:")
    for p in [cam_csv, cfg_csv, gt_csv, scale_csv, blur_csv]:
        if os.path.exists(p):
            print(f" - {p}")


if __name__ == "__main__":
    main()
