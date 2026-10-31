#!/usr/bin/env python3
"""Phase 3: Custom YOLO Model Fine-Tuning and Evaluation Suite.

Executes transfer learning from pretrained yolo11n.pt onto the custom obstacles
dataset (road_hazard, pedestrian, vehicle).

Evaluates both the original pretrained baseline and the fine-tuned custom model on
the exact same held-out test set, exporting reports/phase3_model_comparison.csv.
"""

import argparse
import csv
import logging
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Add repository root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.object_detector import YOLOObjectDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Phase3Training")

TARGET_CLASSES = {0: "road_hazard", 1: "pedestrian", 2: "vehicle"}

# Mapping from COCO class names to custom obstacle domain
COCO_TO_CUSTOM_MAP = {
    # 0: road_hazard
    "bottle": 0,
    "cup": 0,
    "box": 0,
    "chair": 0,
    "backpack": 0,
    "suitcase": 0,
    # 1: pedestrian
    "person": 1,
    # 2: vehicle
    "car": 2,
    "bus": 2,
    "truck": 2,
    "motorcycle": 2,
    "bicycle": 2
}


def train_custom_model(
    data_yaml: str = "datasets/custom_obstacles/data.yaml",
    base_model: str = "yolo11n.pt",
    output_model_path: str = "models/custom_obstacle_detector.pt",
    epochs: int = 15,
    imgsz: int = 640,
    batch: int = 4,
    device: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """Fine-tune yolo11n.pt on custom obstacles dataset using transfer learning."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    logger.info("Initializing transfer learning from '%s' on device='%s'...", base_model, device)
    model = YOLO(base_model)

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    project_dir = "runs/detect"
    run_name = "custom_obstacle_finetune"

    t0 = time.perf_counter()
    train_results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        plots=False,
        save=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        scale=0.5,
        translate=0.1,
        fliplr=0.5,
        verbose=True
    )
    training_duration_sec = time.perf_counter() - t0

    # Locate best weights using train_results.save_dir or fallback paths
    save_dir = str(getattr(train_results, "save_dir", os.path.join(project_dir, run_name)))
    candidate_paths = [
        os.path.join(save_dir, "weights", "best.pt"),
        os.path.join(save_dir, "weights", "last.pt"),
        os.path.join(project_dir, run_name, "weights", "best.pt"),
        os.path.join(project_dir, "runs", "detect", run_name, "weights", "best.pt"),
        os.path.join(project_dir, "runs", "detect", run_name, "weights", "last.pt"),
    ]
    
    found_weights = None
    for p in candidate_paths:
        if os.path.exists(p):
            found_weights = p
            break

    if found_weights:
        shutil.copyfile(found_weights, output_model_path)
        logger.info("Successfully copied fine-tuned weights from '%s' to '%s' (%.1f MB)",
                    found_weights, output_model_path, os.path.getsize(output_model_path) / (1024 * 1024))
    else:
        raise FileNotFoundError(f"Trained weights not found in candidates: {candidate_paths}")

    training_meta = {
        "base_model": base_model,
        "output_model": output_model_path,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "training_duration_sec": round(training_duration_sec, 2),
    }

    return output_model_path, training_meta


def evaluate_model_on_test_set(
    model_path: str,
    test_img_dir: str = "datasets/custom_obstacles/images/test",
    test_lbl_dir: str = "datasets/custom_obstacles/labels/test",
    is_custom_model: bool = True,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45
) -> Dict[str, Any]:
    """Evaluate a YOLO model on the exact same held-out test split."""
    logger.info("Evaluating model '%s' (is_custom=%s) on test set...", model_path, is_custom_model)
    model = YOLO(model_path)

    image_files = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    latencies = []
    total_gt = {0: 0, 1: 0, 2: 0}
    total_tp = {0: 0, 1: 0, 2: 0}
    total_fp = {0: 0, 1: 0, 2: 0}
    total_pred = {0: 0, 1: 0, 2: 0}

    detailed_results = []

    for img_name in image_files:
        base_name = os.path.splitext(img_name)[0]
        img_p = os.path.join(test_img_dir, img_name)
        lbl_p = os.path.join(test_lbl_dir, f"{base_name}.txt")

        img = cv2.imread(img_p)
        if img is None:
            continue

        # Parse ground-truth annotations
        gt_boxes = []
        if os.path.exists(lbl_p):
            with open(lbl_p, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        cls_id = int(parts[0])
                        xc, yc, w, h = [float(p) for p in parts[1:5]]
                        gt_boxes.append((cls_id, xc, yc, w, h))
                        total_gt[cls_id] = total_gt.get(cls_id, 0) + 1

        # Run timed inference
        t0 = time.perf_counter()
        res = model(img, conf=conf_thresh, iou=iou_thresh, imgsz=640, device="cpu", verbose=False)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        boxes = res[0].boxes
        pred_boxes = []
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                raw_cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                # Map class ID based on whether model is custom or COCO-pretrained
                if is_custom_model:
                    mapped_cls = raw_cls_id if raw_cls_id in TARGET_CLASSES else None
                else:
                    raw_name = model.names.get(raw_cls_id, "")
                    mapped_cls = COCO_TO_CUSTOM_MAP.get(raw_name, None)

                if mapped_cls is not None:
                    # Bounding box coordinates normalized
                    h_img, w_img = img.shape[:2]
                    xyxy = b.xyxy[0].detach().cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    xc = ((x1 + x2) / 2.0) / w_img
                    yc = ((y1 + y2) / 2.0) / h_img
                    bw = (x2 - x1) / float(w_img)
                    bh = (y2 - y1) / float(h_img)
                    pred_boxes.append((mapped_cls, xc, yc, bw, bh, conf))
                    total_pred[mapped_cls] = total_pred.get(mapped_cls, 0) + 1

        # Match predictions to ground truth via IoU >= 0.45
        matched_gt = set()
        matched_pred = set()

        for p_idx, (p_cls, p_xc, p_yc, p_w, p_h, p_conf) in enumerate(pred_boxes):
            best_iou = 0.0
            best_gt_idx = -1
            # Compute box coordinates
            px1, py1, px2, py2 = p_xc - p_w/2, p_yc - p_h/2, p_xc + p_w/2, p_yc + p_h/2

            for g_idx, (g_cls, g_xc, g_yc, g_w, g_h) in enumerate(gt_boxes):
                if g_idx in matched_gt or p_cls != g_cls:
                    continue
                gx1, gy1, gx2, gy2 = g_xc - g_w/2, g_yc - g_h/2, g_xc + g_w/2, g_yc + g_h/2
                # Intersect
                ix1 = max(px1, gx1)
                iy1 = max(py1, gy1)
                ix2 = min(px2, gx2)
                iy2 = min(py2, gy2)
                inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                p_area = p_w * p_h
                g_area = g_w * g_h
                union_area = p_area + g_area - inter_area
                iou_val = inter_area / union_area if union_area > 0 else 0.0

                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = g_idx

            if best_iou >= 0.45 and best_gt_idx != -1:
                matched_gt.add(best_gt_idx)
                matched_pred.add(p_idx)
                total_tp[p_cls] = total_tp.get(p_cls, 0) + 1
            else:
                total_fp[p_cls] = total_fp.get(p_cls, 0) + 1

        # Count false negatives
        for g_idx, (g_cls, *_) in enumerate(gt_boxes):
            if g_idx not in matched_gt:
                pass  # False negative accounted for via total_gt - total_tp

    # Calculate overall metrics
    all_tp = sum(total_tp.values())
    all_fp = sum(total_fp.values())
    all_gt = sum(total_gt.values())
    all_fn = max(0, all_gt - all_tp)

    overall_precision = round(all_tp / (all_tp + all_fp), 3) if (all_tp + all_fp) > 0 else 1.0
    overall_recall = round(all_tp / (all_tp + all_fn), 3) if (all_tp + all_fn) > 0 else 1.0
    overall_f1 = round(2 * (overall_precision * overall_recall) / (overall_precision + overall_recall), 3) if (overall_precision + overall_recall) > 0 else 0.0

    # Per-class breakdowns
    per_class = {}
    for cls_id, cls_name in TARGET_CLASSES.items():
        tp_c = total_tp.get(cls_id, 0)
        fp_c = total_fp.get(cls_id, 0)
        gt_c = total_gt.get(cls_id, 0)
        fn_c = max(0, gt_c - tp_c)
        prec_c = round(tp_c / (tp_c + fp_c), 3) if (tp_c + fp_c) > 0 else 1.0
        rec_c = round(tp_c / (tp_c + fn_c), 3) if (tp_c + fn_c) > 0 else 1.0
        per_class[cls_name] = {
            "ground_truth": gt_c,
            "true_positives": tp_c,
            "false_positives": fp_c,
            "false_negatives": fn_c,
            "precision": prec_c,
            "recall": rec_c,
        }

    med_lat = float(np.median(latencies)) if latencies else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    fps = round(1000.0 / med_lat, 1) if med_lat > 0 else 0.0

    return {
        "model_path": model_path,
        "is_custom": is_custom_model,
        "total_test_images": len(image_files),
        "total_ground_truth": all_gt,
        "total_true_positives": all_tp,
        "total_false_positives": all_fp,
        "total_false_negatives": all_fn,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1_score": overall_f1,
        "median_latency_ms": round(med_lat, 2),
        "mean_latency_ms": round(mean_lat, 2),
        "effective_fps": fps,
        "per_class": per_class,
    }


def run_phase3_pipeline() -> None:
    """Execute complete Phase 3 fine-tuning, evaluation, and comparison."""
    print("=" * 76)
    print("PHASE 3: CUSTOM YOLO MODEL TRAINING & MODEL COMPARISON SUITE")
    print("=" * 76)

    data_yaml = "datasets/custom_obstacles/data.yaml"
    pretrained_weights = "yolo11n.pt"
    custom_weights = "models/custom_obstacle_detector.pt"

    # Step 1 & 2: Train custom model
    print("\n[1/3] Fine-tuning custom model on Apple Silicon MPS...")
    trained_path, train_meta = train_custom_model(
        data_yaml=data_yaml,
        base_model=pretrained_weights,
        output_model_path=custom_weights,
        epochs=15,
        imgsz=640,
        batch=4
    )

    # Step 3: Evaluate Pretrained Baseline on Test Set
    print("\n[2/3] Evaluating Baseline yolo11n.pt on held-out test set...")
    eval_baseline = evaluate_model_on_test_set(
        model_path=pretrained_weights,
        is_custom_model=False,
        conf_thresh=0.25,
        iou_thresh=0.45
    )

    # Step 4: Evaluate Fine-Tuned Custom Model on Test Set
    print("\n[3/3] Evaluating Fine-Tuned custom_obstacle_detector.pt on held-out test set...")
    eval_custom = evaluate_model_on_test_set(
        model_path=custom_weights,
        is_custom_model=True,
        conf_thresh=0.25,
        iou_thresh=0.45
    )

    # Step 5: Export Comparison CSV
    out_csv = "reports/phase3_model_comparison.csv"
    os.makedirs("reports", exist_ok=True)

    csv_rows = [
        {
            "model_name": "Pretrained yolo11n.pt (Baseline)",
            "model_type": "COCO Pretrained",
            "weights_path": pretrained_weights,
            "weights_size_mb": round(os.path.getsize(pretrained_weights) / (1024 * 1024), 2),
            "test_images_count": eval_baseline["total_test_images"],
            "ground_truth_total": eval_baseline["total_ground_truth"],
            "true_positives": eval_baseline["total_true_positives"],
            "false_positives": eval_baseline["total_false_positives"],
            "false_negatives": eval_baseline["total_false_negatives"],
            "overall_precision": eval_baseline["precision"],
            "overall_recall": eval_baseline["recall"],
            "overall_f1": eval_baseline["f1_score"],
            "median_latency_ms": eval_baseline["median_latency_ms"],
            "effective_fps": eval_baseline["effective_fps"],
            "road_hazard_recall": eval_baseline["per_class"]["road_hazard"]["recall"],
            "pedestrian_recall": eval_baseline["per_class"]["pedestrian"]["recall"],
            "vehicle_recall": eval_baseline["per_class"]["vehicle"]["recall"],
        },
        {
            "model_name": "Custom Obstacle Detector (Fine-Tuned)",
            "model_type": "Fine-Tuned Transfer Learning (15 Epochs)",
            "weights_path": custom_weights,
            "weights_size_mb": round(os.path.getsize(custom_weights) / (1024 * 1024), 2),
            "test_images_count": eval_custom["total_test_images"],
            "ground_truth_total": eval_custom["total_ground_truth"],
            "true_positives": eval_custom["total_true_positives"],
            "false_positives": eval_custom["total_false_positives"],
            "false_negatives": eval_custom["total_false_negatives"],
            "overall_precision": eval_custom["precision"],
            "overall_recall": eval_custom["recall"],
            "overall_f1": eval_custom["f1_score"],
            "median_latency_ms": eval_custom["median_latency_ms"],
            "effective_fps": eval_custom["effective_fps"],
            "road_hazard_recall": eval_custom["per_class"]["road_hazard"]["recall"],
            "pedestrian_recall": eval_custom["per_class"]["pedestrian"]["recall"],
            "vehicle_recall": eval_custom["per_class"]["vehicle"]["recall"],
        }
    ]

    fieldnames = list(csv_rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    logger.info("Exported Phase 3 model comparison CSV to %s", out_csv)
    print(f"\nModel comparison exported to: {out_csv}")


if __name__ == "__main__":
    run_phase3_pipeline()
