"""
MarineWatch — Phase 7: Inference & Dual-Module Fusion Pipeline

Processes any side-scan sonar image:
1. Slices image into 416x416 overlapping tiles.
2. Runs Module A (YOLO-nano CPU / ONNX) -> maps boxes to full-image coordinates -> global NMS.
3. Runs Module B (Convolutional Autoencoder) on 64x64 patches -> computes anomaly reconstruction map -> extracts connected regions.
4. Performs fusion: retains all YOLO detections and flags non-overlapping high-anomaly seabed regions.
5. Saves annotated visualization and generates report.csv.
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from tile import tile_image
from train_anomaly import AE

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


class MarineWatchPipeline:
    def __init__(
        self,
        yolo_path: Path = None,
        ae_path: Path = None,
        threshold_path: Path = None,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
    ):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # 1. Module A: YOLO-nano
        if yolo_path is None:
            # Check for ONNX first, then PyTorch best.pt, then yolo11n.pt
            onnx_path = OUTPUTS_DIR / "sss_detector" / "weights" / "best.onnx"
            pt_path = OUTPUTS_DIR / "sss_detector" / "weights" / "best.pt"
            if onnx_path.exists():
                yolo_path = onnx_path
            elif pt_path.exists():
                yolo_path = pt_path
            else:
                yolo_path = BASE_DIR / "yolo11n.pt"

        print(f"[Pipeline] Loading Module A (Detector): {yolo_path}")
        self.detector = YOLO(str(yolo_path))
        self.class_names = {0: "ghost_gear", 1: "manmade_object"}

        # 2. Module B: Autoencoder Anomaly Detector
        if ae_path is None:
            ae_path = MODELS_DIR / "autoencoder.pt"
        if threshold_path is None:
            threshold_path = MODELS_DIR / "anomaly_threshold.json"

        self.ae = None
        self.anomaly_threshold = 0.015  # Fallback default

        if ae_path.exists():
            print(f"[Pipeline] Loading Module B (Autoencoder): {ae_path}")
            self.ae = AE()
            self.ae.load_state_dict(torch.load(str(ae_path), map_location="cpu"))
            self.ae.eval()

            if threshold_path.exists():
                with open(threshold_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.anomaly_threshold = meta.get("recommended_threshold", 0.015)
                print(f"[Pipeline] Anomaly threshold loaded: {self.anomaly_threshold:.6f}")
        else:
            print("[Pipeline] Module B weights not found. Running in detector-only mode.")

    def run_detection(self, image_path: Path, output_dir: Path = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if output_dir is None:
            output_dir = OUTPUTS_DIR

        output_dir.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        img_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # -------------------------------------------------------------
        # STEP 1 & 2: Module A - Tiled YOLO Detection
        # -------------------------------------------------------------
        raw_boxes = []
        confidences = []
        class_ids = []

        for crop, _, (x0, y0) in tile_image(img_3ch, [], tile=416, overlap=0.2):
            th, tw = crop.shape[:2]
            results = self.detector.predict(crop, imgsz=416, device="cpu", conf=self.conf_thresh, verbose=False)
            
            for res in results:
                for box in res.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    score = float(box.conf[0].cpu().numpy())
                    cid = int(box.cls[0].cpu().numpy())

                    # Map coordinates back to full image space
                    gx1 = int(round(x0 + bx1))
                    gy1 = int(round(y0 + by1))
                    gx2 = int(round(x0 + bx2))
                    gy2 = int(round(y0 + by2))

                    raw_boxes.append([gx1, gy1, gx2 - gx1, gy2 - gy1])  # [x, y, w, h] for cv2 NMS
                    confidences.append(score)
                    class_ids.append(cid)

        # Global NMS across overlapping tiles
        detector_detections = []
        if raw_boxes:
            indices = cv2.dnn.NMSBoxes(raw_boxes, confidences, self.conf_thresh, self.iou_thresh)
            if len(indices) > 0:
                for idx in indices.flatten():
                    x, y, bw, bh = raw_boxes[idx]
                    detector_detections.append({
                        "x1": max(0, x),
                        "y1": max(0, y),
                        "x2": min(w, x + bw),
                        "y2": min(h, y + bh),
                        "source": "Module A (YOLO)",
                        "label": self.class_names.get(class_ids[idx], "contact"),
                        "score": confidences[idx],
                    })

        # -------------------------------------------------------------
        # STEP 3: Module B - Autoencoder Anomaly Mapping
        # -------------------------------------------------------------
        anomaly_regions = []
        if self.ae is not None:
            patch_size = 64
            step = 32
            anomaly_map = np.zeros((h, w), dtype=np.float32)
            counts_map = np.zeros((h, w), dtype=np.float32)

            patches = []
            coords = []

            for py in range(0, max(1, h - patch_size + 1), step):
                for px in range(0, max(1, w - patch_size + 1), step):
                    patch = gray[py : py + patch_size, px : px + patch_size]
                    if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                        t = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0) / 255.0
                        patches.append(t)
                        coords.append((px, py))

            if patches:
                batch_tensors = torch.cat(patches, dim=0)
                with torch.no_grad():
                    recons = self.ae(batch_tensors)
                    # MSE per patch
                    mses = torch.mean((batch_tensors - recons) ** 2, dim=[1, 2, 3]).numpy()

                for (px, py), mse in zip(coords, mses):
                    anomaly_map[py : py + patch_size, px : px + patch_size] += mse
                    counts_map[py : py + patch_size, px : px + patch_size] += 1.0

                counts_map[counts_map == 0] = 1.0
                avg_anomaly_map = anomaly_map / counts_map

                # Threshold to create binary mask
                binary_mask = (avg_anomaly_map > self.anomaly_threshold).astype(np.uint8) * 255
                # Morphological filtering to remove single-pixel speckles
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask)
                for label_idx in range(1, num_labels):
                    area = stats[label_idx, cv2.CC_STAT_AREA]
                    if area >= 200:  # Ignore tiny noise spots
                        ax1 = stats[label_idx, cv2.CC_STAT_LEFT]
                        ay1 = stats[label_idx, cv2.CC_STAT_TOP]
                        aw = stats[label_idx, cv2.CC_STAT_WIDTH]
                        ah = stats[label_idx, cv2.CC_STAT_HEIGHT]
                        peak_score = float(np.max(avg_anomaly_map[labels == label_idx]))

                        anomaly_regions.append({
                            "x1": ax1,
                            "y1": ay1,
                            "x2": ax1 + aw,
                            "y2": ay1 + ah,
                            "source": "Module B (AE)",
                            "label": "anomaly",
                            "score": peak_score,
                        })

        # -------------------------------------------------------------
        # STEP 4: Fusion & Filtering (IoU < 0.1)
        # -------------------------------------------------------------
        final_detections = list(detector_detections)

        def compute_iou(b1, b2):
            ix1 = max(b1["x1"], b2["x1"])
            iy1 = max(b1["y1"], b2["y1"])
            ix2 = min(b1["x2"], b2["x2"])
            iy2 = min(b1["y2"], b2["y2"])
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            inter = (ix2 - ix1) * (iy2 - iy1)
            area1 = (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"])
            area2 = (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"])
            union = area1 + area2 - inter
            return inter / union if union > 0 else 0.0

        for anom in anomaly_regions:
            overlaps_detector = False
            for det in detector_detections:
                if compute_iou(anom, det) >= 0.1:
                    overlaps_detector = True
                    break
            if not overlaps_detector:
                final_detections.append(anom)

        # -------------------------------------------------------------
        # STEP 5: Annotate Image & Export Report
        # -------------------------------------------------------------
        annotated_img = img_3ch.copy()
        color_map = {
            "ghost_gear": (0, 0, 255),       # Red
            "manmade_object": (0, 165, 255), # Orange
            "anomaly": (255, 255, 0),        # Cyan / Yellow
        }

        for d in final_detections:
            color = color_map.get(d["label"], (0, 255, 0))
            cv2.rectangle(annotated_img, (d["x1"], d["y1"]), (d["x2"], d["y2"]), color, 2)
            lbl_txt = f"{d['label']} {d['score']:.2f} ({d['source'][:3]})"
            cv2.putText(annotated_img, lbl_txt, (d["x1"], max(d["y1"] - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Save annotated image
        out_img_name = f"annotated_{Path(image_path).name}"
        out_img_path = output_dir / out_img_name
        cv2.imwrite(str(out_img_path), annotated_img)

        # Append to report.csv
        report_path = output_dir / "report.csv"
        write_header = not report_path.exists()
        with open(report_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["image", "x1", "y1", "x2", "y2", "source", "label", "score"])
            for d in final_detections:
                writer.writerow([
                    Path(image_path).name,
                    d["x1"], d["y1"], d["x2"], d["y2"],
                    d["source"], d["label"], f"{d['score']:.4f}",
                ])

        print(f"[Inference] Detections found: {len(final_detections)} (Annotated image: {out_img_path})")
        return annotated_img, final_detections


def main():
    if len(sys.argv) < 2:
        # Run test on sample image from test split
        test_images = list((BASE_DIR / "data" / "processed" / "sss" / "images" / "test").glob("*.jpg"))
        if test_images:
            sample_img = test_images[0]
            print(f"[Inference] No image provided. Using test sample: {sample_img}")
        else:
            print("Usage: python src/infer.py <path_to_sonar_image>")
            return
    else:
        sample_img = Path(sys.argv[1])

    pipeline = MarineWatchPipeline()
    pipeline.run_detection(sample_img)


if __name__ == "__main__":
    main()
