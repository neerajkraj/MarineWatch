"""
MarineWatch — Phase 7 (Upgraded): Dual-Module Inference with Anomaly Heatmap

Features:
1. Module A: YOLO-nano ONNX / CPU detection with tile-level slicing and global NMS.
2. Module B: Native Patch Autoencoder Anomaly Mapping -> generates a thermal anomaly heatmap
   (red-hot glow on strange seabed areas, calm blue on normal sand).
3. Dual-Module Fusion: Combines known object boxes with non-overlapping unsupervised anomaly regions.
4. Tri-view Visual Outputs:
   - annotated_<img.jpg>: Bounding boxes with class labels and confidence.
   - heatmap_<img.jpg>: Thermal acoustic anomaly heatmap overlay.
   - fused_<img.jpg>: Combined view showing bounding boxes AND thermal anomaly glow together!
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
        self.anomaly_threshold = 0.0075  # Default threshold

        if ae_path.exists():
            print(f"[Pipeline] Loading Module B (Autoencoder): {ae_path}")
            self.ae = AE()
            self.ae.load_state_dict(torch.load(str(ae_path), map_location="cpu"))
            self.ae.eval()

            if threshold_path.exists():
                with open(threshold_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.anomaly_threshold = meta.get("recommended_threshold", 0.0075)
                print(f"[Pipeline] Calibrated anomaly threshold: {self.anomaly_threshold:.6f}")
        else:
            print("[Pipeline] Module B weights not found. Running in detector-only mode.")

    def run_detection(self, image_path: Path, output_dir: Path = None) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, str]]:
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
            results = self.detector.predict(crop, imgsz=416, device="cpu", conf=self.conf_thresh, verbose=False)
            for res in results:
                for box in res.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    score = float(box.conf[0].cpu().numpy())
                    cid = int(box.cls[0].cpu().numpy())

                    gx1 = int(round(x0 + bx1))
                    gy1 = int(round(y0 + by1))
                    gx2 = int(round(x0 + bx2))
                    gy2 = int(round(y0 + by2))

                    raw_boxes.append([gx1, gy1, gx2 - gx1, gy2 - gy1])
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
        # STEP 3: Module B - Thermal Anomaly Heatmap & Region Extraction
        # -------------------------------------------------------------
        anomaly_regions = []
        heatmap_overlay = img_3ch.copy()

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
                    mses = torch.mean((batch_tensors - recons) ** 2, dim=[1, 2, 3]).numpy()

                for (px, py), mse in zip(coords, mses):
                    anomaly_map[py : py + patch_size, px : px + patch_size] += mse
                    counts_map[py : py + patch_size, px : px + patch_size] += 1.0

                counts_map[counts_map == 0] = 1.0
                avg_anomaly_map = anomaly_map / counts_map

                # Generate Thermal Colormap for Anomaly Heatmap
                # Clip and scale errors so threshold is the inflection point
                v_min = np.percentile(avg_anomaly_map, 10)
                v_max = max(self.anomaly_threshold * 2.0, float(np.percentile(avg_anomaly_map, 99.5)))
                norm_map = np.clip((avg_anomaly_map - v_min) / (v_max - v_min + 1e-6), 0.0, 1.0)
                heatmap_uint8 = (norm_map * 255).astype(np.uint8)

                # Smooth the heatmap for clean visual thermal contouring
                heatmap_blurred = cv2.GaussianBlur(heatmap_uint8, (15, 15), 0)
                color_heatmap = cv2.applyColorMap(heatmap_blurred, cv2.COLORMAP_JET)

                # Blend thermal heatmap over grayscale sonar image
                heatmap_overlay = cv2.addWeighted(img_3ch, 0.60, color_heatmap, 0.40, 0)

                # Threshold to extract connected anomaly bounding regions
                binary_mask = (avg_anomaly_map > self.anomaly_threshold).astype(np.uint8) * 255
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask)
                for label_idx in range(1, num_labels):
                    area = stats[label_idx, cv2.CC_STAT_AREA]
                    if area >= 200:
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
        # STEP 4: Dual-Module Fusion (IoU < 0.1)
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
        # STEP 5: Render Multi-View Output Images
        # -------------------------------------------------------------
        # View A: Pure Bounding Boxes
        bbox_img = img_3ch.copy()
        color_map = {
            "ghost_gear": (0, 0, 255),       # Red
            "manmade_object": (0, 165, 255), # Orange
            "anomaly": (255, 255, 0),        # Cyan / Yellow
        }

        for d in final_detections:
            color = color_map.get(d["label"], (0, 255, 0))
            cv2.rectangle(bbox_img, (d["x1"], d["y1"]), (d["x2"], d["y2"]), color, 2)
            lbl_txt = f"{d['label']} {d['score']:.2f} ({d['source'][:3]})"
            cv2.putText(bbox_img, lbl_txt, (d["x1"], max(d["y1"] - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # View B: Combined Fused View (Heatmap + Bounding Boxes)
        fused_img = heatmap_overlay.copy()
        for d in final_detections:
            color = color_map.get(d["label"], (0, 255, 0))
            cv2.rectangle(fused_img, (d["x1"], d["y1"]), (d["x2"], d["y2"]), color, 2)
            lbl_txt = f"{d['label']} {d['score']:.2f}"
            cv2.putText(fused_img, lbl_txt, (d["x1"], max(d["y1"] - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Save files
        img_name = Path(image_path).name
        annotated_path = output_dir / f"annotated_{img_name}"
        heatmap_path = output_dir / f"heatmap_{img_name}"
        fused_path = output_dir / f"fused_{img_name}"

        cv2.imwrite(str(annotated_path), bbox_img)
        cv2.imwrite(str(heatmap_path), heatmap_overlay)
        cv2.imwrite(str(fused_path), fused_img)

        # Append to report.csv
        report_path = output_dir / "report.csv"
        write_header = not report_path.exists()
        with open(report_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["image", "x1", "y1", "x2", "y2", "source", "label", "score"])
            for d in final_detections:
                writer.writerow([
                    img_name,
                    d["x1"], d["y1"], d["x2"], d["y2"],
                    d["source"], d["label"], f"{d['score']:.4f}",
                ])

        file_dict = {
            "annotated": annotated_path.name,
            "heatmap": heatmap_path.name,
            "fused": fused_path.name,
        }

        print(f"[Inference] Finished: {len(final_detections)} detections logged. Heatmap & Fused views saved.")
        return bbox_img, final_detections, file_dict


def main():
    test_images = list((BASE_DIR / "data" / "processed" / "sss" / "images" / "test").glob("*.jpg"))
    if not test_images:
        print("No test images found.")
        return
    sample_img = test_images[0]
    pipeline = MarineWatchPipeline()
    pipeline.run_detection(sample_img)


if __name__ == "__main__":
    main()
