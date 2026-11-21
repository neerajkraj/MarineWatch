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


def compute_box_iou(b1: List[float], b2: List[float]) -> float:
    """Computes Intersection-over-Union for [x1, y1, x2, y2]."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def apply_soft_nms(
    boxes: List[List[float]],
    scores: List[float],
    class_ids: List[int],
    sigma: float = 0.5,
    score_thresh: float = 0.12,
    iou_thresh: float = 0.45,
) -> Tuple[List[List[float]], List[float], List[int]]:
    """
    Soft-NMS with Gaussian decay to preserve clustered crab pot strings
    and adjacent debris without abruptly dropping overlapping boxes.
    """
    if not boxes:
        return [], [], []

    boxes_arr = np.array(boxes, dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)
    classes_arr = np.array(class_ids, dtype=np.int32)

    kept_boxes = []
    kept_scores = []
    kept_classes = []

    for cid in np.unique(classes_arr):
        mask = classes_arr == cid
        c_boxes = boxes_arr[mask].copy()
        c_scores = scores_arr[mask].copy()

        while len(c_boxes) > 0:
            max_idx = int(np.argmax(c_scores))
            max_box = c_boxes[max_idx]
            max_score = float(c_scores[max_idx])

            if max_score < score_thresh:
                break

            kept_boxes.append(max_box.tolist())
            kept_scores.append(max_score)
            kept_classes.append(int(cid))

            c_boxes = np.delete(c_boxes, max_idx, axis=0)
            c_scores = np.delete(c_scores, max_idx, axis=0)

            if len(c_boxes) == 0:
                break

            for i in range(len(c_boxes)):
                iou = compute_box_iou(max_box.tolist(), c_boxes[i].tolist())
                if iou >= 0.45:
                    # Same physical target detected multiple times (e.g. from TTA flip or overlapping tiles)
                    # Discard duplicate to prevent two boxes rendering on top of each other
                    c_scores[i] = 0.0
                elif iou > 0.0:
                    # Adjacent touching objects (e.g. clustered crab pots in a line): apply smooth Gaussian decay
                    weight = np.exp(-(iou * iou) / sigma)
                    c_scores[i] *= weight

    return kept_boxes, kept_scores, kept_classes


def verify_acoustic_shadow(
    gray_img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    raw_score: float,
) -> Tuple[float, bool, float]:
    """
    Physics-Informed Acoustic Shadow Verifier:
    Legitimate seabed debris reflects sound (bright highlight) and occludes sound behind it (dark acoustic shadow).
    Natural sediment ripples lack deep trailing acoustic shadows.
    """
    h_img, w_img = gray_img.shape[:2]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    obj_patch = gray_img[y1:y2, x1:x2]
    obj_mean = float(np.mean(obj_patch)) if obj_patch.size > 0 else 128.0

    # Sample adjacent downstream trailing shadow region
    sy1 = min(h_img, y2)
    sy2 = min(h_img, y2 + int(bh * 1.2) + 4)
    sx1 = max(0, x1 - 2)
    sx2 = min(w_img, x2 + 2)
    shadow_patch = gray_img[sy1:sy2, sx1:sx2]
    shadow_mean = float(np.mean(shadow_patch)) if shadow_patch.size > 0 else obj_mean

    # Sample ambient background ring around contact
    by1 = max(0, y1 - int(bh * 0.4))
    by2 = min(h_img, y2 + int(bh * 0.4))
    bx1 = max(0, x1 - int(bw * 0.4))
    bx2 = min(w_img, x2 + int(bw * 0.4))
    ambient_patch = gray_img[by1:by2, bx1:bx2]
    ambient_mean = float(np.mean(ambient_patch)) if ambient_patch.size > 0 else 100.0

    # Contrast metrics
    shadow_contrast = max(0.0, (ambient_mean - shadow_mean) / (ambient_mean + 1e-4))
    has_shadow = (shadow_mean < ambient_mean * 0.90) and (shadow_contrast >= 0.10)

    if has_shadow:
        # Physical acoustic shadow confirmed -> boost confidence score
        adjusted_score = min(0.99, raw_score + 0.05 * (1.0 + shadow_contrast))
    else:
        # High reflection without trailing shadow indicates sediment ripple or seabed texture
        adjusted_score = max(0.05, raw_score - 0.08)

    return adjusted_score, has_shadow, float(shadow_contrast)


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

    def run_detection(
        self,
        image_path: Path,
        output_dir: Path = None,
        conf_thresh: float = None,
        anomaly_threshold: float = None,
        use_tta: bool = True,
        use_shadow_filter: bool = True,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, str], Dict[str, int]]:
        if output_dir is None:
            output_dir = OUTPUTS_DIR

        output_dir.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        img_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Active thresholds
        current_conf = conf_thresh if conf_thresh is not None else 0.15
        current_anomaly_th = anomaly_threshold if anomaly_threshold is not None else self.anomaly_threshold

        # -------------------------------------------------------------
        # STEP 1 & 2: Module A - Tiled YOLO Detection with TTA & Soft-NMS
        # -------------------------------------------------------------
        raw_boxes_xyxy = []
        confidences = []
        class_ids = []

        for crop, _, (x0, y0) in tile_image(img_3ch, [], tile=416, overlap=0.2):
            th, tw = crop.shape[:2]

            # Pass 1: Standard orientation
            results = self.detector.predict(crop, imgsz=416, device="cpu", conf=current_conf * 0.85, verbose=False)
            for res in results:
                for box in res.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    score = float(box.conf[0].cpu().numpy())
                    cid = int(box.cls[0].cpu().numpy())
                    raw_boxes_xyxy.append([float(x0 + bx1), float(y0 + by1), float(x0 + bx2), float(y0 + by2)])
                    confidences.append(score)
                    class_ids.append(cid)

            # Pass 2: TTA Horizontal Flip (Acoustic angle reflection)
            if use_tta:
                crop_flip = cv2.flip(crop, 1)
                results_flip = self.detector.predict(crop_flip, imgsz=416, device="cpu", conf=current_conf * 0.85, verbose=False)
                for res in results_flip:
                    for box in res.boxes:
                        bx1_f, by1_f, bx2_f, by2_f = box.xyxy[0].cpu().numpy()
                        score = float(box.conf[0].cpu().numpy())
                        cid = int(box.cls[0].cpu().numpy())

                        bx1_unflip = tw - bx2_f
                        bx2_unflip = tw - bx1_f
                        raw_boxes_xyxy.append([float(x0 + bx1_unflip), float(y0 + by1_f), float(x0 + bx2_unflip), float(y0 + by2_f)])
                        confidences.append(score)
                        class_ids.append(cid)

        # Soft-NMS across all overlapping candidates (preserves dense pot lines)
        detector_detections = []
        if raw_boxes_xyxy:
            kept_boxes, kept_scores, kept_classes = apply_soft_nms(
                raw_boxes_xyxy,
                confidences,
                class_ids,
                sigma=0.5,
                score_thresh=current_conf * 0.80,
                iou_thresh=self.iou_thresh,
            )

            # Physics-Informed Acoustic Shadow Verification
            for box, score, cid in zip(kept_boxes, kept_scores, kept_classes):
                gx1, gy1, gx2, gy2 = [int(round(v)) for v in box]
                gx1 = max(0, min(w - 1, gx1))
                gy1 = max(0, min(h - 1, gy1))
                gx2 = max(0, min(w, gx2))
                gy2 = max(0, min(h, gy2))

                if use_shadow_filter:
                    adj_score, has_shadow, shadow_contrast = verify_acoustic_shadow(gray, gx1, gy1, gx2, gy2, score)
                else:
                    adj_score, has_shadow, shadow_contrast = score, True, 0.0

                if adj_score >= current_conf:
                    detector_detections.append({
                        "x1": gx1,
                        "y1": gy1,
                        "x2": gx2,
                        "y2": gy2,
                        "source": "Module A (YOLO+TTA)",
                        "label": self.class_names.get(cid, "contact"),
                        "score": float(adj_score),
                        "has_shadow": bool(has_shadow),
                        "shadow_contrast": round(float(shadow_contrast), 3),
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
                v_max = max(current_anomaly_th * 2.0, float(np.percentile(avg_anomaly_map, 99.5)))
                norm_map = np.clip((avg_anomaly_map - v_min) / (v_max - v_min + 1e-6), 0.0, 1.0)
                heatmap_uint8 = (norm_map * 255).astype(np.uint8)

                # Smooth the heatmap for clean visual thermal contouring
                heatmap_blurred = cv2.GaussianBlur(heatmap_uint8, (15, 15), 0)
                color_heatmap = cv2.applyColorMap(heatmap_blurred, cv2.COLORMAP_JET)

                # Blend thermal heatmap over grayscale sonar image
                heatmap_overlay = cv2.addWeighted(img_3ch, 0.60, color_heatmap, 0.40, 0)

                # Threshold to extract connected anomaly bounding regions
                binary_mask = (avg_anomaly_map > current_anomaly_th).astype(np.uint8) * 255
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

        # Add structural attributes (id, width, height, area)
        for idx, d in enumerate(final_detections):
            d["id"] = idx + 1
            d["width"] = int(d["x2"] - d["x1"])
            d["height"] = int(d["y2"] - d["y1"])
            d["area"] = int(d["width"] * d["height"])

        # Save files
        img_name = Path(image_path).name
        raw_path = output_dir / f"raw_{img_name}"
        annotated_path = output_dir / f"annotated_{img_name}"
        heatmap_path = output_dir / f"heatmap_{img_name}"
        fused_path = output_dir / f"fused_{img_name}"

        cv2.imwrite(str(raw_path), img_3ch)
        cv2.imwrite(str(annotated_path), bbox_img)
        cv2.imwrite(str(heatmap_path), heatmap_overlay)
        cv2.imwrite(str(fused_path), fused_img)

        # Append to report.csv
        report_path = output_dir / "report.csv"
        write_header = not report_path.exists()
        with open(report_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["image", "id", "x1", "y1", "x2", "y2", "width", "height", "source", "label", "score", "has_shadow", "shadow_contrast"])
            for d in final_detections:
                writer.writerow([
                    img_name,
                    d.get("id", ""),
                    d["x1"], d["y1"], d["x2"], d["y2"],
                    d.get("width", ""), d.get("height", ""),
                    d["source"], d["label"], f"{d['score']:.4f}",
                    d.get("has_shadow", False),
                    d.get("shadow_contrast", 0.0),
                ])

        file_dict = {
            "raw": raw_path.name,
            "annotated": annotated_path.name,
            "heatmap": heatmap_path.name,
            "fused": fused_path.name,
        }

        meta = {
            "width": w,
            "height": h,
            "filename": img_name,
            "total_detections": len(final_detections),
            "anomaly_threshold_used": float(current_anomaly_th),
            "conf_thresh_used": float(current_conf),
        }

        print(f"[Inference] Finished: {len(final_detections)} detections logged. Heatmap & Fused views saved.")
        return bbox_img, final_detections, file_dict, meta


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
