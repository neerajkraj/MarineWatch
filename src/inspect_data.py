"""
MarineWatch — Phase 2: Data Inspection Script (CPU Friendly)

Inspects raw datasets (GhostVision and NOMBO / MILCO):
- Image counts, file extensions, and resolutions
- Annotation format (GhostVision JSONL vs NOMBO YOLO txt)
- Class distributions and box statistics
- Generates 5-image preview grids with ground-truth bounding boxes overlaid
"""

import os
import sys
import json
import random
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUTS_DIR = BASE_DIR / "outputs"


def inspect_ghostvision(gv_dir: Path):
    print("\n" + "=" * 65)
    print("INSPECTING DATASET: GhostVision (Derelict Crab Pot Detection)")
    print("=" * 65)

    ds_dir = gv_dir / "sss-crab-pot-detection-ds"
    if not ds_dir.exists():
        ds_dir = gv_dir
        
    splits = ["train", "valid", "test"]
    total_images = 0
    split_stats = {}
    all_categories = {}
    box_counts_per_image = []
    resolutions = set()
    samples_with_boxes = []

    for split in splits:
        split_dir = ds_dir / split
        if not split_dir.exists():
            continue
            
        jsonl_file = split_dir / "metadata.jsonl"
        imgs = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
        split_stats[split] = {"images": len(imgs), "annotated": 0, "total_boxes": 0}
        total_images += len(imgs)

        if jsonl_file.exists():
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    fn = item.get("file_name")
                    img_path = split_dir / fn
                    objects = item.get("objects", {})
                    bboxes = objects.get("bbox", [])
                    categories = objects.get("category", [])

                    split_stats[split]["annotated"] += 1
                    split_stats[split]["total_boxes"] += len(bboxes)
                    box_counts_per_image.append(len(bboxes))

                    for cat in categories:
                        all_categories[cat] = all_categories.get(cat, 0) + 1

                    if bboxes and img_path.exists():
                        samples_with_boxes.append((img_path, bboxes, categories))

    # Sample image resolutions
    sample_imgs = list(ds_dir.rglob("*.jpg"))[:40]
    for p in sample_imgs:
        im = cv2.imread(str(p))
        if im is not None:
            resolutions.add((im.shape[1], im.shape[0]))

    print(f"Directory: {ds_dir.resolve()}")
    print(f"Total images across all splits: {total_images}")
    for split, st in split_stats.items():
        print(f"  - Split '{split}': {st['images']} images, {st['total_boxes']} total boxes")
    print(f"Image resolutions (Width x Height): {list(resolutions)}")
    print(f"Annotation format: Roboflow / COCO JSONL [x_min, y_min, width, height] in pixel coordinates")
    print(f"Class distribution: {all_categories}")
    print(f"Images with at least 1 object: {len(samples_with_boxes)} / {total_images} ({len(samples_with_boxes)/max(total_images,1)*100:.1f}%)")
    if box_counts_per_image:
        print(f"Average boxes per annotated tile: {np.mean(box_counts_per_image):.2f} (max: {max(box_counts_per_image)})")

    # Save visual verification grid
    if samples_with_boxes:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        save_ghostvision_grid(samples_with_boxes, OUTPUTS_DIR / "inspection_ghostvision.png")


def save_ghostvision_grid(samples, out_path: Path):
    random.seed(42)
    chosen = random.sample(samples, min(5, len(samples)))
    fig, axes = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 4.5))
    if len(chosen) == 1:
        axes = [axes]

    for ax, (img_path, bboxes, categories) in zip(axes, chosen):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for bbox, cat in zip(bboxes, categories):
            # bbox is [x_min, y_min, width, height]
            x, y, bw, bh = bbox
            x1, y1 = int(round(x)), int(round(y))
            x2, y2 = int(round(x + bw)), int(round(y + bh))
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 60, 60), 2)
            cv2.putText(img_rgb, f"{cat}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 230, 0), 1)

        ax.imshow(img_rgb)
        ax.set_title(f"{img_path.name[:20]}...\n({w}x{h}, {len(bboxes)} pot(s))", fontsize=9)
        ax.axis("off")

    plt.suptitle("GhostVision — Side-Scan Sonar Crab Pot Samples (Raw Ground Truth)", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"\n[Visual Grid Saved] {out_path}")


def inspect_nombo(nombo_dir: Path):
    print("\n" + "=" * 65)
    print("INSPECTING DATASET: NOMBO / MILCO (Gavia AUV Seabed Sonar)")
    print("=" * 65)

    year_dirs = [d for d in nombo_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    year_dirs.sort(key=lambda d: d.name)

    total_images = 0
    total_txts = 0
    year_stats = {}
    class_map = {0: "MILCO", 1: "NOMBO"}
    class_counts = {}
    resolutions = set()
    samples_with_boxes = []

    # Read obj.names if present
    obj_names_file = nombo_dir / "obj.names.txt"
    if not obj_names_file.exists():
        obj_names_file = nombo_dir / "Training" / "obj.names.txt"
    if obj_names_file.exists():
        with open(obj_names_file, "r", encoding="utf-8") as f:
            names = [l.strip() for l in f if l.strip()]
            class_map = {i: n for i, n in enumerate(names)}

    for yd in year_dirs:
        imgs = sorted(list(yd.glob("*.jpg")) + list(yd.glob("*.png")))
        txts = sorted(list(yd.glob("*.txt")))
        year_stats[yd.name] = {"images": len(imgs), "labels": len(txts), "boxes": 0}
        total_images += len(imgs)
        total_txts += len(txts)

        for img_path in imgs:
            lbl_path = img_path.with_suffix(".txt")
            if lbl_path.exists():
                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]
                boxes = []
                for line in lines:
                    parts = line.split()
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                    year_stats[yd.name]["boxes"] += 1
                    boxes.append((cls_id, cx, cy, w, h))
                if boxes:
                    samples_with_boxes.append((img_path, boxes))

    # Sample resolutions
    sample_imgs = list(nombo_dir.rglob("*.jpg"))[:40]
    for p in sample_imgs:
        im = cv2.imread(str(p))
        if im is not None:
            resolutions.add((im.shape[1], im.shape[0]))

    print(f"Directory: {nombo_dir.resolve()}")
    print(f"Survey years found: {[yd.name for yd in year_dirs]}")
    print(f"Total raw images: {total_images} (across {len(year_dirs)} survey years)")
    print(f"Total label files: {total_txts}")
    for y, st in year_stats.items():
        print(f"  - Year {y}: {st['images']} images, {st['boxes']} contact boxes")
    print(f"Image resolutions (Width x Height): {list(resolutions)}")
    print(f"Annotation format: YOLO txt [class cx cy w h] (normalized)")
    print(f"Class mapping: {class_map}")
    print(f"Class distribution: { {class_map.get(k, k): v for k, v in class_counts.items()} }")
    print(f"Images with contacts: {len(samples_with_boxes)} / {total_images} ({len(samples_with_boxes)/max(total_images,1)*100:.1f}%)")

    # Save visual verification grid
    if samples_with_boxes:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        save_nombo_grid(samples_with_boxes, class_map, OUTPUTS_DIR / "inspection_nombo.png")


def save_nombo_grid(samples, class_map, out_path: Path):
    random.seed(42)
    chosen = random.sample(samples, min(5, len(samples)))
    fig, axes = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 4.5))
    if len(chosen) == 1:
        axes = [axes]

    for ax, (img_path, boxes) in zip(axes, chosen):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for cls_id, cx, cy, bw, bh in boxes:
            x1 = int(round((cx - bw / 2) * w))
            y1 = int(round((cy - bh / 2) * h))
            x2 = int(round((cx + bw / 2) * w))
            y2 = int(round((cy + bh / 2) * h))
            cname = class_map.get(cls_id, str(cls_id))
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(img_rgb, cname, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        ax.imshow(img_rgb)
        ax.set_title(f"{img_path.parent.name}/{img_path.name}\n({w}x{h}, {len(boxes)} obj)", fontsize=9)
        ax.axis("off")

    plt.suptitle("NOMBO / MILCO — AUV Side-Scan Sonar Samples (Raw Ground Truth)", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"\n[Visual Grid Saved] {out_path}")


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    inspect_ghostvision(RAW_DIR / "ghostvision")
    inspect_nombo(RAW_DIR / "nombo")
    print("\n" + "=" * 65)
    print("PHASE 2 DATA INSPECTION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
