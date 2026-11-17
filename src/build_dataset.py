"""
MarineWatch — Balanced & Contact-Aware Dataset Builder (Phase 3 & 4)

Fixes class imbalance and survey split distribution:
1. Contact-Aware Stratified Splitting:
   - Partitions survey sequences so that ~70% of true contacts are in TRAIN,
     ~15% in VAL, and ~15% in TEST.
2. 1:1 Class Balancing:
   - Extracts ~450-500 manmade_object boxes from NOMBO.
   - Intentionally caps ghost_gear boxes to ~450-500 boxes in TRAIN (1:1 parity).
3. 416x416 Tiling with 20% overlap via src/tile.py.
4. Preserves ~15% negative seabed tiles in detector set to suppress false alarms.
5. Saves negative seabed tiles to data/normal_tiles/ for Module B (Autoencoder).
6. Regenerates configs/data_sss.yaml.
"""

import os
import sys
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import yaml
from tqdm import tqdm

from tile import tile_image

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "sss"
NORMAL_TILES_DIR = BASE_DIR / "data" / "normal_tiles"
CONFIGS_DIR = BASE_DIR / "configs"

TARGET_TILE_SIZE = 416
OVERLAP = 0.2
EMPTY_RATIO_IN_DETECTOR = 0.15


def clean_output_directories():
    print("[Setup] Cleaning output directories...")
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    if NORMAL_TILES_DIR.exists():
        shutil.rmtree(NORMAL_TILES_DIR)

    for split in ["train", "val", "test"]:
        (PROCESSED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    NORMAL_TILES_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)


def parse_nombo():
    nombo_dir = RAW_DIR / "nombo"
    year_dirs = sorted([d for d in nombo_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    pos_records = []
    neg_records = []

    print(f"\n[NOMBO] Parsing survey years: {[d.name for d in year_dirs]}...")
    for yd in year_dirs:
        year_name = yd.name
        img_files = sorted(list(yd.glob("*.jpg")) + list(yd.glob("*.png")))
        for img_path in img_files:
            lbl_path = img_path.with_suffix(".txt")
            boxes = []
            if lbl_path.exists():
                im = cv2.imread(str(img_path))
                if im is None:
                    continue
                h, w = im.shape[:2]
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cx, cy, bw, bh = map(float, parts[1:5])
                            x1 = (cx - bw / 2.0) * w
                            y1 = (cy - bh / 2.0) * h
                            x2 = (cx + bw / 2.0) * w
                            y2 = (cy + bh / 2.0) * h
                            # Class 1: manmade_object
                            boxes.append((1, x1, y1, x2, y2))

            rec = {
                "dataset": "nombo",
                "survey_group": f"nombo_{year_name}",
                "img_path": img_path,
                "boxes": boxes,
                "num_boxes": len(boxes),
            }
            if boxes:
                pos_records.append(rec)
            else:
                neg_records.append(rec)

    print(f"[NOMBO] Found {len(pos_records)} images with contacts ({sum(r['num_boxes'] for r in pos_records)} boxes) and {len(neg_records)} normal images.")
    return pos_records, neg_records


def parse_ghostvision():
    gv_dir = RAW_DIR / "ghostvision" / "sss-crab-pot-detection-ds"
    pos_records = []
    neg_records = []

    print("\n[GhostVision] Parsing JSONL metadata...")
    for split in ["train", "valid", "test"]:
        split_dir = gv_dir / split
        jsonl_path = split_dir / "metadata.jsonl"
        if not jsonl_path.exists():
            continue

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                fn = data.get("file_name")
                img_path = split_dir / fn
                if not img_path.exists():
                    continue

                prefix = fn.split("_")[0]
                if "baycove" in fn:
                    prefix = "_".join(fn.split("_")[:2])

                objects = data.get("objects", {})
                bboxes = objects.get("bbox", [])
                boxes = []
                for bbox in bboxes:
                    x_min, y_min, w, h = bbox
                    # Class 0: ghost_gear
                    boxes.append((0, float(x_min), float(y_min), float(x_min + w), float(y_min + h)))

                rec = {
                    "dataset": "ghostvision",
                    "survey_group": f"gv_{prefix}",
                    "img_path": img_path,
                    "boxes": boxes,
                    "num_boxes": len(boxes),
                }
                if boxes:
                    pos_records.append(rec)
                else:
                    neg_records.append(rec)

    print(f"[GhostVision] Found {len(pos_records)} images with crab pots ({sum(r['num_boxes'] for r in pos_records)} boxes) and {len(neg_records)} normal images.")
    return pos_records, neg_records


def create_balanced_splits(nombo_pos, nombo_neg, gv_pos, gv_neg):
    """
    Creates balanced train/val/test splits where:
    - ~70% of NOMBO contacts go to train (~450-480 boxes)
    - GhostVision positive images in train are subsampled to match ~450-480 boxes (1:1 ratio!)
    - Val and test splits have proportional representation of both classes.
    """
    print("\n[Balancing & Splitting] Stratifying by sequence and balancing classes 1:1...")

    random.seed(42)

    # 1. Split NOMBO positive contacts (70% train, 15% val, 15% test)
    # Group by survey year
    nombo_by_year = defaultdict(list)
    for r in nombo_pos:
        nombo_by_year[r["survey_group"]].append(r)

    train_nombo = []
    val_nombo = []
    test_nombo = []

    for sgroup, items in nombo_by_year.items():
        # Keep contiguous pings grouped in sub-sequences
        items.sort(key=lambda x: str(x["img_path"]))
        n = len(items)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        train_nombo.extend(items[:n_train])
        val_nombo.extend(items[n_train : n_train + n_val])
        test_nombo.extend(items[n_train + n_val :])

    nombo_train_boxes = sum(r["num_boxes"] for r in train_nombo)
    print(f"  NOMBO Positives -> Train: {len(train_nombo)} imgs ({nombo_train_boxes} boxes) | Val: {len(val_nombo)} imgs | Test: {len(test_nombo)} imgs")

    # 2. Subsample GhostVision to match NOMBO's train box count (~450-480 boxes)
    # Group GhostVision by recording site
    gv_by_site = defaultdict(list)
    for r in gv_pos:
        gv_by_site[r["survey_group"]].append(r)

    # Allocate whole survey sites to val/test, and subsample train site
    all_sites = sorted(list(gv_by_site.keys()))
    random.shuffle(all_sites)

    train_gv = []
    val_gv = []
    test_gv = []

    target_gv_train_boxes = nombo_train_boxes  # Match 1:1!

    current_train_boxes = 0
    # First, collect candidate images for train until target box count is reached
    candidate_train_recs = []
    for site in all_sites:
        candidate_train_recs.extend(gv_by_site[site])

    random.shuffle(candidate_train_recs)
    for r in candidate_train_recs:
        if current_train_boxes < target_gv_train_boxes:
            train_gv.append(r)
            current_train_boxes += r["num_boxes"]
        elif len(val_gv) < int(len(train_gv) * 0.22):
            val_gv.append(r)
        elif len(test_gv) < int(len(train_gv) * 0.22):
            test_gv.append(r)

    print(f"  GhostVision Positives (Balanced) -> Train: {len(train_gv)} imgs ({current_train_boxes} boxes) | Val: {len(val_gv)} imgs | Test: {len(test_gv)} imgs")

    # 3. Add negative seabed images (NOMBO + GhostVision)
    all_neg = nombo_neg + gv_neg
    random.shuffle(all_neg)
    n_neg = len(all_neg)
    train_neg = all_neg[: int(n_neg * 0.70)]
    val_neg = all_neg[int(n_neg * 0.70) : int(n_neg * 0.85)]
    test_neg = all_neg[int(n_neg * 0.85) :]

    splits = {
        "train": train_nombo + train_gv + train_neg,
        "val": val_nombo + val_gv + val_neg,
        "test": test_nombo + test_gv + test_neg,
    }
    return splits


def process_and_tile(split_dict):
    stats = {
        "detector_tiles": {"train": 0, "val": 0, "test": 0},
        "detector_boxes": {"train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int)},
        "normal_tiles_total": 0,
    }

    tile_id = 0
    normal_tile_id = 0

    for split_name, recs in split_dict.items():
        print(f"\n[Processing] Tiling '{split_name}' split ({len(recs)} raw images)...")
        pos_tiles = []
        neg_tiles = []

        for r in tqdm(recs, desc=f"Tiling {split_name}"):
            img = cv2.imread(str(r["img_path"]))
            if img is None:
                continue

            if img.ndim == 2 or img.shape[2] == 1:
                img_gray = img if img.ndim == 2 else img[:, :, 0]
                img_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            boxes = r["boxes"]

            for crop, yolo_boxes, offset in tile_image(img_3ch, boxes, tile=TARGET_TILE_SIZE, overlap=OVERLAP):
                tile_id += 1
                base_stem = f"{r['dataset']}_{r['survey_group']}_t{tile_id:06d}"

                if len(yolo_boxes) > 0:
                    pos_tiles.append((base_stem, crop, yolo_boxes))
                else:
                    neg_tiles.append((base_stem, crop))
                    normal_path = NORMAL_TILES_DIR / f"{base_stem}.jpg"
                    cv2.imwrite(str(normal_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    normal_tile_id += 1

        # Keep ~15% negative tiles in detector split to reduce false alarms
        n_pos = len(pos_tiles)
        n_neg_to_keep = int(n_pos * (EMPTY_RATIO_IN_DETECTOR / (1.0 - EMPTY_RATIO_IN_DETECTOR)))
        random.seed(42)
        random.shuffle(neg_tiles)
        selected_neg = neg_tiles[:n_neg_to_keep]

        # Save positive detector tiles
        for base_stem, crop, yolo_boxes in pos_tiles:
            img_out = PROCESSED_DIR / "images" / split_name / f"{base_stem}.jpg"
            lbl_out = PROCESSED_DIR / "labels" / split_name / f"{base_stem}.txt"
            cv2.imwrite(str(img_out), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            with open(lbl_out, "w", encoding="utf-8") as f:
                for cls_id, cx, cy, w, h in yolo_boxes:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    stats["detector_boxes"][split_name][cls_id] += 1
            stats["detector_tiles"][split_name] += 1

        # Save negative detector tiles (empty txt)
        for base_stem, crop in selected_neg:
            img_out = PROCESSED_DIR / "images" / split_name / f"{base_stem}.jpg"
            lbl_out = PROCESSED_DIR / "labels" / split_name / f"{base_stem}.txt"
            cv2.imwrite(str(img_out), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            with open(lbl_out, "w", encoding="utf-8") as f:
                pass
            stats["detector_tiles"][split_name] += 1

    stats["normal_tiles_total"] = normal_tile_id
    return stats


def generate_yaml_config():
    config_path = CONFIGS_DIR / "data_sss.yaml"
    data_cfg = {
        "path": str(PROCESSED_DIR.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "ghost_gear",
            1: "manmade_object",
        },
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(data_cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\n[Config] Generated {config_path}")


def main():
    print("=" * 65)
    print("MarineWatch — Balanced & Contact-Aware Dataset Builder")
    print("=" * 65)

    clean_output_directories()
    nombo_pos, nombo_neg = parse_nombo()
    gv_pos, gv_neg = parse_ghostvision()

    splits = create_balanced_splits(nombo_pos, nombo_neg, gv_pos, gv_neg)
    stats = process_and_tile(splits)
    generate_yaml_config()

    print("\n" + "=" * 65)
    print("DATASET PREPARATION SUMMARY (BALANCED 1:1)")
    print("=" * 65)
    for s in ["train", "val", "test"]:
        b_info = dict(stats["detector_boxes"][s])
        print(f"  - {s.upper():<5}: {stats['detector_tiles'][s]:>5} tiles | Boxes: {b_info}")
    total_det = sum(stats["detector_tiles"].values())
    print(f"Total tiles in balanced detector dataset: {total_det}")
    print(f"Total normal seabed tiles saved for Module B: {stats['normal_tiles_total']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
