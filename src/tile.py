"""
MarineWatch — Phase 4: Sonar Image Tiling Module

Tiles side-scan sonar images into 416x416 crops with 20% overlap,
re-maps bounding boxes, and handles object truncation/filtering.
"""

from typing import List, Tuple, Generator
import numpy as np


def tile_image(
    img: np.ndarray,
    boxes: List[Tuple[int, float, float, float, float]],
    tile: int = 416,
    overlap: float = 0.2,
    min_box_area_ratio: float = 0.5,
) -> Generator[Tuple[np.ndarray, List[Tuple[int, float, float, float, float]], Tuple[int, int]], None, None]:
    """
    Tiles an image into fixed-size crops with overlap.
    
    Args:
        img: np.ndarray of shape (H, W) or (H, W, C)
        boxes: list of (class_id, x1, y1, x2, y2) in absolute pixel coordinates.
        tile: crop size (default 416)
        overlap: overlap fraction between adjacent tiles (default 0.2)
        min_box_area_ratio: minimum fraction of original box area that must be
                            contained inside the tile to keep the box (default 0.5)
                            
    Yields:
        crop: np.ndarray (tile x tile)
        out_boxes: list of (class_id, cx, cy, w, h) in normalized YOLO format relative to the crop
        (x0, y0): top-left pixel offset of the crop in the source image
    """
    h, w = img.shape[:2]
    
    # If image is already smaller than or equal to tile size, pad or return directly
    if h <= tile and w <= tile:
        pad_bottom = max(0, tile - h)
        pad_right = max(0, tile - w)
        if pad_bottom > 0 or pad_right > 0:
            if img.ndim == 3:
                padded = np.pad(img, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="reflect")
            else:
                padded = np.pad(img, ((0, pad_bottom), (0, pad_right)), mode="reflect")
        else:
            padded = img
            
        th, tw = padded.shape[:2]
        out_boxes = []
        for c, x1, y1, x2, y2 in boxes:
            bw = x2 - x1
            bh = y2 - y1
            cx = (x1 + x2) / 2.0 / tw
            cy = (y1 + y2) / 2.0 / th
            nw = bw / tw
            nh = bh / th
            out_boxes.append((c, cx, cy, nw, nh))
        yield padded, out_boxes, (0, 0)
        return

    step = int(tile * (1.0 - overlap))
    step = max(step, 1)

    y_max = max(h - tile, 0)
    x_max = max(w - tile, 0)

    # Generate grid coordinates
    y_starts = list(range(0, y_max + step, step))
    x_starts = list(range(0, x_max + step, step))

    # Ensure edge tiles reach the exact image boundaries
    y_starts = sorted(list(set([min(y, y_max) for y in y_starts])))
    x_starts = sorted(list(set([min(x, x_max) for x in x_starts])))

    for y0c in y_starts:
        for x0c in x_starts:
            crop = img[y0c : y0c + tile, x0c : x0c + tile]
            th, tw = crop.shape[:2]
            
            # If edge crop is smaller than tile size, pad it
            if th < tile or tw < tile:
                pad_b = tile - th
                pad_r = tile - tw
                if crop.ndim == 3:
                    crop = np.pad(crop, ((0, pad_b), (0, pad_r), (0, 0)), mode="reflect")
                else:
                    crop = np.pad(crop, ((0, pad_b), (0, pad_r)), mode="reflect")
                th, tw = tile, tile

            out_boxes = []
            for c, x1, y1, x2, y2 in boxes:
                orig_area = max(0.0, (x2 - x1) * (y2 - y1))
                if orig_area <= 0:
                    continue

                # Intersection with current crop
                ix1 = max(x1, float(x0c))
                iy1 = max(y1, float(y0c))
                ix2 = min(x2, float(x0c + tw))
                iy2 = min(y2, float(y0c + th))

                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                inter_area = (ix2 - ix1) * (iy2 - iy1)
                if (inter_area / orig_area) < min_box_area_ratio:
                    continue

                # Calculate normalized YOLO format coordinates relative to the crop
                cx = ((ix1 + ix2) / 2.0 - x0c) / tw
                cy = ((iy1 + iy2) / 2.0 - y0c) / th
                nw = (ix2 - ix1) / tw
                nh = (iy2 - iy1) / th

                # Clip to valid range [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.001, min(1.0, nw))
                nh = max(0.001, min(1.0, nh))

                out_boxes.append((c, cx, cy, nw, nh))

            yield crop, out_boxes, (x0c, y0c)
