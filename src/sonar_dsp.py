"""
MarineWatch — Acoustic Signal Digital Processing (DSP) & Slant-Range Correction Engine

Hydrographic Physics Principles:
1. Water Column & Nadir Detection:
   Side-scan sonar waterfall imagery has a central zero-reflection band corresponding to the
   sound travel time through the water column before striking the first seabed return (First Bottom Return - FBR).
   Altitude H (in pixels) = distance from nadir track to FBR.

2. Slant-Range to Ground-Range (SR-to-GR) Projection:
   Acoustic sensors measure slant range Rs (hypotenuse from towfish to target).
   Flat seabed ground range Rg satisfies:
       Rg = sqrt(Rs^2 - H^2)   for Rs >= H
   Pixels near the nadir (Rs ~ H) are severely compressed.
   This module dynamically constructs a bilinear coordinate remap grid:
       Xs = sqrt(Xg^2 + H^2)
   and calls cv2.remap() to remove nadir compression and project the sonar swath onto true ground range.

3. Acoustic Contrast-Limited Adaptive Histogram Equalization (CLAHE):
   Compensates for acoustic transmission attenuation (20*log(R) + 2*alpha*R) and specular backscatter gradients.

4. Acoustic Bilateral Despeckling:
   Suppresses multiplicative acoustic speckle while strictly preserving razor-sharp shadow boundaries and debris edges.
"""

import time
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


def detect_water_column(
    img: np.ndarray,
    search_center_ratio: float = 0.50,
    intensity_threshold_factor: float = 0.35,
) -> Tuple[int, int, np.ndarray]:
    """
    Detects the central towfish flight track (nadir) and the First Bottom Return (FBR).

    Returns:
        nadir_x (int): Horizontal pixel index of the towfish track.
        altitude_px (int): Estimated towfish height above seabed in pixels.
        water_column_mask (np.ndarray): Binary mask (255 = inside water column blind zone).
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    col_means = np.mean(gray, axis=0)

    # Search for nadir minimum in the central region of the image
    c_margin = int(w * (1.0 - search_center_ratio) / 2.0)
    c_start = max(0, c_margin)
    c_end = min(w, w - c_margin)

    nadir_x = c_start + int(np.argmin(col_means[c_start:c_end]))

    # Determine threshold for first bottom acoustic backscatter return (FBR)
    mean_val = float(np.mean(col_means))
    thresh = max(12.0, mean_val * intensity_threshold_factor)

    # Scan left (port channel)
    port_fbr = nadir_x
    for x in range(nadir_x, -1, -1):
        if col_means[x] > thresh:
            port_fbr = x
            break

    # Scan right (starboard channel)
    stbd_fbr = nadir_x
    for x in range(nadir_x, w):
        if col_means[x] > thresh:
            stbd_fbr = x
            break

    h_port = abs(nadir_x - port_fbr)
    h_stbd = abs(stbd_fbr - nadir_x)

    # Average port/starboard to estimate altitude
    altitude_px = max(6, int((h_port + h_stbd) / 2))

    # Construct water column mask
    mask = np.zeros((h, w), dtype=np.uint8)
    x1 = max(0, nadir_x - altitude_px)
    x2 = min(w, nadir_x + altitude_px)
    mask[:, x1:x2] = 255

    return nadir_x, altitude_px, mask


def slant_to_ground_range(
    img: np.ndarray,
    nadir_x: Optional[int] = None,
    altitude_px: Optional[int] = None,
) -> Tuple[np.ndarray, int, int]:
    """
    Applies hydrographic Pythagorean remapping to convert slant range to flat ground range:
        Xs = sqrt(Xg^2 + H^2)

    Removes nadir distortion and expands compressed seabed textures.
    Runs in < 5 ms on Intel CPU via cv2.remap.
    """
    h, w = img.shape[:2]

    if nadir_x is None or altitude_px is None:
        det_nadir, det_alt, _ = detect_water_column(img)
        if nadir_x is None:
            nadir_x = det_nadir
        if altitude_px is None:
            altitude_px = det_alt

    # Construct coordinate grids
    grid_y, grid_x = np.indices((h, w), dtype=np.float32)
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = grid_y

    h_flt = float(altitude_px)

    # Port swath (x < nadir_x)
    port_mask = grid_x < nadir_x
    if np.any(port_mask):
        xg_port = (nadir_x - 1 - grid_x[port_mask])
        xs_port = np.sqrt(xg_port**2 + h_flt**2)
        map_x[port_mask] = np.clip(nadir_x - xs_port, 0.0, float(w - 1))

    # Starboard swath (x >= nadir_x)
    stbd_mask = grid_x >= nadir_x
    if np.any(stbd_mask):
        xg_stbd = (grid_x[stbd_mask] - nadir_x)
        xs_stbd = np.sqrt(xg_stbd**2 + h_flt**2)
        map_x[stbd_mask] = np.clip(nadir_x + xs_stbd, 0.0, float(w - 1))

    corrected = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return corrected, nadir_x, altitude_px


def enhance_acoustic_clahe(
    img: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Acoustic Contrast-Limited Adaptive Histogram Equalization.
    Normalizes local acoustic backscatter gradients across the swath without blowing out highlights.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(img)


def despeckle_acoustic(
    img: np.ndarray,
    diameter: int = 5,
    sigma_color: float = 30.0,
    sigma_space: float = 30.0,
) -> np.ndarray:
    """
    Edge-preserving acoustic bilateral despeckling filter.
    Smooths high-frequency multiplicative noise while preserving object boundaries and shadow drop-offs.
    """
    return cv2.bilateralFilter(img, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def apply_acoustic_dsp_suite(
    img: np.ndarray,
    enable_clahe: bool = True,
    enable_despeckle: bool = True,
    enable_slant_range: bool = True,
    clahe_clip: float = 2.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Executes the full acoustic DSP enhancement pipeline.
    Latency: ~8-12 ms total on CPU.
    """
    t0 = time.time()
    out = img.copy()
    meta = {}

    if enable_slant_range:
        out, nadir_x, alt_px = slant_to_ground_range(out)
        meta["nadir_x"] = nadir_x
        meta["altitude_px"] = alt_px

    if enable_despeckle:
        out = despeckle_acoustic(out)

    if enable_clahe:
        out = enhance_acoustic_clahe(out, clip_limit=clahe_clip)

    meta["dsp_latency_ms"] = round((time.time() - t0) * 1000, 2)
    return out, meta
