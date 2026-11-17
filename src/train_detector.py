"""
MarineWatch — Phase 5: Train Module A (YOLO-nano CPU Detector)

Trains YOLO-nano on 416x416 side-scan sonar tiles using pure CPU settings.
Features:
- Mandatory 1-epoch speed test benchmark
- Freeze early backbone layers (freeze=10) for accelerated CPU training
- Sonar-tailored data augmentations (grayscale HSV, flips, mosaic)
- Automatic ONNX export of best weights for fast inference
"""

import sys
import time
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "data_sss.yaml"
OUTPUTS_DIR = BASE_DIR / "outputs"


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print("=" * 65)
    print(f"MarineWatch — Phase 5: YOLO-nano Detector Training ({epochs} epoch(s))")
    print("Device: CPU strictly (no CUDA)")
    print(f"Data config: {CONFIG_PATH}")
    print("=" * 65)

    # Use nano weights (yolo11n.pt or yolov8n.pt fallback)
    weights_name = "yolo11n.pt"
    print(f"[Model] Loading pretrained nano weights: {weights_name}")
    try:
        model = YOLO(weights_name)
    except Exception as e:
        print(f"[Model] yolo11n.pt failed ({e}), falling back to yolov8n.pt")
        weights_name = "yolov8n.pt"
        model = YOLO(weights_name)

    start_time = time.time()

    results = model.train(
        data=str(CONFIG_PATH),
        device="cpu",
        imgsz=416,
        epochs=epochs,
        batch=8,
        workers=2,
        cache="disk",          # Safe disk caching to preserve Core i3 RAM
        patience=4,
        freeze=10,            # Freeze early backbone layers for faster CPU updates
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,            # Sonar is grayscale; preserve intensity variations
        fliplr=0.5,
        flipud=0.5,           # Sonar images are symmetric along heading
        mosaic=1.0,
        scale=0.3,
        project=str(OUTPUTS_DIR),
        name="sss_detector",
        exist_ok=True,
    )

    elapsed_sec = time.time() - start_time
    minutes = elapsed_sec / 60.0
    sec_per_epoch = elapsed_sec / max(epochs, 1)

    print("\n" + "=" * 65)
    print("TRAINING SPEED BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Total time elapsed: {minutes:.2f} minutes ({elapsed_sec:.1f} seconds)")
    print(f"Average time per epoch: {sec_per_epoch / 60.0:.2f} minutes/epoch ({sec_per_epoch:.1f} sec/epoch)")
    print("=" * 65)

    # If completed multi-epoch training, export to ONNX
    best_weights = OUTPUTS_DIR / "sss_detector" / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\n[Export] Exporting {best_weights} to ONNX format (imgsz=416)...")
        onnx_model = YOLO(str(best_weights))
        onnx_path = onnx_model.export(format="onnx", imgsz=416)
        print(f"[Export] ONNX model ready: {onnx_path}")


if __name__ == "__main__":
    main()
