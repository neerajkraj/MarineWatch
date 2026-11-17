"""
MarineWatch — Phase 5: Train Module A (YOLO-nano CPU Detector)

Supports two operating modes:
1. Baseline Training:
   - Starts from yolo11n.pt
   - Backbone layers frozen (freeze=10)
   - Standard learning rate (lr0=0.01)

2. Anti-Overfitting Fine-Tuning Pass:
   - Starts from previous best checkpoint (outputs/sss_detector/weights/best.pt)
   - Unfreezes mid-level layers (freeze=5) to learn sonar acoustic shadows
   - Strict regularization against overfitting:
     * dropout=0.15 in detection head
     * weight_decay=0.0015 (3x stronger L2 penalty)
     * lr0=0.001 (10x smaller initial learning rate) with cos_lr=True
     * patience=3 (early stopping if validation loss plateaus)
   - Sonar-tailored augmentations (grayscale intensity, flips, mosaic)
   - Automatic export to ONNX (imgsz=416) for CPU dashboard inference
"""

import sys
import time
import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "data_sss.yaml"
OUTPUTS_DIR = BASE_DIR / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="Train or Fine-Tune MarineWatch YOLO-nano Detector")
    parser.add_argument("epochs_pos", nargs="?", type=int, default=None, help="Number of epochs (positional)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--finetune", action="store_true", help="Run anti-overfitting fine-tuning on best.pt")
    parser.add_argument("--freeze", type=int, default=None, help="Number of backbone layers to freeze")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate in detection head")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="L2 weight decay penalty")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--weights", type=str, default=None, help="Custom starting weights path")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine epochs
    if args.epochs is not None:
        epochs = args.epochs
    elif args.epochs_pos is not None:
        epochs = args.epochs_pos
    else:
        epochs = 5 if args.finetune else 1

    stage1_best = OUTPUTS_DIR / "sss_detector" / "weights" / "best.pt"
    is_finetune = args.finetune or (args.weights is not None)

    # Configuration profiles
    if is_finetune:
        weights_to_load = Path(args.weights) if args.weights else stage1_best
        if not weights_to_load.exists():
            print(f"[ERROR] Fine-tune requested but starting checkpoint not found: {weights_to_load}")
            sys.exit(1)

        # Back up stage 1 checkpoint if not already done
        backup_pt = weights_to_load.parent / "best_stage1.pt"
        if not backup_pt.exists() and weights_to_load.name == "best.pt":
            shutil.copy2(weights_to_load, backup_pt)
            print(f"[Backup] Saved stage 1 baseline to {backup_pt}")

        freeze_layers = args.freeze if args.freeze is not None else 5
        dropout_val = args.dropout if args.dropout is not None else 0.15
        lr0_val = args.lr0 if args.lr0 is not None else 0.001
        weight_decay_val = args.weight_decay if args.weight_decay is not None else 0.0015
        patience_val = args.patience if args.patience is not None else 3
        cos_lr_val = True

        mode_name = "ANTI-OVERFITTING FINE-TUNING PASS"
    else:
        weights_to_load = "yolo11n.pt"
        freeze_layers = args.freeze if args.freeze is not None else 10
        dropout_val = args.dropout if args.dropout is not None else 0.0
        lr0_val = args.lr0 if args.lr0 is not None else 0.01
        weight_decay_val = args.weight_decay if args.weight_decay is not None else 0.0005
        patience_val = args.patience if args.patience is not None else 4
        cos_lr_val = False

        mode_name = "BASELINE DETECTOR TRAINING"

    print("=" * 70)
    print(f"MarineWatch — {mode_name} ({epochs} epoch(s))")
    print(f"Base weights    : {weights_to_load}")
    print(f"Layers frozen   : 0-{freeze_layers - 1} (freeze={freeze_layers})")
    print(f"Dropout rate    : {dropout_val} (anti-overfitting regularizer)")
    print(f"Weight decay    : {weight_decay_val} (L2 regularization)")
    print(f"Learning rate   : {lr0_val} (cos_lr={cos_lr_val})")
    print(f"Early patience  : {patience_val} epochs")
    print(f"Batch size / res: batch=8, imgsz=416, cache=disk")
    print("Device          : CPU strictly (no CUDA)")
    print(f"Data config     : {CONFIG_PATH}")
    print("=" * 70)

    try:
        model = YOLO(str(weights_to_load))
    except Exception as e:
        if not is_finetune:
            print(f"[Model] yolo11n.pt failed ({e}), falling back to yolov8n.pt")
            model = YOLO("yolov8n.pt")
        else:
            raise e

    start_time = time.time()

    results = model.train(
        data=str(CONFIG_PATH),
        device="cpu",
        imgsz=416,
        epochs=epochs,
        batch=8,
        workers=2,
        cache="disk",
        patience=patience_val,
        freeze=freeze_layers,
        dropout=dropout_val,
        weight_decay=weight_decay_val,
        lr0=lr0_val,
        lrf=0.01,
        cos_lr=cos_lr_val,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,            # Sonar is grayscale intensity
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

    print("\n" + "=" * 70)
    print("TRAINING RUN COMPLETE")
    print("=" * 70)
    print(f"Total time elapsed: {minutes:.2f} minutes ({elapsed_sec:.1f} seconds)")
    print(f"Average time per epoch: {sec_per_epoch / 60.0:.2f} minutes/epoch ({sec_per_epoch:.1f} sec/epoch)")
    print("=" * 70)

    best_weights = OUTPUTS_DIR / "sss_detector" / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\n[Export] Exporting updated best weights {best_weights} to ONNX format (imgsz=416)...")
        onnx_model = YOLO(str(best_weights))
        onnx_path = onnx_model.export(format="onnx", imgsz=416)
        print(f"[Export] ONNX model ready: {onnx_path}")


if __name__ == "__main__":
    main()
