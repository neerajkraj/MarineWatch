"""
MarineWatch — Phase 6: Train Module B (Tiny Convolutional Autoencoder for Anomaly Detection)

Trains a lightweight autoencoder on normal seabed tiles (64x64 grayscale).
Runs in minutes on a standard CPU.
- MSE reconstruction loss
- Adam optimizer (lr=1e-3)
- Anomaly threshold determined at 99th percentile of held-out normal validation patches
- Saves weights to models/autoencoder.pt and metadata to models/anomaly_threshold.json
"""

import os
import sys
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
NORMAL_TILES_DIR = BASE_DIR / "data" / "normal_tiles"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


class AE(nn.Module):
    """
    Lightweight Convolutional Autoencoder for 64x64 single-channel grayscale sonar patches.
    Encoder: 64x64 -> 32x32 -> 16x16 -> 8x8 (32 channels)
    Decoder: 8x8 -> 16x16 -> 32x32 -> 64x64 (1 channel, Sigmoid)
    """
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class SonarNormalPatchDataset(Dataset):
    def __init__(self, image_paths, patch_size=64):
        self.image_paths = image_paths
        self.patch_size = patch_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1] and add channel dimension (1, 64, 64)
        tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        return tensor


def train_autoencoder(epochs: int = 20, batch_size: int = 64, lr: float = 1e-3, max_samples: int = 3000):
    print("=" * 65)
    print("MarineWatch — Phase 6: Training Anomaly Autoencoder (CPU)")
    print("=" * 65)

    all_tile_paths = list(NORMAL_TILES_DIR.glob("*.jpg"))
    if not all_tile_paths:
        raise FileNotFoundError(f"No normal tiles found in {NORMAL_TILES_DIR}. Run build_dataset.py first.")

    random.seed(42)
    random.shuffle(all_tile_paths)
    selected_paths = all_tile_paths[:max_samples]

    n_val = int(len(selected_paths) * 0.15)
    train_paths = selected_paths[n_val:]
    val_paths = selected_paths[:n_val]

    print(f"Normal tiles: {len(train_paths)} train, {len(val_paths)} validation (64x64 patches)")

    train_loader = DataLoader(SonarNormalPatchDataset(train_paths), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SonarNormalPatchDataset(val_paths), batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")
    model = AE().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)

        epoch_train_loss = running_loss / len(train_paths)
        train_losses.append(epoch_train_loss)

        # Validation loss
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = criterion(recon, batch)
                running_val_loss += loss.item() * batch.size(0)

        epoch_val_loss = running_val_loss / len(val_paths)
        val_losses.append(epoch_val_loss)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE: {epoch_train_loss:.6f} | Val MSE: {epoch_val_loss:.6f}")

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f} seconds ({elapsed / 60.0:.2f} minutes).")

    # Determine 99th percentile threshold on validation set
    model.eval()
    val_errors = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            recon = model(batch)
            # Per-patch mean squared error
            mse_per_patch = torch.mean((batch - recon) ** 2, dim=[1, 2, 3]).cpu().numpy()
            val_errors.extend(mse_per_patch.tolist())

    threshold_95 = float(np.percentile(val_errors, 95))
    threshold_99 = float(np.percentile(val_errors, 99))
    threshold_995 = float(np.percentile(val_errors, 99.5))

    print("\nReconstruction Error Statistics on Normal Seabed Validation Patches:")
    print(f"  Mean Error       : {np.mean(val_errors):.6f}")
    print(f"  Std Dev          : {np.std(val_errors):.6f}")
    print(f"  95th percentile  : {threshold_95:.6f}")
    print(f"  99th percentile  : {threshold_99:.6f} (Standard Anomaly Threshold)")
    print(f"  99.5th percentile: {threshold_995:.6f}")

    # Save model and metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = MODELS_DIR / "autoencoder.pt"
    torch.save(model.state_dict(), str(model_save_path))
    print(f"\n[Saved Model] {model_save_path}")

    meta = {
        "patch_size": 64,
        "input_channels": 1,
        "threshold_95": threshold_95,
        "threshold_99": threshold_99,
        "threshold_995": threshold_995,
        "recommended_threshold": threshold_99,
        "mean_error": float(np.mean(val_errors)),
        "std_error": float(np.std(val_errors)),
        "epochs": epochs,
        "final_val_loss": epoch_val_loss,
    }
    meta_path = MODELS_DIR / "anomaly_threshold.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[Saved Metadata] {meta_path}")

    # Plot sample reconstructions and loss curve
    save_training_artifacts(train_losses, val_losses, model, val_loader, threshold_99)


def save_training_artifacts(train_losses, val_losses, model, val_loader, threshold):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    ax1.plot(train_losses, label="Train Loss (MSE)", color="tab:blue")
    ax1.plot(val_losses, label="Val Loss (MSE)", color="tab:orange")
    ax1.set_title("Autoencoder Training & Validation Loss", fontsize=11)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Reconstructions preview
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            recon = model(batch)
            break

    orig = batch[:4].squeeze(1).numpy()
    rec = recon[:4].squeeze(1).numpy()

    # Stack original on top, reconstruction below
    orig_strip = np.hstack([orig[i] for i in range(4)])
    rec_strip = np.hstack([rec[i] for i in range(4)])
    combined = np.vstack([orig_strip, rec_strip])

    ax2.imshow(combined, cmap="gray")
    ax2.set_title("Sample Normal Patches (Top: Original, Bottom: Recon)", fontsize=10)
    ax2.axis("off")

    plt.tight_layout()
    plot_path = OUTPUTS_DIR / "anomaly_training.png"
    plt.savefig(str(plot_path), dpi=150)
    plt.close()
    print(f"[Visual Saved] {plot_path}")


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    train_autoencoder(epochs=epochs)


if __name__ == "__main__":
    main()
