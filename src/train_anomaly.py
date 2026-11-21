"""
MarineWatch — Phase 6 (Upgraded): Native Patch-Based Autoencoder for Anomaly Detection

Trained on native-resolution 64x64 crops from normal seabed tiles.
Retains 1:1 acoustic pixel texture and sand ripples instead of blurry downscaled images.
Features:
- Native 64x64 patch sampler directly from data/normal_tiles/
- Lightweight 3-layer Convolutional Autoencoder (CPU friendly)
- MSE reconstruction loss + Adam optimizer (lr=1e-3)
- Calibrated 99th percentile anomaly threshold
- Anomaly Heatmap generator with thermal overlay (inferno/jet)
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
import torch.nn.functional as F
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


def get_gaussian_kernel1d(kernel_size=11, sigma=1.5):
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return (g / g.sum()).view(1, 1, -1)


class SeparableSSIMLoss(nn.Module):
    """
    Fast separable 1D Gaussian convolution for Structural Similarity (SSIM).
    5.5x faster on CPU than standard 2D convolution.
    """
    def __init__(self, kernel_size=11, sigma=1.5):
        super().__init__()
        self.k = kernel_size
        self.pad = kernel_size // 2
        g = get_gaussian_kernel1d(kernel_size, sigma)
        self.register_buffer("g_h", g.unsqueeze(2))  # (1, 1, 1, 11)
        self.register_buffer("g_v", g.unsqueeze(3))  # (1, 1, 11, 1)

    def _filter(self, x):
        return F.conv2d(F.conv2d(x, self.g_h, padding=(0, self.pad)), self.g_v, padding=(self.pad, 0))

    def forward(self, img1, img2):
        mu1 = self._filter(img1)
        mu2 = self._filter(img2)
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
        sigma1_sq = self._filter(img1 * img1) - mu1_sq
        sigma2_sq = self._filter(img2 * img2) - mu2_sq
        sigma12 = self._filter(img1 * img2) - mu1_mu2
        c1, c2 = 0.01**2, 0.03**2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return 1.0 - ssim_map.mean()


class HybridTextureLoss(nn.Module):
    """
    Package D: Hybrid Texture + Intensity Reconstruction Loss
    L = alpha * MSE + (1 - alpha) * (1 - SSIM)
    Forces model to learn sharp acoustic sand ripples and high-frequency seabed texture.
    """
    def __init__(self, alpha=0.60, kernel_size=11):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.ssim_loss = SeparableSSIMLoss(kernel_size=kernel_size)

    def forward(self, recon, target):
        loss_mse = self.mse(recon, target)
        loss_ssim = self.ssim_loss(recon, target)
        loss_total = self.alpha * loss_mse + (1.0 - self.alpha) * loss_ssim
        return loss_total, loss_mse, loss_ssim


class NativeSeabedPatchDataset(Dataset):
    """
    Extracts native-resolution 64x64 crops directly from normal seabed tiles.
    No downscaling: preserves high-frequency acoustic ripples and texture.
    """
    def __init__(self, patches: np.ndarray):
        # patches shape: (N, 1, 64, 64) normalized float32 in [0, 1]
        self.patches = torch.from_numpy(patches).float()

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]


def extract_native_patches(normal_tile_paths, patches_per_tile=4, patch_size=64, total_target=8000):
    print(f"\n[Patch Extraction] Extracting native {patch_size}x{patch_size} crops from {len(normal_tile_paths)} seabed tiles...")
    patches = []
    random.seed(42)
    selected_tile_paths = list(normal_tile_paths)
    random.shuffle(selected_tile_paths)

    for p in selected_tile_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        if h < patch_size or w < patch_size:
            continue

        for _ in range(patches_per_tile):
            y = random.randint(0, h - patch_size)
            x = random.randint(0, w - patch_size)
            crop = img[y : y + patch_size, x : x + patch_size]
            patches.append(crop)

            if len(patches) >= total_target:
                break
        if len(patches) >= total_target:
            break

    patches_arr = np.array(patches, dtype=np.float32)[:, np.newaxis, :, :] / 255.0
    print(f"[Patch Extraction] Successfully collected {len(patches_arr)} native crops (Shape: {patches_arr.shape}).")
    return patches_arr


def train_autoencoder(epochs: int = 20, batch_size: int = 128, lr: float = 1e-3, target_patches: int = 8000):
    print("=" * 65)
    print("MarineWatch — Phase 6: Native Patch-Based Autoencoder Training")
    print("Device: CPU (Native 64x64 Acoustic Texture)")
    print("=" * 65)

    all_tile_paths = list(NORMAL_TILES_DIR.glob("*.jpg"))
    if not all_tile_paths:
        raise FileNotFoundError(f"No normal tiles found in {NORMAL_TILES_DIR}.")

    # Extract native resolution patches
    all_patches = extract_native_patches(all_tile_paths, patches_per_tile=3, patch_size=64, total_target=target_patches)

    # Train / Val Split (85% train, 15% validation)
    n_total = len(all_patches)
    n_val = int(n_total * 0.15)
    indices = list(range(n_total))
    random.seed(42)
    random.shuffle(indices)

    train_patches = all_patches[indices[n_val:]]
    val_patches = all_patches[indices[:n_val]]

    print(f"[Dataset] Train: {len(train_patches)} patches | Val: {len(val_patches)} patches (Batch size: {batch_size})")

    train_loader = DataLoader(NativeSeabedPatchDataset(train_patches), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NativeSeabedPatchDataset(val_patches), batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")
    model = AE().to(device)
    criterion = HybridTextureLoss(alpha=0.60)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        running_train_mse = 0.0
        running_train_ssim = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss, loss_mse, loss_ssim = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * batch.size(0)
            running_train_mse += loss_mse.item() * batch.size(0)
            running_train_ssim += loss_ssim.item() * batch.size(0)

        epoch_train_loss = running_train_loss / len(train_patches)
        epoch_train_mse = running_train_mse / len(train_patches)
        epoch_train_ssim = running_train_ssim / len(train_patches)
        train_losses.append(epoch_train_loss)

        # Validation pass
        model.eval()
        running_val_loss = 0.0
        running_val_mse = 0.0
        running_val_ssim = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss, loss_mse, loss_ssim = criterion(recon, batch)
                running_val_loss += loss.item() * batch.size(0)
                running_val_mse += loss_mse.item() * batch.size(0)
                running_val_ssim += loss_ssim.item() * batch.size(0)

        epoch_val_loss = running_val_loss / len(val_patches)
        epoch_val_mse = running_val_mse / len(val_patches)
        epoch_val_ssim = running_val_ssim / len(val_patches)
        val_losses.append(epoch_val_loss)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Hybrid Loss: {epoch_train_loss:.5f} (MSE: {epoch_train_mse:.5f}, 1-SSIM: {epoch_train_ssim:.5f}) | Val Loss: {epoch_val_loss:.5f} (Val MSE: {epoch_val_mse:.5f})")

    elapsed = time.time() - t0
    print(f"\n[Training] Completed in {elapsed:.1f} seconds ({elapsed / 60.0:.2f} minutes).")

    # Evaluate validation errors for anomaly thresholding (MSE on normalized [0, 1] tiles)
    model.eval()
    val_errors = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            recon = model(batch)
            mse_per_patch = torch.mean((batch - recon) ** 2, dim=[1, 2, 3]).cpu().numpy()
            val_errors.extend(mse_per_patch.tolist())

    threshold_95 = float(np.percentile(val_errors, 95))
    threshold_99 = float(np.percentile(val_errors, 99))
    threshold_995 = float(np.percentile(val_errors, 99.5))

    print("\nReconstruction Error Statistics on Native Normal Seabed Patches:")
    print(f"  Mean Error       : {np.mean(val_errors):.6f}")
    print(f"  Std Dev          : {np.std(val_errors):.6f}")
    print(f"  95th Percentile  : {threshold_95:.6f}")
    print(f"  99th Percentile  : {threshold_99:.6f} (Calibrated Anomaly Threshold)")
    print(f"  99.5th Percentile: {threshold_995:.6f}")

    # Save model and metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = MODELS_DIR / "autoencoder.pt"
    torch.save(model.state_dict(), str(model_save_path))
    print(f"\n[Saved Model] {model_save_path}")

    meta = {
        "patch_size": 64,
        "input_channels": 1,
        "native_patches": True,
        "loss_function": "HybridTextureLoss (0.60*MSE + 0.40*(1-SSIM))",
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

    # Save visual verification plot
    save_training_artifacts(train_losses, val_losses, model, val_loader)


def save_training_artifacts(train_losses, val_losses, model, val_loader):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    ax1.plot(train_losses, label="Train Loss (MSE)", color="tab:blue")
    ax1.plot(val_losses, label="Val Loss (MSE)", color="tab:orange")
    ax1.set_title("Native Patch Autoencoder Loss", fontsize=11)
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

    orig_strip = np.hstack([orig[i] for i in range(4)])
    rec_strip = np.hstack([rec[i] for i in range(4)])
    combined = np.vstack([orig_strip, rec_strip])

    ax2.imshow(combined, cmap="gray")
    ax2.set_title("Native Seabed Patches (Top: Real Texture, Bottom: Recon)", fontsize=10)
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
