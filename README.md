# MarineWatch — Underwater Debris & Anomaly Detector (CPU-only edition)

> Automatically flag ghost fishing gear, seabed objects and unusual sonar returns, so operators no longer have to scan sonar imagery by eye.
>
> **This version is designed for a laptop with a Core i3 CPU and no GPU.** Every choice below (datasets, model size, image size, training length) is made so the whole project runs on that machine.

---

## 0. How to use this README in Antigravity

1. Create an empty folder `marinewatch/`, open it in Antigravity, put this README in the root.
2. Paste this prompt into the agent:

   > Read README.md fully. Build the project exactly as described, phase by phase (Phase 1 → Phase 7). My machine is a Core i3 with NO GPU, so never use CUDA, never use large models, and follow the CPU settings in the README. After each phase, run the code, fix errors, and report what works before moving on. Do the Phase 2 data inspection before writing any conversion code. If a download link fails, tell me and use the fallback in Section 4.

3. Do **Phase 5, step 1 (the 1-epoch speed test)** before committing to a long training run.

---

## 1. Problem

Ghost nets and lost fishing gear keep entangling marine life for years but are hard to locate. They can be as thin as a finger and almost invisible in raw imagery. Sonar operators scan side-scan sonar (SSS) images by eye, and there is **no automated way to flag unusual seabed returns**.

Precedent: WWF Germany with Microsoft AI for Good and Accenture built GhostNetZero.ai, which applies a segmentation CNN to side-scan sonar and reports roughly 90–94% detection. MarineWatch is a small, open, CPU-friendly version of that idea.

## 2. What MarineWatch does

Two modules on the same sonar image:

| Module | Question | Method | Labels needed? |
|---|---|---|---|
| **A. Object detector** | Is there a known object (ghost gear / seabed object)? | YOLO **nano** model, fine-tuned from pretrained weights | Yes (public datasets) |
| **B. Anomaly detector** | Does this seabed patch look unlike normal seabed? | Tiny convolutional autoencoder trained only on normal tiles | No |

Final flag = Module A detection OR Module B anomaly score above threshold. Output: annotated image, CSV report, simple Flask dashboard.

---

## 3. Research summary

- Labeled public sonar datasets are scarce.
- **No public labeled ghost-*net* sonar dataset was found.** GhostNetZero's data appears closed. The nearest public ghost-gear data is a **ghost crab pot** side-scan dataset (University of Delaware, GhostVision).
- Because you have no GPU, dataset **size** matters as much as relevance. Big sets (7,000+ images) are excluded or subsampled.

---

## 4. Datasets chosen for a CPU-only machine

### 4.1 Use these (in this order)

| Priority | Dataset | Sensor | Why it fits a CPU laptop | Format |
|---|---|---|---|---|
| **1** | **NOMBO / MILCO side-scan dataset** (Data in Brief, PMC10879765) | Side-scan sonar | Only **1,170 images**, ready YOLO-format boxes, collected by a Gavia AUV. Small enough to train a nano model on CPU. Provides "man-made object on seabed" class plus plenty of normal seabed. | YOLO txt |
| **2** | **GhostVision ghost crab-pot dataset** (HF: `PINGEcosystem/sss-crab-pot-detection-ds`; Zenodo record 20056679) | Side-scan sonar | Closest to the ghost-gear use case. The Zenodo bundle is ~899 MB because it includes models; **download from Hugging Face and use only the dataset, then cap it at ~1,500 tiles.** | Check after download (exact layout not verified) |
| **3 (optional)** | **Marine Debris FLS – Watertank subset** (Zenodo DOI 10.5281/zenodo.15101686, CC0) | Forward-looking sonar | Watertank images are lab-scale and light. Skip the 7,004-image quarry set unless you subsample to ~1,000. Treat as a **separate** model/demo because FLS looks very different from SSS. | Check after download |

### 4.2 Do NOT use on this machine
- **TrashCan** (7,212 optical images, ~527 MB, segmentation): wrong sensor (optical) and heavy for CPU.
- **UATD** (9,000+ multibeam images), **AI4Shipwrecks** (large high-res wreck images): too big or off-target. AI4Shipwrecks is only useful as extra "normal seabed" tiles if needed.

### 4.3 Recommended combination
**NOMBO/MILCO + GhostVision (capped)** → one side-scan YOLO-nano detector (≈2,000–2,700 tiles total). Marine Debris FLS watertank is an optional second mini-model.

> Licenses: Zenodo lists GhostVision as CC BY-SA 4.0 and the FLS data as CC0. I did not verify NOMBO's license — check the article page before publishing.

> Honesty note for your report: *"No public labeled ghost-net sonar dataset exists; ghost crab pots serve as the closest proxy, and an unsupervised anomaly module covers unlabeled returns."*

### 4.4 Class scheme

```
0 ghost_gear      # crab pots / nets (GhostVision)
1 manmade_object  # NOMBO/MILCO contacts
```
(Optional FLS model: `0 debris_object`.)

---

## 5. Architecture

```
    sonar image
        │
  [preprocess]  grayscale, tile 416x416 w/ overlap
        │
  ┌─────┴──────────────┐
[A] YOLO-nano     [B] tiny autoencoder
 boxes+conf        per-tile error score
  └─────┬──────────────┘
   [fusion + NMS]
        │
 annotated image + report.csv + Flask dashboard
```

Stack: Python 3.10+, PyTorch **CPU build**, Ultralytics, OpenCV, NumPy, pandas, scikit-learn, Flask, huggingface_hub.

---

## 6. Repository layout

```
marinewatch/
├── README.md
├── requirements.txt
├── configs/data_sss.yaml
├── data/{raw,processed,normal_tiles}/
├── src/
│   ├── download_data.py
│   ├── inspect_data.py
│   ├── tile.py
│   ├── build_dataset.py
│   ├── train_detector.py
│   ├── train_anomaly.py
│   ├── infer.py
│   └── evaluate.py
├── app/app.py
├── models/
└── outputs/
```

`requirements.txt`:

```
torch
torchvision
ultralytics
opencv-python
numpy
pandas
scikit-learn
matplotlib
tqdm
pyyaml
huggingface_hub
flask
onnxruntime
```

Install the CPU build of PyTorch (smaller download, no CUDA):

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## 7. Build phases

### Phase 1 — Environment
Create a venv, install as above, verify:
`python -c "import torch, ultralytics; print(torch.__version__)"` (GPU will be `False`; that is expected).

### Phase 2 — Download and inspect (do not skip)

```python
# src/download_data.py
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="PINGEcosystem/sss-crab-pot-detection-ds",
    repo_type="dataset",
    local_dir="data/raw/ghostvision",
)
```

Manual: download NOMBO/MILCO from the link in the PMC article (PMC10879765) into `data/raw/nombo/`.

`src/inspect_data.py` must print, per dataset: image count, image sizes, label format (YOLO txt / COCO json / masks), class names and counts, and save a grid of 5 random images with boxes drawn. The GhostVision layout was not verified, so this output decides the conversion code in Phase 3.

Disk tip: check free space first; delete anything in `data/raw` you do not use.

### Phase 3 — Convert and split
- Convert everything to YOLO txt (`class cx cy w h`, normalized). COCO → YOLO if needed; masks → boxes via connected components.
- Remap classes to Section 4.4.
- **Cap GhostVision at ~1,500 tiles** (random sample, keeping every tile that contains a pot).
- Split by survey/site, not randomly by image (neighbouring sonar tiles are near-duplicates and random splits inflate scores). 70/15/15.

### Phase 4 — Tiling at CPU-friendly size

Tile large SSS images to **416×416** with 20% overlap. Keep a box only if ≥50% of it lies inside the tile.

```python
def tile_image(img, boxes, tile=416, overlap=0.2):
    """boxes: list of (cls, x1, y1, x2, y2) in pixels. Yields (crop, yolo_boxes, (x0, y0))."""
    h, w = img.shape[:2]
    step = int(tile * (1 - overlap))
    for y0 in range(0, max(h - tile, 0) + step, step):
        for x0 in range(0, max(w - tile, 0) + step, step):
            y0c, x0c = min(y0, max(h - tile, 0)), min(x0, max(w - tile, 0))
            crop = img[y0c:y0c + tile, x0c:x0c + tile]
            th, tw = crop.shape[:2]
            out = []
            for c, x1, y1, x2, y2 in boxes:
                ix1, iy1 = max(x1, x0c), max(y1, y0c)
                ix2, iy2 = min(x2, x0c + tw), min(y2, y0c + th)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                if (ix2 - ix1) * (iy2 - iy1) / ((x2 - x1) * (y2 - y1)) < 0.5:
                    continue
                out.append((c, ((ix1 + ix2) / 2 - x0c) / tw, ((iy1 + iy2) / 2 - y0c) / th,
                            (ix2 - ix1) / tw, (iy2 - iy1) / th))
            yield crop, out, (x0c, y0c)
```

- Save tiles with **no objects** into `data/normal_tiles/` (training data for Module B).
- Keep ~15% empty tiles in the detector set to reduce false positives.
- Convert tiles to 3-channel grayscale PNG/JPG so YOLO's pretrained weights load cleanly.

### Phase 5 — Train Module A on CPU

`configs/data_sss.yaml`:

```yaml
path: data/processed/sss
train: images/train
val: images/val
test: images/test
names:
  0: ghost_gear
  1: manmade_object
```

**Step 1 — speed test (mandatory):** train 1 epoch and time it.

```python
# src/train_detector.py
import sys
from ultralytics import YOLO

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 1   # run with 1 first!
model = YOLO("yolo11n.pt")   # nano; use the newest *n* weights your ultralytics install supports
model.train(
    data="configs/data_sss.yaml",
    device="cpu",
    imgsz=416,
    epochs=epochs,
    batch=8,
    workers=2,
    cache="ram",          # use "disk" if RAM is low
    patience=10,
    freeze=10,            # freeze early backbone layers: faster, fewer parameters to update
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,
    fliplr=0.5, flipud=0.5,
    mosaic=1.0, scale=0.3,
    project="outputs", name="sss_detector",
)
```

Run `python src/train_detector.py 1`, note the minutes per epoch, then decide:
- If ≤ ~10 min/epoch → train 30–50 epochs (leave it running overnight).
- If much slower → drop `imgsz` to 320, reduce the GhostVision cap to ~800 tiles, use `batch=4`, and/or train fewer epochs.
- Use `resume=True` if the run is interrupted.

I did not benchmark your machine, so treat any time estimate as unknown until Step 1 measures it.

**Step 2 — export for fast CPU inference:**

```python
YOLO("outputs/sss_detector/weights/best.pt").export(format="onnx", imgsz=416)
```

### Phase 6 — Train Module B (tiny autoencoder, CPU-friendly)

Use small **64×64 grayscale** patches from `data/normal_tiles/` (resize each 416 tile, or crop 64×64 patches). This trains in minutes on a CPU.

```python
import torch, torch.nn as nn

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, 2, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.dec(self.enc(x))

# Train: MSE loss, Adam lr=1e-3, batch 64, ~20-30 epochs, input scaled to [0,1].
# anomaly_score = mean((x - recon)**2)
# threshold = 99th percentile of scores on held-out NORMAL validation patches.
```

Skip PatchCore/anomalib on this machine; it is heavier than needed.

### Phase 7 — Inference, fusion, dashboard

`src/infer.py`:
1. Tile the input image (416×416, 20% overlap).
2. Run the ONNX/YOLO model per tile on CPU; map boxes back to full-image coordinates; merge with NMS.
3. Run the autoencoder on 64×64 patches → anomaly map → threshold → connected regions.
4. Fusion: keep all YOLO boxes; add anomaly regions that do not overlap any YOLO box (IoU < 0.1) as `anomaly`.
5. Write annotated image + `report.csv` (`image, x1, y1, x2, y2, source, label, score`).

`app/app.py`: Flask page with an upload form, a "Run detection" button, the annotated image, and the results table. Process one image at a time (fine for CPU).

---

## 8. Compute plan for Core i3, no GPU

| Step | Cost on your laptop |
|---|---|
| Download, inspect, tile, inference, dashboard | Fine |
| Autoencoder training (64×64) | Minutes |
| YOLO-nano training (416 px, ~2k tiles) | Slow but feasible: measure with the 1-epoch test, run long jobs overnight |
| Anything larger (bigger YOLO, 640+ px, 7k-image sets) | Avoid |

Practical tips: close other programs while training; keep the laptop plugged in with sleep disabled; use `cache="ram"` only if you have enough RAM (otherwise `"disk"`); save progress with `resume=True`.

**Optional, if training is too slow:** you can run only the training script on free Google Colab or Kaggle (they provide a GPU at no cost, with usage limits) and copy `best.pt` back to your laptop. Everything else still runs locally. The project does not depend on this.

---

## 9. Evaluation

On the held-out test split (separate surveys):
- Module A: precision, recall, mAP@0.5 per class, confusion matrix.
- Module B: AUROC (normal tiles vs tiles containing objects), plus precision/recall at the chosen threshold.
- Combined: fraction of tiles flagged vs fraction of true objects caught (operator-workload view).
- Failure analysis: show ~10 false positives and ~10 misses and explain (sediment ripples, cables/pipelines that look like net lines, acoustic shadows, noise).

Reference points from the literature (not guaranteed on your data): GhostNetZero ~90% detection; NOMBO's original baseline used YOLOv4. Expect lower numbers from a nano model on CPU and say so.

---

## 10. Known limitations (put in your report)

1. No public labeled ghost-net sonar dataset; ghost crab pots are a proxy.
2. Small datasets and narrow geography (Delaware waters, one AUV survey series): expect a domain gap on other sonar hardware or seabeds.
3. Nano model + 416 px trades accuracy for CPU speed; very thin nets may be missed.
4. Nets can look like cables, pipelines or sediment lines in sonar.
5. Autoencoder thresholds are site-specific.
6. Keep a human in the loop for final verification.

---

## 11. Timeline (~2–3 weeks part-time)

| Days | Goal |
|---|---|
| 1–2 | Setup, download, inspect |
| 3–4 | Convert, split, tile |
| 5–8 | Speed test, then detector training (overnight runs) |
| 9 | Autoencoder |
| 10–11 | Inference, fusion, dashboard |
| 12–14 | Evaluation, failure analysis, write-up, demo |

---

## 12. Sources

- Marine Debris Forward-Looking Sonar Datasets: https://arxiv.org/abs/2503.22880 (data: https://doi.org/10.5281/zenodo.15101686)
- GhostVision crab-pot dataset and models: https://zenodo.org/records/20056679 (code: https://github.com/PINGEcosystem/GhostVision)
- NOMBO/MILCO side-scan dataset: https://pmc.ncbi.nlm.nih.gov/articles/PMC10879765
- AI4Shipwrecks (sonar dataset survey): https://arxiv.org/abs/2401.14546
- TrashCan 1.0 (optical, not used): https://arxiv.org/abs/2007.08097
- GhostNetZero: https://www.microsoft.com/en-us/research/?p=1150234 and https://developer.nvidia.com/blog/ai-helps-locate-dangerous-fishing-nets-lost-at-sea

---

## 13. Checklist

- [ ] CPU PyTorch + Ultralytics installed
- [ ] NOMBO/MILCO and GhostVision downloaded, `inspect_data.py` output checked
- [ ] GhostVision capped, site-based split done
- [ ] 416×416 tiles + normal tiles generated
- [ ] 1-epoch speed test done, training plan chosen
- [ ] YOLO-nano trained, ONNX exported
- [ ] Autoencoder trained, threshold set
- [ ] Fusion inference + `report.csv`
- [ ] Dashboard runs
- [ ] Limitations written
