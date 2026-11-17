"""
MarineWatch — Phase 2: Data Extraction Script

Extracts GhostVision and NOMBO / MILCO raw datasets from downloaded zip archives:
- GhostVision: extracts sss-crab-pot-detection-ds/ from GhostVision_DatasetAndModels.zip
- NOMBO: extracts 24574879.zip and its inner year zips (2010, 2015, 2017, 2018, 2021)
"""

import os
import sys
import io
import zipfile
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
GHOSTVISION_DIR = RAW_DIR / "ghostvision"
NOMBO_DIR = RAW_DIR / "nombo"


def extract_ghostvision():
    gv_zip = GHOSTVISION_DIR / "GhostVision_DatasetAndModels.zip"
    if not gv_zip.exists():
        print(f"[GhostVision] Error: {gv_zip} does not exist.")
        return False
        
    ds_target = GHOSTVISION_DIR / "sss-crab-pot-detection-ds"
    if ds_target.exists() and any(ds_target.iterdir()):
        print(f"[GhostVision] Target directory {ds_target} already exists with content.")
        return True

    print(f"\n[GhostVision] Extracting sss-crab-pot-detection-ds from {gv_zip.name}...")
    with zipfile.ZipFile(gv_zip, "r") as z:
        members = [m for m in z.namelist() if m.startswith("sss-crab-pot-detection-ds/")]
        print(f"Total files to extract: {len(members)}")
        for member in tqdm(members, desc="Extracting GhostVision"):
            z.extract(member, GHOSTVISION_DIR)
            
    print("[GhostVision] Extraction complete.")
    return True


def extract_nombo():
    outer_zip = NOMBO_DIR / "24574879.zip"
    if not outer_zip.exists():
        print(f"[NOMBO] Error: {outer_zip} does not exist.")
        return False

    print(f"\n[NOMBO] Extracting survey year archives from {outer_zip.name}...")
    with zipfile.ZipFile(outer_zip, "r") as z_outer:
        outer_members = z_outer.namelist()
        print(f"Archives in {outer_zip.name}: {outer_members}")
        
        # Extract each year zip directly into NOMBO_DIR
        for item in outer_members:
            if item.endswith(".zip"):
                year_name = Path(item).stem
                year_dir = NOMBO_DIR / year_name
                if year_dir.exists() and any(year_dir.iterdir()):
                    print(f"  - {item}: already extracted in {year_dir.name}")
                    continue
                
                print(f"  - Extracting {item}...")
                zip_data = io.BytesIO(z_outer.read(item))
                with zipfile.ZipFile(zip_data) as z_inner:
                    z_inner.extractall(NOMBO_DIR)
            else:
                z_outer.extract(item, NOMBO_DIR)
                
    print("[NOMBO] Extraction complete.")
    return True


def main():
    print("=" * 60)
    print("MarineWatch — Phase 2: Extracting Raw Sonar Datasets")
    print("=" * 60)
    
    ok_gv = extract_ghostvision()
    ok_nombo = extract_nombo()
    
    if ok_gv and ok_nombo:
        print("\nAll datasets extracted successfully!")
    else:
        print(f"\nExtraction summary: GhostVision={ok_gv}, NOMBO={ok_nombo}")


if __name__ == "__main__":
    main()
