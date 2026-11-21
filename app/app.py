"""
MarineWatch v2.0 — Modern Sonar Intelligence Studio & REST API (Packages A & B & C)

Features:
- Async REST API (/api/analyze, /api/sample)
- Multi-Image Batch Survey Processor (/api/batch_analyze, /api/sample_batch)
- Side-by-Side Raw vs AI Inspection
- Test-Time Augmentation (TTA) + Soft-NMS + Physics Shadow Verification
- Professional Printable Survey Dossier (/report/print)
- Fast ONNX-accelerated CPU Inference (<120 ms)
"""

import os
import sys
import shutil
import time
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask.json.provider import DefaultJSONProvider

# Add src to python path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
from infer import MarineWatchPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "outputs" / "dashboard_uploads"
TEST_IMG_DIR = BASE_DIR / "data" / "processed" / "sss" / "images" / "test"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

class RobustNumpyJSONProvider(DefaultJSONProvider):
    """Safely serializes numpy scalars, arrays, and int32/float32 types to standard JSON."""
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

def sanitize_json(data):
    """Recursively converts all numpy scalars and arrays to native Python types."""
    if isinstance(data, dict):
        return {str(k): sanitize_json(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [sanitize_json(v) for v in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    return data

app = Flask(__name__, template_folder="templates", static_folder="static")
app.json = RobustNumpyJSONProvider(app)
app.config["UPLOAD_FOLDER"] = str(UPLOADS_DIR)
app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024

PIPELINE = None
LAST_ANALYSIS = {}
LAST_BATCH_RESULTS = {}


def get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = MarineWatchPipeline()
    return PIPELINE


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    global LAST_ANALYSIS
    if "sonar_file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded in form data"}), 400

    file = request.files["sonar_file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    raw_path = UPLOADS_DIR / file.filename
    file.save(str(raw_path))

    pipeline = get_pipeline()
    t0 = time.time()
    bbox_img, detections, file_dict, meta = pipeline.run_detection(raw_path, output_dir=UPLOADS_DIR)
    elapsed_ms = int((time.time() - t0) * 1000)

    count_ghost = sum(1 for d in detections if d["label"] == "ghost_gear")
    count_manmade = sum(1 for d in detections if d["label"] == "manmade_object")
    count_anomaly = sum(1 for d in detections if d["label"] == "anomaly")

    meta["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    meta["latency_ms"] = elapsed_ms

    LAST_ANALYSIS = sanitize_json({
        "files": file_dict,
        "detections": detections,
        "meta": meta,
        "counts": {
            "ghost_gear": count_ghost,
            "manmade_object": count_manmade,
            "anomaly": count_anomaly,
            "total": len(detections),
        },
    })

    return jsonify({
        "success": True,
        "files": file_dict,
        "detections": LAST_ANALYSIS["detections"],
        "counts": LAST_ANALYSIS["counts"],
        "meta": LAST_ANALYSIS["meta"],
        "latency": elapsed_ms,
    })


@app.route("/api/sample", methods=["GET"])
def api_sample():
    global LAST_ANALYSIS
    sample_key = request.args.get("key", "sample1")

    # Pick samples from test split
    nombo_samples = sorted(list(TEST_IMG_DIR.glob("*nombo*.jpg")))
    ghost_samples = sorted(list(TEST_IMG_DIR.glob("*baycove*.jpg")))

    if sample_key == "sample2" and ghost_samples:
        target_sample = ghost_samples[0]
    elif nombo_samples:
        target_sample = nombo_samples[0]
    else:
        all_samples = sorted(list(TEST_IMG_DIR.glob("*.jpg")))
        if not all_samples:
            return jsonify({"success": False, "error": "No test samples available"}), 404
        target_sample = all_samples[0]

    dest_path = UPLOADS_DIR / target_sample.name
    shutil.copy(target_sample, dest_path)

    pipeline = get_pipeline()
    t0 = time.time()
    bbox_img, detections, file_dict, meta = pipeline.run_detection(dest_path, output_dir=UPLOADS_DIR)
    elapsed_ms = int((time.time() - t0) * 1000)

    count_ghost = sum(1 for d in detections if d["label"] == "ghost_gear")
    count_manmade = sum(1 for d in detections if d["label"] == "manmade_object")
    count_anomaly = sum(1 for d in detections if d["label"] == "anomaly")

    meta["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    meta["latency_ms"] = elapsed_ms

    LAST_ANALYSIS = sanitize_json({
        "files": file_dict,
        "detections": detections,
        "meta": meta,
        "counts": {
            "ghost_gear": count_ghost,
            "manmade_object": count_manmade,
            "anomaly": count_anomaly,
            "total": len(detections),
        },
    })

    return jsonify({
        "success": True,
        "files": file_dict,
        "detections": LAST_ANALYSIS["detections"],
        "counts": LAST_ANALYSIS["counts"],
        "meta": LAST_ANALYSIS["meta"],
        "latency": elapsed_ms,
    })


@app.route("/api/batch_analyze", methods=["POST"])
def api_batch_analyze():
    global LAST_BATCH_RESULTS
    files = request.files.getlist("sonar_files")
    if not files or files[0].filename == "":
        return jsonify({"success": False, "error": "No sonar files selected for batch"}), 400

    pipeline = get_pipeline()
    results = []
    total_ghost = 0
    total_manmade = 0
    total_anomaly = 0
    t_start = time.time()

    for f in files:
        filename = f.filename
        save_path = UPLOADS_DIR / filename
        f.save(str(save_path))

        t0 = time.time()
        bbox_img, dets, file_dict, meta = pipeline.run_detection(save_path, output_dir=UPLOADS_DIR)
        lat = int((time.time() - t0) * 1000)

        n_ghost = sum(1 for d in dets if d["label"] == "ghost_gear")
        n_manmade = sum(1 for d in dets if d["label"] == "manmade_object")
        n_anom = sum(1 for d in dets if d["label"] == "anomaly")

        total_ghost += n_ghost
        total_manmade += n_manmade
        total_anomaly += n_anom

        if n_ghost > 0:
            status = "GHOST_GEAR"
            status_label = f"Ghost Gear ({n_ghost})"
            badge = "ghost"
        elif n_manmade > 0:
            status = "MANMADE"
            status_label = f"Contact ({n_manmade})"
            badge = "manmade"
        elif n_anom > 0:
            status = "ANOMALY"
            status_label = f"Anomaly ({n_anom})"
            badge = "anomaly"
        else:
            status = "CLEAN"
            status_label = "Clean Seafloor"
            badge = "clean"

        results.append({
            "filename": filename,
            "status": status,
            "status_label": status_label,
            "badge": badge,
            "counts": {
                "ghost_gear": n_ghost,
                "manmade_object": n_manmade,
                "anomaly": n_anom,
                "total": len(dets),
            },
            "files": file_dict,
            "detections": dets,
            "meta": meta,
            "latency_ms": lat,
        })

    total_elapsed_ms = int((time.time() - t_start) * 1000)

    # Write batch_report.csv
    batch_report_path = UPLOADS_DIR / "batch_report.csv"
    with open(batch_report_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["image", "id", "status", "label", "score", "has_shadow", "shadow_contrast", "x1", "y1", "x2", "y2", "width", "height"])
        for r in results:
            for d in r["detections"]:
                writer.writerow([
                    r["filename"],
                    d.get("id", ""),
                    r["status"],
                    d["label"],
                    f"{d['score']:.4f}",
                    d.get("has_shadow", False),
                    d.get("shadow_contrast", 0.0),
                    d["x1"], d["y1"], d["x2"], d["y2"],
                    d.get("width", ""), d.get("height", ""),
                ])

    LAST_BATCH_RESULTS = {
        "success": True,
        "total_images": len(results),
        "results": results,
        "summary": {
            "total_images": len(results),
            "flagged_images": sum(1 for r in results if r["status"] != "CLEAN"),
            "clean_images": sum(1 for r in results if r["status"] == "CLEAN"),
            "total_ghost_gear": total_ghost,
            "total_manmade": total_manmade,
            "total_anomalies": total_anomaly,
            "total_latency_ms": total_elapsed_ms,
        },
    }
    LAST_BATCH_RESULTS = sanitize_json(LAST_BATCH_RESULTS)
    return jsonify(LAST_BATCH_RESULTS)


@app.route("/api/sample_batch", methods=["GET"])
def api_sample_batch():
    global LAST_BATCH_RESULTS
    nombo_samples = sorted(list(TEST_IMG_DIR.glob("*nombo*.jpg")))
    gv_samples = sorted(list(TEST_IMG_DIR.glob("*baycove*.jpg")))

    selected = []
    if nombo_samples:
        selected.append(nombo_samples[0])
    if gv_samples:
        selected.append(gv_samples[0])
    if len(nombo_samples) > 1:
        selected.append(nombo_samples[1])
    if len(gv_samples) > 1:
        selected.append(gv_samples[1])

    pipeline = get_pipeline()
    results = []
    total_ghost = 0
    total_manmade = 0
    total_anomaly = 0
    t_start = time.time()

    for target_sample in selected:
        dest_path = UPLOADS_DIR / target_sample.name
        shutil.copy(target_sample, dest_path)

        t0 = time.time()
        bbox_img, dets, file_dict, meta = pipeline.run_detection(dest_path, output_dir=UPLOADS_DIR)
        lat = int((time.time() - t0) * 1000)

        n_ghost = sum(1 for d in dets if d["label"] == "ghost_gear")
        n_manmade = sum(1 for d in dets if d["label"] == "manmade_object")
        n_anom = sum(1 for d in dets if d["label"] == "anomaly")

        total_ghost += n_ghost
        total_manmade += n_manmade
        total_anomaly += n_anom

        if n_ghost > 0:
            status = "GHOST_GEAR"
            status_label = f"Ghost Gear ({n_ghost})"
            badge = "ghost"
        elif n_manmade > 0:
            status = "MANMADE"
            status_label = f"Contact ({n_manmade})"
            badge = "manmade"
        elif n_anom > 0:
            status = "ANOMALY"
            status_label = f"Anomaly ({n_anom})"
            badge = "anomaly"
        else:
            status = "CLEAN"
            status_label = "Clean Seafloor"
            badge = "clean"

        results.append({
            "filename": target_sample.name,
            "status": status,
            "status_label": status_label,
            "badge": badge,
            "counts": {
                "ghost_gear": n_ghost,
                "manmade_object": n_manmade,
                "anomaly": n_anom,
                "total": len(dets),
            },
            "files": file_dict,
            "detections": dets,
            "meta": meta,
            "latency_ms": lat,
        })

    total_elapsed_ms = int((time.time() - t_start) * 1000)

    # Write batch_report.csv
    batch_report_path = UPLOADS_DIR / "batch_report.csv"
    with open(batch_report_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["image", "id", "status", "label", "score", "has_shadow", "shadow_contrast", "x1", "y1", "x2", "y2", "width", "height"])
        for r in results:
            for d in r["detections"]:
                writer.writerow([
                    r["filename"],
                    d.get("id", ""),
                    r["status"],
                    d["label"],
                    f"{d['score']:.4f}",
                    d.get("has_shadow", False),
                    d.get("shadow_contrast", 0.0),
                    d["x1"], d["y1"], d["x2"], d["y2"],
                    d.get("width", ""), d.get("height", ""),
                ])

    LAST_BATCH_RESULTS = {
        "success": True,
        "total_images": len(results),
        "results": results,
        "summary": {
            "total_images": len(results),
            "flagged_images": sum(1 for r in results if r["status"] != "CLEAN"),
            "clean_images": sum(1 for r in results if r["status"] == "CLEAN"),
            "total_ghost_gear": total_ghost,
            "total_manmade": total_manmade,
            "total_anomalies": total_anomaly,
            "total_latency_ms": total_elapsed_ms,
        },
    }
    LAST_BATCH_RESULTS = sanitize_json(LAST_BATCH_RESULTS)
    return jsonify(LAST_BATCH_RESULTS)


@app.route("/download_batch_report", methods=["GET"])
def download_batch_report():
    batch_file = UPLOADS_DIR / "batch_report.csv"
    if batch_file.exists():
        return send_file(str(batch_file), as_attachment=True, download_name="marinewatch_batch_survey_report.csv")
    return "Batch report not generated yet.", 404


@app.route("/report/print", methods=["GET"])
def report_print():
    global LAST_ANALYSIS
    if not LAST_ANALYSIS:
        # Preload sample 1 if no analysis run yet
        api_sample()

    return render_template(
        "report_print.html",
        files=LAST_ANALYSIS.get("files", {}),
        detections=LAST_ANALYSIS.get("detections", []),
        meta=LAST_ANALYSIS.get("meta", {}),
        count_ghost=LAST_ANALYSIS.get("counts", {}).get("ghost_gear", 0),
        count_manmade=LAST_ANALYSIS.get("counts", {}).get("manmade_object", 0),
        count_anomaly=LAST_ANALYSIS.get("counts", {}).get("anomaly", 0),
    )


@app.route("/download_report", methods=["GET"])
def download_report():
    report_file = UPLOADS_DIR / "report.csv"
    if not report_file.exists():
        report_file = BASE_DIR / "outputs" / "report.csv"

    if report_file.exists():
        return send_file(str(report_file), as_attachment=True, download_name="marinewatch_survey_report.csv")
    return "Report not generated yet.", 404


@app.route("/files/<path:filename>")
def serve_file(filename):
    return send_from_directory(str(UPLOADS_DIR), filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[MarineWatch Dashboard v2.0] Starting server at http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
