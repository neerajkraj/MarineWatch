"""
MarineWatch v2.0 — Modern Sonar Intelligence Studio & REST API (Package A)

Features:
- Async REST API (/api/analyze, /api/sample)
- Split Curtain Before/After Comparison
- Real-time Client-Side Confidence & Opacity Tuning
- Acoustic Shadow Target Elevation Estimator
- Professional Printable Survey Dossier (/report/print)
- Fast ONNX-accelerated CPU Inference (<120 ms)
"""

import os
import sys
import shutil
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file, redirect, url_for

# Add src to python path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
from infer import MarineWatchPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "outputs" / "dashboard_uploads"
TEST_IMG_DIR = BASE_DIR / "data" / "processed" / "sss" / "images" / "test"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = str(UPLOADS_DIR)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

PIPELINE = None
LAST_ANALYSIS = {}


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

    LAST_ANALYSIS = {
        "files": file_dict,
        "detections": detections,
        "meta": meta,
        "counts": {
            "ghost_gear": count_ghost,
            "manmade_object": count_manmade,
            "anomaly": count_anomaly,
            "total": len(detections),
        },
    }

    return jsonify({
        "success": True,
        "files": file_dict,
        "detections": detections,
        "counts": LAST_ANALYSIS["counts"],
        "meta": meta,
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

    LAST_ANALYSIS = {
        "files": file_dict,
        "detections": detections,
        "meta": meta,
        "counts": {
            "ghost_gear": count_ghost,
            "manmade_object": count_manmade,
            "anomaly": count_anomaly,
            "total": len(detections),
        },
    }

    return jsonify({
        "success": True,
        "files": file_dict,
        "detections": detections,
        "counts": LAST_ANALYSIS["counts"],
        "meta": meta,
        "latency": elapsed_ms,
    })


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
