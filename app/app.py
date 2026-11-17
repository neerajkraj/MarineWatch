"""
MarineWatch — Phase 7: Interactive Flask Dashboard (CPU Friendly)

Web interface for operators to upload side-scan sonar imagery,
run dual-module inference (YOLO-nano + Anomaly Autoencoder),
view annotated detections, and inspect tabular contact reports.
"""

import os
import sys
import shutil
from pathlib import Path
from flask import Flask, request, render_template_string, send_from_directory, redirect, url_for

# Add src to python path for pipeline import
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
from infer import MarineWatchPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "outputs" / "dashboard_uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOADS_DIR)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max

# Lazy-loaded global pipeline
PIPELINE = None


def get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = MarineWatchPipeline()
    return PIPELINE


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MarineWatch — Underwater Debris & Anomaly Detector</title>
    <style>
        :root {
            --bg: #0b1320;
            --card-bg: #131e33;
            --border: #233554;
            --accent: #00d2ff;
            --ghost: #ff4757;
            --manmade: #ffa502;
            --anomaly: #2ed573;
            --text: #e1e7f0;
            --text-muted: #8892b0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 24px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }
        h1 {
            margin: 0;
            font-size: 24px;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .tagline {
            color: var(--text-muted);
            font-size: 14px;
            margin-top: 4px;
        }
        .badge {
            background: #1e3a5f;
            color: var(--accent);
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        .upload-section {
            background: var(--card-bg);
            border: 2px dashed var(--border);
            border-radius: 8px;
            padding: 24px;
            text-align: center;
            margin-bottom: 24px;
        }
        input[type="file"] {
            display: none;
        }
        .upload-btn {
            background: #1e3a5f;
            color: #fff;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            display: inline-block;
            font-weight: 500;
            border: 1px solid var(--accent);
            transition: all 0.2s;
        }
        .upload-btn:hover {
            background: var(--accent);
            color: var(--bg);
        }
        .submit-btn {
            background: var(--accent);
            color: var(--bg);
            border: none;
            padding: 10px 24px;
            border-radius: 6px;
            font-weight: bold;
            cursor: pointer;
            margin-left: 12px;
        }
        .submit-btn:hover {
            opacity: 0.9;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .stat-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }
        .stat-number {
            font-size: 28px;
            font-weight: bold;
            margin: 8px 0;
        }
        .stat-label {
            font-size: 13px;
            color: var(--text-muted);
        }
        .image-preview-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }
        .preview-box {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }
        .preview-box h3 {
            margin-top: 0;
            font-size: 15px;
            color: var(--accent);
        }
        .preview-box img {
            width: 100%;
            height: auto;
            border-radius: 4px;
            border: 1px solid #000;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
        }
        th, td {
            padding: 12px 16px;
            text-align: left;
            font-size: 14px;
        }
        th {
            background: #182844;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border);
        }
        tr:not(:last-child) td {
            border-bottom: 1px solid var(--border);
        }
        .badge-ghost { color: var(--ghost); font-weight: bold; }
        .badge-manmade { color: var(--manmade); font-weight: bold; }
        .badge-anomaly { color: var(--anomaly); font-weight: bold; }
        .sample-links {
            margin-top: 14px;
            font-size: 13px;
            color: var(--text-muted);
        }
        .sample-links a {
            color: var(--accent);
            text-decoration: none;
            margin: 0 6px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>🌊 MarineWatch — Sonar Debris & Anomaly Detector</h1>
                <div class="tagline">Dual-Module CPU Architecture: YOLO-nano (Known Contacts) + Autoencoder (Unsupervised Anomalies)</div>
            </div>
            <span class="badge">CPU-Only Edition (Core i3)</span>
        </header>

        <div class="upload-section">
            <form method="POST" action="/analyze" enctype="multipart/form-data">
                <label class="upload-btn" for="sonar_file">Choose Sonar Image...</label>
                <input type="file" id="sonar_file" name="sonar_file" accept=".jpg,.jpeg,.png,.bmp" onchange="document.getElementById('file-chosen').textContent = this.files[0].name">
                <span id="file-chosen" style="margin-left: 12px; color: var(--accent);">No file selected</span>
                <button type="submit" class="submit-btn">Run Detection</button>
            </form>
            <div class="sample-links">
                Quick Test Samples: 
                <a href="/sample?name=test_sample_1.jpg">Test Sample 1</a> | 
                <a href="/sample?name=test_sample_2.jpg">Test Sample 2</a>
            </div>
        </div>

        {% if analyzed %}
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Flagged Returns</div>
                <div class="stat-number" style="color: var(--accent);">{{ detections|length }}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Ghost Gear (Module A)</div>
                <div class="stat-number" style="color: var(--ghost);">{{ count_ghost }}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Man-Made Contacts (Module A)</div>
                <div class="stat-number" style="color: var(--manmade);">{{ count_manmade }}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Unusual Anomalies (Module B)</div>
                <div class="stat-number" style="color: var(--anomaly);">{{ count_anomaly }}</div>
            </div>
        </div>

        <div class="image-preview-grid">
            <div class="preview-box">
                <h3>Original Sonar Imagery</h3>
                <img src="/files/{{ raw_filename }}" alt="Original Sonar">
            </div>
            <div class="preview-box">
                <h3>Annotated Detections (Module A & B Fusion)</h3>
                <img src="/files/{{ annotated_filename }}" alt="Annotated Sonar">
            </div>
        </div>

        <h3>Detection Report</h3>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Source Module</th>
                    <th>Classification</th>
                    <th>Confidence / Peak Score</th>
                    <th>Bounding Box (x1, y1, x2, y2)</th>
                </tr>
            </thead>
            <tbody>
                {% for d in detections %}
                <tr>
                    <td>{{ loop.index }}</td>
                    <td>{{ d.source }}</td>
                    <td>
                        {% if d.label == 'ghost_gear' %}
                            <span class="badge-ghost">🦀 Ghost Gear / Crab Pot</span>
                        {% elif d.label == 'manmade_object' %}
                            <span class="badge-manmade">⚓ Manmade Contact</span>
                        {% else %}
                            <span class="badge-anomaly">⚠️ Seabed Anomaly</span>
                        {% endif %}
                    </td>
                    <td>{{ "%.3f"|format(d.score) }}</td>
                    <td>({{ d.x1 }}, {{ d.y1 }}, {{ d.x2 }}, {{ d.y2 }})</td>
                </tr>
                {% else %}
                <tr>
                    <td colspan="5" style="text-align: center; color: var(--text-muted);">
                        No anomalous returns or objects detected above threshold. Normal seabed condition.
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, analyzed=False)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "sonar_file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["sonar_file"]
    if file.filename == "":
        return redirect(url_for("index"))

    raw_path = UPLOADS_DIR / file.filename
    file.save(str(raw_path))

    pipeline = get_pipeline()
    annotated_img, detections = pipeline.run_detection(raw_path, output_dir=UPLOADS_DIR)

    annotated_filename = f"annotated_{file.filename}"

    count_ghost = sum(1 for d in detections if d["label"] == "ghost_gear")
    count_manmade = sum(1 for d in detections if d["label"] == "manmade_object")
    count_anomaly = sum(1 for d in detections if d["label"] == "anomaly")

    return render_template_string(
        HTML_TEMPLATE,
        analyzed=True,
        raw_filename=file.filename,
        annotated_filename=annotated_filename,
        detections=detections,
        count_ghost=count_ghost,
        count_manmade=count_manmade,
        count_anomaly=count_anomaly,
    )


@app.route("/sample", methods=["GET"])
def sample():
    # Copy a sample from the test split
    test_dir = BASE_DIR / "data" / "processed" / "sss" / "images" / "test"
    test_files = list(test_dir.glob("*.jpg"))
    if not test_files:
        return "No test sample found. Run build_dataset.py first.", 404

    target_sample = test_files[0] if "1" in request.args.get("name", "1") else test_files[-1]
    dest_path = UPLOADS_DIR / target_sample.name
    shutil.copy(target_sample, dest_path)

    pipeline = get_pipeline()
    annotated_img, detections = pipeline.run_detection(dest_path, output_dir=UPLOADS_DIR)

    annotated_filename = f"annotated_{target_sample.name}"
    count_ghost = sum(1 for d in detections if d["label"] == "ghost_gear")
    count_manmade = sum(1 for d in detections if d["label"] == "manmade_object")
    count_anomaly = sum(1 for d in detections if d["label"] == "anomaly")

    return render_template_string(
        HTML_TEMPLATE,
        analyzed=True,
        raw_filename=target_sample.name,
        annotated_filename=annotated_filename,
        detections=detections,
        count_ghost=count_ghost,
        count_manmade=count_manmade,
        count_anomaly=count_anomaly,
    )


@app.route("/files/<path:filename>")
def serve_file(filename):
    return send_from_directory(str(UPLOADS_DIR), filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[MarineWatch Dashboard] Starting server at http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
