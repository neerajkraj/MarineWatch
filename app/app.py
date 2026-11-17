"""
MarineWatch — Interactive Sonar Dashboard (Option 1 & Option 2: Multi-View Anomaly Heatmap)

Features:
- Side-by-side / Multi-tabbed views:
    Tab 1: Combined Dual-Module Fused View (Bounding Boxes + Thermal Anomaly Heatmap)
    Tab 2: Module A YOLO Detections
    Tab 3: Module B Acoustic Anomaly Heatmap (Thermal Overlay)
    Tab 4: Raw Sonar Imagery
- Real-time confidence & peak anomaly metrics
- Downloadable Mission CSV Report
- Fast CPU inference (<150 ms per tile)
"""

import os
import sys
import shutil
from pathlib import Path
from flask import Flask, request, render_template_string, send_from_directory, redirect, url_for, send_file

# Add src to python path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
from infer import MarineWatchPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "outputs" / "dashboard_uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOADS_DIR)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

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
    <title>MarineWatch — Sonar Debris & Anomaly Detector</title>
    <style>
        :root {
            --bg: #090e17;
            --card-bg: #111a2e;
            --border: #1f2f4e;
            --accent: #00d2ff;
            --ghost: #ff4757;
            --manmade: #ffa502;
            --anomaly: #00e676;
            --text: #e2e8f0;
            --text-muted: #8c9cb8;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 24px;
        }
        .container {
            max-width: 1240px;
            margin: 0 auto;
        }
        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border);
            padding-bottom: 18px;
            margin-bottom: 24px;
        }
        h1 {
            margin: 0;
            font-size: 24px;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .tagline {
            color: var(--text-muted);
            font-size: 13.5px;
            margin-top: 4px;
        }
        .badge-cpu {
            background: rgba(0, 210, 255, 0.12);
            color: var(--accent);
            border: 1px solid var(--accent);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        .upload-section {
            background: var(--card-bg);
            border: 2px dashed var(--border);
            border-radius: 10px;
            padding: 24px;
            text-align: center;
            margin-bottom: 24px;
        }
        input[type="file"] {
            display: none;
        }
        .upload-btn {
            background: #192642;
            color: #fff;
            padding: 10px 22px;
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
            transition: opacity 0.2s;
        }
        .submit-btn:hover {
            opacity: 0.9;
        }
        .sample-links {
            margin-top: 14px;
            font-size: 13px;
            color: var(--text-muted);
        }
        .sample-links a {
            color: var(--accent);
            text-decoration: none;
            margin: 0 8px;
            font-weight: 500;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
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
            font-size: 32px;
            font-weight: bold;
            margin: 6px 0;
        }
        .stat-label {
            font-size: 13px;
            color: var(--text-muted);
        }
        /* Tab Navigation */
        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
        }
        .tab-btn {
            background: transparent;
            color: var(--text-muted);
            border: 1px solid transparent;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.2s;
        }
        .tab-btn.active {
            background: #1c2c4d;
            color: var(--accent);
            border-color: var(--border);
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .view-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
            text-align: center;
            margin-bottom: 24px;
        }
        .view-card img {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            border: 1px solid #000;
        }
        .view-caption {
            font-size: 13.5px;
            color: var(--text-muted);
            margin-top: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
            margin-bottom: 24px;
        }
        th, td {
            padding: 12px 18px;
            text-align: left;
            font-size: 13.5px;
        }
        th {
            background: #14223d;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border);
        }
        tr:not(:last-child) td {
            border-bottom: 1px solid var(--border);
        }
        .badge-ghost { color: var(--ghost); font-weight: bold; }
        .badge-manmade { color: var(--manmade); font-weight: bold; }
        .badge-anomaly { color: var(--anomaly); font-weight: bold; }
        .download-btn {
            background: #1c2c4d;
            color: var(--accent);
            border: 1px solid var(--border);
            padding: 8px 16px;
            border-radius: 6px;
            text-decoration: none;
            display: inline-block;
            font-size: 13px;
            font-weight: 600;
            margin-top: 10px;
        }
        .download-btn:hover {
            background: var(--accent);
            color: var(--bg);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>🌊 MarineWatch — Sonar Debris & Anomaly Detector</h1>
                <div class="tagline">Dual-Module Fusion: Supervised YOLO-nano + Unsupervised Thermal Anomaly Heatmap</div>
            </div>
            <span class="badge-cpu">CPU Accelerated (Core i3 Edition)</span>
        </header>

        <div class="upload-section">
            <form method="POST" action="/analyze" enctype="multipart/form-data">
                <label class="upload-btn" for="sonar_file">Choose Sonar Image...</label>
                <input type="file" id="sonar_file" name="sonar_file" accept=".jpg,.jpeg,.png,.bmp" onchange="document.getElementById('file-chosen').textContent = this.files[0].name">
                <span id="file-chosen" style="margin-left: 12px; color: var(--accent);">No file selected</span>
                <button type="submit" class="submit-btn">Run Detection & Heatmap</button>
            </form>
            <div class="sample-links">
                Quick Sonar Samples: 
                <a href="/sample?name=test_sample_1.jpg">Test Sample 1 (Contacts)</a> | 
                <a href="/sample?name=test_sample_2.jpg">Test Sample 2 (Ghost Gear)</a>
            </div>
        </div>

        {% if analyzed %}
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Flagged Returns</div>
                <div class="stat-number" style="color: var(--accent);">{{ detections|length }}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Ghost Gear / Pots (Module A)</div>
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

        <!-- Multi-View Tabs -->
        <div class="tabs">
            <button class="tab-btn active" onclick="openTab(event, 'tab-fused')">🔀 Fused Dual-View</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-heatmap')">🔥 Thermal Anomaly Heatmap</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-boxes')">📦 YOLO Bounding Boxes</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-raw')">📷 Raw Sonar Image</button>
        </div>

        <div id="tab-fused" class="tab-content active">
            <div class="view-card">
                <img src="/files/{{ files.fused }}" alt="Fused View">
                <div class="view-caption">Combined Overlay: YOLO Bounding Boxes layered over the Autoencoder Acoustic Heatmap</div>
            </div>
        </div>

        <div id="tab-heatmap" class="tab-content">
            <div class="view-card">
                <img src="/files/{{ files.heatmap }}" alt="Thermal Heatmap">
                <div class="view-caption">Module B Thermal Map: Red/Yellow highlights abnormal seabed texturing and acoustic shadow disruptions</div>
            </div>
        </div>

        <div id="tab-boxes" class="tab-content">
            <div class="view-card">
                <img src="/files/{{ files.annotated }}" alt="YOLO BBoxes">
                <div class="view-caption">Module A Bounding Boxes: Red for Ghost Gear, Orange for Manmade Contacts, Cyan for Unsupervised Anomalies</div>
            </div>
        </div>

        <div id="tab-raw" class="tab-content">
            <div class="view-card">
                <img src="/files/{{ raw_filename }}" alt="Raw Sonar">
                <div class="view-caption">Original Side-Scan Sonar Imagery (Grayscale Acoustic Backscatter)</div>
            </div>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <h3 style="margin: 0;">Detection & Anomaly Log</h3>
            <a href="/download_report" class="download-btn">⬇️ Download report.csv</a>
        </div>

        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Detection Source</th>
                    <th>Classification</th>
                    <th>Confidence / Peak Score</th>
                    <th>Coordinates (x1, y1, x2, y2)</th>
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
                    <td colspan="5" style="text-align: center; color: var(--text-muted); padding: 24px;">
                        No targets or abnormal disturbances detected above threshold. Normal seafloor condition.
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}
    </div>

    <script>
        function openTab(evt, tabId) {
            var contents = document.getElementsByClassName("tab-content");
            for (var i = 0; i < contents.length; i++) {
                contents[i].classList.remove("active");
            }
            var buttons = document.getElementsByClassName("tab-btn");
            for (var i = 0; i < buttons.length; i++) {
                buttons[i].classList.remove("active");
            }
            document.getElementById(tabId).classList.add("active");
            evt.currentTarget.classList.add("active");
        }
    </script>
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
    bbox_img, detections, files = pipeline.run_detection(raw_path, output_dir=UPLOADS_DIR)

    count_ghost = sum(1 for d in detections if d["label"] == "ghost_gear")
    count_manmade = sum(1 for d in detections if d["label"] == "manmade_object")
    count_anomaly = sum(1 for d in detections if d["label"] == "anomaly")

    return render_template_string(
        HTML_TEMPLATE,
        analyzed=True,
        raw_filename=file.filename,
        files=files,
        detections=detections,
        count_ghost=count_ghost,
        count_manmade=count_manmade,
        count_anomaly=count_anomaly,
    )


@app.route("/sample", methods=["GET"])
def sample():
    test_dir = BASE_DIR / "data" / "processed" / "sss" / "images" / "test"
    test_files = sorted(list(test_dir.glob("*.jpg")))
    if not test_files:
        return "No test samples available. Run build_dataset.py first.", 404

    target_sample = test_files[0] if "1" in request.args.get("name", "1") else test_files[-1]
    dest_path = UPLOADS_DIR / target_sample.name
    shutil.copy(target_sample, dest_path)

    pipeline = get_pipeline()
    bbox_img, detections, files = pipeline.run_detection(dest_path, output_dir=UPLOADS_DIR)

    count_ghost = sum(1 for d in detections if d["label"] == "ghost_gear")
    count_manmade = sum(1 for d in detections if d["label"] == "manmade_object")
    count_anomaly = sum(1 for d in detections if d["label"] == "anomaly")

    return render_template_string(
        HTML_TEMPLATE,
        analyzed=True,
        raw_filename=target_sample.name,
        files=files,
        detections=detections,
        count_ghost=count_ghost,
        count_manmade=count_manmade,
        count_anomaly=count_anomaly,
    )


@app.route("/download_report", methods=["GET"])
def download_report():
    report_file = BASE_DIR / "outputs" / "report.csv"
    if report_file.exists():
        return send_file(str(report_file), as_attachment=True, download_name="marinewatch_report.csv")
    return "Report not generated yet.", 404


@app.route("/files/<path:filename>")
def serve_file(filename):
    return send_from_directory(str(UPLOADS_DIR), filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[MarineWatch Dashboard] Starting server at http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
