"""Local/cloud dashboard and authenticated camera-ingest API."""

from __future__ import annotations

import argparse
import atexit
import csv
import io
import json
import logging
import time
from functools import wraps
from typing import Any, Callable

import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

from . import __version__
from .comparison import DualCameraCoordinator, infer_camera_id
from .config import AppConfig
from .dashboard import DUAL_DASHBOARD
from .operator_dashboard import OPERATOR_DASHBOARD
from .pipeline import VisionPipeline
from .storage import BucketSync
from .streaming import LatestFrameProcessor
from .types import CameraIntrinsics


LOGGER = logging.getLogger(__name__)


DASHBOARD = r"""
<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Local Life | Waste Plant Monitor</title>
<style>
:root{color-scheme:dark;--bg:#091116;--panel:#111d23;--line:#263740;--text:#edf4f5;--muted:#92a9af;--accent:#66e4be;--warning:#ffc470}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top right,#12312d 0,#091116 42%);font:15px/1.5 system-ui,sans-serif;color:var(--text)}
main{max-width:1260px;margin:auto;padding:28px}header{display:flex;justify-content:space-between;align-items:center;gap:15px}h1{font-size:30px;margin:0}h2{font-size:17px;margin:0 0 12px}.muted{color:var(--muted)}
.badge{padding:7px 12px;border:1px solid var(--line);border-radius:20px}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:24px 0}.card{background:rgba(17,29,35,.94);border:1px solid var(--line);border-radius:15px;padding:15px}.metric .label{font-size:12px;color:var(--muted)}.metric .value{font-size:25px;font-weight:750;margin-top:5px}.grid{display:grid;grid-template-columns:1.25fr .75fr;gap:14px}.depth-grid,.report-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}.preview{width:100%;min-height:240px;background:#060a0d;border-radius:10px}.depth-preview{min-height:170px}.actions{display:flex;flex-wrap:wrap;gap:9px;margin-top:14px}button,.export{cursor:pointer;border:0;border-radius:9px;padding:11px 14px;font-weight:700;background:#233840;color:var(--text);text-decoration:none}button.primary{background:var(--accent);color:#073025}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:9px 5px;border-bottom:1px solid var(--line);font-size:13px}th{color:var(--muted)}.warnings{color:var(--warning)}.ready{color:var(--accent)}.history-card{margin-top:14px;overflow:auto}.color-dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:7px;border:1px solid #799}code{font-size:12px;color:var(--muted)}@media(max-width:850px){header,.grid,.depth-grid,.report-grid{display:block}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.grid>.card+ .card,.depth-grid>.card+ .card,.report-grid>.card+ .card{margin-top:14px}}
</style></head><body><main>
<header><div><h1>Waste Plant Monitoring Station</h1><div class="muted">Bags and boxes · automatic settled deposits · color-based waste streams</div></div><div class="badge" id="runtime">Waiting for GPU information</div></header>
<section class="metrics"><div class="card metric"><div class="label">BAGS OBSERVED</div><div class="value" id="bags-seen">0</div></div><div class="card metric"><div class="label">BOXES OBSERVED</div><div class="value" id="boxes-seen">0</div></div><div class="card metric"><div class="label">BAGS DEPOSITED</div><div class="value" id="bags-deposited">0</div></div><div class="card metric"><div class="label">BOXES DEPOSITED</div><div class="value" id="boxes-deposited">0</div></div><div class="card metric"><div class="label">CUMULATIVE VOLUME</div><div class="value" id="cumulative">0.000 L</div></div><div class="card metric"><div class="label">CURRENT ITEM VOLUME</div><div class="value" id="hardware">—</div></div><div class="card metric"><div class="label">TOTAL BIN OCCUPANCY</div><div class="value" id="occupancy">—</div></div><div class="card metric"><div class="label">VALID DEPTH PIXELS</div><div class="value" id="coverage">—</div></div></section>
<section class="grid"><article class="card"><h2>Overhead waste-bin camera</h2><img class="preview" src="/video/latest" alt="Processed waste camera stream"><div class="actions"><button class="primary" onclick="act('/api/baseline')">Capture empty-bin baseline</button><button onclick="act('/api/commit')">Commit current item</button><button onclick="act('/api/accept')">Validate measurement</button><button onclick="act('/api/reset')">Reset live tracking</button><a class="export" href="/api/history.csv">Download history CSV</a></div><p id="volume-status"></p><p class="warnings" id="warnings"></p></article><article class="card"><h2>Currently visible bags and boxes</h2><table><thead><tr><th>ID</th><th>Object</th><th>Color</th><th>Depth</th><th>Height</th><th>Volume</th></tr></thead><tbody id="rows"><tr><td colspan="6">Waiting for the Raspberry Pi</td></tr></tbody></table><p class="muted" id="status"></p><code id="models"></code></article></section>
<section class="report-grid"><article class="card"><h2>Bag and box totals by color</h2><table><thead><tr><th>Color</th><th>Waste stream</th><th>Seen</th><th>Deposited</th><th>Volume</th></tr></thead><tbody id="color-rows"></tbody></table></article><article class="card"><h2>Waste-stream totals</h2><table><thead><tr><th>Configured stream</th><th>Deposited</th><th>Volume</th></tr></thead><tbody id="stream-rows"></tbody></table><p class="muted">Waste streams are shown only when the plant configures its actual bag-color meanings.</p></article></section>
<section class="card history-card"><h2>Running history of observed bags and boxes</h2><table><thead><tr><th>Time</th><th>ID</th><th>Type</th><th>Detected object</th><th>Color</th><th>Waste stream</th><th>Volume</th><th>Status</th></tr></thead><tbody id="history-rows"></tbody></table></section>
<section class="depth-grid"><article class="card"><h2>RealSense aligned depth map</h2><img class="preview depth-preview" src="/video/depth/realsense" alt="RealSense hardware depth"><p class="muted" id="hardware-depth-status">Waiting for RealSense depth</p></article><article class="card"><h2>Depth Anything estimated depth map</h2><img class="preview depth-preview" src="/video/depth/monocular" alt="AI-estimated monocular depth"><p class="muted" id="monocular-depth-status">Waiting for depth-model predictions</p></article></section>
<script>
const node=id=>document.getElementById(id);const fmt=x=>x==null?'—':x.toFixed(3)+' L';
async function act(path){const response=await fetch(path,{method:'POST'});const data=await response.json();if(!response.ok)alert(data.error||'Request failed');else if(path==='/api/baseline'){const signal=data.baseline.realsense_depth_signal||{};alert(signal.valid_pixels?'Empty-bin baseline captured from '+data.baseline.baseline_frame_count+' depth frame(s).':'Baseline captured, but RealSense depth is missing.');}else if(path==='/api/commit')alert('Current item recorded. The next deposit will be measured against the updated bin contents.');}
function signalSummary(signal){if(!signal||!signal.available)return 'No depth signal received';if(!signal.valid_pixels)return 'Depth frames received, but all pixels are invalid';return signal.valid_percent.toFixed(1)+'% valid pixels | median '+signal.median_m.toFixed(2)+' m | range '+signal.min_m.toFixed(2)+'–'+signal.max_m.toFixed(2)+' m';}
function tableRow(target,values){const tr=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=String(value??'—');tr.appendChild(td);}node(target).appendChild(tr);}
async function update(){try{const response=await fetch('/api/state');const state=await response.json();const result=state.latest||{};const hardware=state.realsense_depth||{};const mono=state.monocular_depth||{};const readiness=state.volume_status||{};const plant=state.plant||{};node('bags-seen').textContent=plant.observed_bags||0;node('boxes-seen').textContent=plant.observed_boxes||0;node('bags-deposited').textContent=plant.deposited_bags||0;node('boxes-deposited').textContent=plant.deposited_boxes||0;node('cumulative').textContent=fmt(plant.cumulative_volume_l||0);node('hardware').textContent=fmt(result.realsense_volume_l);node('occupancy').textContent=fmt(result.bin_total_volume_l);node('coverage').textContent=hardware.available?hardware.valid_percent.toFixed(0)+'%':'none';node('runtime').textContent=state.runtime.gpu_name||state.runtime.device||'GPU initializes on first frame';node('warnings').textContent=(result.warnings||[]).join(' • ');node('volume-status').className=readiness.ready?'ready':'warnings';node('volume-status').textContent=readiness.message||'';node('hardware-depth-status').textContent=signalSummary(hardware);node('monocular-depth-status').textContent=signalSummary(mono)+(mono.available?(state.monocular_calibrated?' | calibrated':' | not calibrated'):'');node('status').textContent='Frames: '+state.frames_processed+' | Baseline frames: '+(state.baseline_frame_count||0)+' | Auto-deposit: '+(state.auto_deposit?'on':'off')+' | Intrinsics: '+(state.camera_intrinsics_ready?'ready':'missing');node('models').textContent=state.detector_model+' | '+(state.depth_model||'depth disabled');for(const id of ['rows','color-rows','stream-rows','history-rows'])node(id).replaceChildren();for(const item of result.detections||[]){const depth=item.depth_distance_m!=null?item.depth_distance_m.toFixed(2)+' m':'—';const height=item.height_above_baseline_cm!=null?item.height_above_baseline_cm.toFixed(1)+' cm':'—';const volume=item.volume_uncertainty_l!=null?fmt(item.realsense_volume_l)+' ± '+item.volume_uncertainty_l.toFixed(3)+' L':fmt(item.realsense_volume_l);tableRow('rows',[item.track_id??'—',item.label,item.color,depth,height,volume]);}if(!(result.detections||[]).length)tableRow('rows',['Waiting for the next bag or box','','','','','']);for(const item of plant.colors||[])tableRow('color-rows',[item.color,item.waste_stream||'not configured',item.observed_count,item.deposited_count,fmt(item.volume_l)]);for(const item of plant.waste_streams||[])tableRow('stream-rows',[item.waste_stream,item.deposited_count,fmt(item.volume_l)]);if(!(plant.waste_streams||[]).length)tableRow('stream-rows',['No color mappings configured','','']);for(const item of plant.history||[])tableRow('history-rows',[new Date(item.observed_at*1000).toLocaleTimeString(),item.track_id,item.object_type,item.label,item.color,item.waste_stream||'not configured',fmt(item.volume_l),item.status]);}catch(error){node('warnings').textContent='Dashboard cannot reach the cloud service.';}}setInterval(update,550);update();
</script></main></body></html>
"""


def _annotate_frame(frame: np.ndarray, pipeline: VisionPipeline) -> np.ndarray:
    import cv2

    output = frame.copy()
    latest = pipeline.latest_analysis
    if latest is None:
        return output
    for detection in latest.detections:
        track = pipeline.tracker.tracks.get(detection.track_id) if detection.track_id is not None else None
        if detection.source != "tracked-prediction" and track is not None and not track.counted:
            # Tentative one-frame text-prompt hits must not flicker as if they
            # were confirmed waste objects.
            continue
        color = (66, 228, 190) if detection.source.startswith("yolo") else (90, 184, 250)
        if detection.mask is not None and detection.mask.shape == output.shape[:2]:
            overlay = output.copy()
            overlay[detection.mask] = color
            output = cv2.addWeighted(overlay, 0.27, output, 0.73, 0)
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        distance = (
            f" | {detection.depth_distance_m:.2f} m"
            if detection.depth_distance_m is not None
            else f" | AI {detection.monocular_distance_m:.2f} m"
            if detection.monocular_distance_m is not None
            else ""
        )
        height = (
            f" | {detection.height_above_baseline_cm:.1f} cm"
            if detection.height_above_baseline_cm is not None
            else ""
        )
        active_volume = (
            detection.monocular_volume_l if pipeline.camera_id == "logitech"
            else detection.realsense_volume_l
        )
        volume = f" | {active_volume:.2f} L" if active_volume is not None else ""
        visible_color = f" [{detection.color}]" if detection.color not in {"unknown", ""} else ""
        continuity = " (tracked)" if detection.source == "tracked-prediction" else ""
        display_type = (
            f"test object ({detection.label})"
            if detection.accepted_class == "measurement_object"
            else detection.accepted_class.replace("_", " ")
            if detection.accepted_class is not None else detection.label
        )
        caption = f"#{detection.track_id or '?'} {display_type}{continuity}{visible_color}{distance}{height}{volume}"
        cv2.putText(output, caption, (x1, max(19, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if (
            detection.footprint_length_mm is not None
            and detection.footprint_width_mm is not None
            and detection.physical_height_mm is not None
        ):
            dimensions = (
                f"LxWxH {detection.footprint_length_mm:.0f}x"
                f"{detection.footprint_width_mm:.0f}x{detection.physical_height_mm:.0f} mm"
            )
            dimension_y = min(output.shape[0] - 8, max(20, y1 + 20))
            cv2.putText(
                output, dimensions, (x1, dimension_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2,
            )
    return output


def _render_depth_map(depth_m: np.ndarray | None, label: str) -> np.ndarray:
    """Render metric depth without implying missing/invalid pixels are real."""
    import cv2

    if depth_m is None or depth_m.ndim != 2:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(image, f"WAITING FOR {label.upper()}", (32, 240), cv2.FONT_HERSHEY_SIMPLEX, .70, (180, 200, 200), 2)
        return image

    valid = np.isfinite(depth_m) & (depth_m > 0.10) & (depth_m < 20.0)
    if not np.any(valid):
        image = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
        cv2.putText(image, "NO VALID DEPTH PIXELS", (24, max(35, depth_m.shape[0] // 2)), cv2.FONT_HERSHEY_SIMPLEX, .70, (95, 150, 255), 2)
        return image

    values = depth_m[valid]
    near = float(np.percentile(values, 3))
    far = float(np.percentile(values, 97))
    if far - near < 0.03:
        far = near + 0.03
    safe = np.where(valid, depth_m, far)
    normalized = np.clip((safe - near) / (far - near), 0.0, 1.0)
    intensity = np.asarray((1.0 - normalized) * 255.0, dtype=np.uint8)
    image = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
    image[~valid] = (8, 12, 15)
    cv2.putText(
        image,
        f"{label} | median {np.median(values):.2f} m | {100 * values.size / depth_m.size:.0f}% valid",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
    )
    return image


def create_app(
    config: AppConfig | None = None,
    pipeline: VisionPipeline | DualCameraCoordinator | None = None,
) -> Flask:
    settings = config or AppConfig.from_env()
    settings.validate()
    if isinstance(pipeline, DualCameraCoordinator):
        manager = pipeline
    elif isinstance(pipeline, VisionPipeline):
        manager = DualCameraCoordinator(settings, detector=pipeline.detector, depth_estimator=pipeline.depth_estimator)
        manager.pipelines["realsense"] = pipeline
    else:
        manager = DualCameraCoordinator(settings)
    vision = manager.camera("realsense")
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_mb * 1024 * 1024
    app.config["VISION_PIPELINE"] = vision
    app.config["CAMERA_COORDINATOR"] = manager
    app.config["LOCALLIFE_CONFIG"] = settings
    frame_processor = LatestFrameProcessor(manager)
    app.extensions["locallife_frame_processor"] = frame_processor
    atexit.register(frame_processor.stop)

    def protected(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if settings.api_token:
                authorization = request.headers.get("Authorization", "")
                token = request.headers.get("X-API-Token", "")
                if authorization != f"Bearer {settings.api_token}" and token != settings.api_token:
                    return jsonify(error="Invalid or missing API token"), 401
            return function(*args, **kwargs)

        return wrapper

    @app.errorhandler(413)
    def payload_too_large(_: Any) -> Any:
        return jsonify(error=f"Upload exceeds {settings.max_upload_mb} MB"), 413

    @app.get("/")
    def index() -> str:
        """Operator page (Accuracy Deployment v3.0) -- the default landing page."""
        return render_template_string(OPERATOR_DASHBOARD, api_token=settings.api_token)

    @app.get("/research")
    def research_dashboard() -> str:
        # The research page's own POST calls (baseline/reset/calibrate/
        # reference-distance) hit @protected endpoints; when
        # LOCALLIFE_API_TOKEN is set (required whenever --host binds outside
        # localhost, e.g. the normal 0.0.0.0 launcher run) those calls need the
        # token too, so it is embedded into the page here and attached by the
        # JS fetch calls.
        return render_template_string(DUAL_DASHBOARD, api_token=settings.api_token)

    @app.get("/api/color-map")
    def get_color_map() -> Any:
        return jsonify(mapping=dict(manager.config.color_waste_streams))

    @app.post("/api/color-map")
    @protected
    def set_color_map() -> Any:
        payload = request.get_json(silent=True) or {}
        mapping = payload.get("mapping")
        if not isinstance(mapping, dict):
            return jsonify(
                error="Provide mapping as a JSON object of color to content label"
            ), 400
        return jsonify(ok=True, **manager.update_color_map(mapping))

    @app.post("/api/operator/session/new")
    @protected
    def new_operator_session() -> Any:
        try:
            return jsonify(ok=True, session=manager.start_new_operator_session())
        except (OSError, ValueError) as exc:
            return jsonify(error=str(exc)), 400

    @app.get("/health")
    def health() -> Any:
        return jsonify(status="ok", version=__version__, frames_processed=manager.frames_processed,
                       cameras={name: station.frames_processed for name, station in manager.pipelines.items()},
                       transport=frame_processor.snapshot())

    @app.get("/api/state")
    def state() -> Any:
        payload = manager.state()
        transport = frame_processor.snapshot()
        payload["transport"] = transport
        for camera_id, stats in transport.items():
            payload["cameras"][camera_id]["stream"].update(stats)
        return jsonify(payload)

    @app.get("/api/cameras/<camera_id>/state")
    def camera_state(camera_id: str) -> Any:
        try:
            return jsonify(manager.camera(camera_id).state())
        except ValueError as exc:
            return jsonify(error=str(exc)), 404

    @app.get("/api/history")
    @app.get("/api/cameras/<camera_id>/history")
    def history(camera_id: str = "realsense") -> Any:
        try:
            return jsonify(manager.camera(camera_id).ledger.summary())
        except ValueError as exc:
            return jsonify(error=str(exc)), 404

    @app.get("/api/history.csv")
    @app.get("/api/cameras/<camera_id>/history.csv")
    def history_csv(camera_id: str = "realsense") -> Response | tuple[Any, int]:
        try:
            station = manager.camera(camera_id)
        except ValueError as exc:
            return jsonify(error=str(exc)), 404
        output = io.StringIO()
        columns = [
            "camera_id", "entry_id", "track_id", "object_type", "label", "color", "waste_stream",
            "volume_l", "volume_uncertainty_l", "depth_coverage_percent", "confidence",
            "measurement_method", "measurement_quality", "calibration_mode",
            "observed_at", "deposited_at", "status",
        ]
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(station.ledger.all_records())
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={camera_id}_waste_plant_history.csv"},
        )

    @app.get("/api/measurements.csv")
    @app.get("/api/cameras/<camera_id>/measurements.csv")
    def measurements_csv(camera_id: str = "realsense") -> Response | tuple[Any, int]:
        """The per-object measurement log, in every operating mode.

        The waste-ledger export below is empty in geometry_validation mode
        because that mode deliberately disables the ledger; this file is written
        by the pipeline itself for every finalised object, so it is what an
        operator should download. A run with no completed measurements yet
        returns the header alone rather than an empty file.
        """
        try:
            station = manager.camera(camera_id)
        except ValueError as exc:
            return jsonify(error=str(exc)), 404
        # Served straight from the event log the pipeline writes through, so the
        # route cannot drift onto a different path than the writer -- the defect
        # class that made this download look broken when the real fault was
        # upstream.
        return Response(
            station.event_log.csv_text(),
            # content_type, not mimetype: Flask appends its own charset to a
            # mimetype, which produced the malformed
            # "text/csv; charset=utf-8; charset=utf-8" this route used to send.
            content_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition":
                    f"attachment; filename={camera_id}_measurements.csv",
                # Lets the operator page show where the file actually is without
                # a second round trip.
                "X-LocalLife-CSV-Path": str(station.event_log.path),
            },
        )

    @app.post("/api/cameras/<camera_id>/measurements/retry")
    @protected
    def retry_measurement_persistence(camera_id: str = "realsense") -> Any:
        """Re-attempt rows that failed to reach disk.

        A persistence failure is shown on the operator page rather than
        swallowed, so it needs somewhere to be retried from.
        """
        try:
            station = manager.camera(camera_id)
        except ValueError as exc:
            return jsonify(error=str(exc)), 404
        return jsonify(ok=True, result=station.event_log.retry_failed(),
                       status=station.event_log.status())

    @app.post("/api/cameras/<camera_id>/measurements/ground-truth")
    @protected
    def set_measurement_ground_truth(camera_id: str = "realsense") -> Any:
        """Attach an operator-entered true volume to an already-written event."""
        try:
            station = manager.camera(camera_id)
        except ValueError as exc:
            return jsonify(error=str(exc)), 404
        payload = request.get_json(silent=True) or {}
        event_id = payload.get("event_id")
        litres = payload.get("litres")
        if not isinstance(event_id, str) or not event_id.strip():
            return jsonify(error="Provide event_id as a string"), 400
        try:
            litres_value = float(litres)
        except (TypeError, ValueError):
            return jsonify(error="Provide litres as a number"), 400
        if litres_value < 0:
            return jsonify(error="Ground truth volume cannot be negative"), 400
        if not station.event_log.set_ground_truth(event_id.strip(), litres_value):
            return jsonify(error=f"No recorded event {event_id}"), 404
        return jsonify(ok=True, event_id=event_id.strip(), litres=litres_value)

    @app.get("/api/benchmark")
    def benchmark_state() -> Any:
        """Local-versus-cloud comparison, or an explanation of what is missing."""
        return jsonify(manager.benchmark_summary())

    @app.get("/api/benchmark.csv")
    def benchmark_csv() -> Response:
        body = manager.benchmark_csv()
        return Response(
            body,
            content_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=benchmark.csv"},
        )

    @app.get("/api/comparison")
    def comparison() -> Any:
        return jsonify(manager.comparison())

    @app.get("/api/comparison.csv")
    def comparison_csv() -> Response:
        output = io.StringIO()
        columns = ["pair_id", "object_type", "realsense_color", "logitech_color", "color_agreement",
                   "realsense_volume_l", "logitech_volume_l", "difference_l", "absolute_difference_l",
                   "difference_percent", "timestamp_gap_seconds", "realsense_measurement_method",
                   "logitech_measurement_method", "realsense_calibration_mode", "logitech_calibration_mode",
                   "realsense_entry_id", "logitech_entry_id"]
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manager.pairs())
        return Response(output.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=dual_camera_thesis_comparison.csv"})

    @app.post("/api/comparison/reference")
    @protected
    def add_reference() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            record = manager.record_reference(float(payload["known_liters"]),
                                              pair_id=payload.get("pair_id"), notes=str(payload.get("notes", "")))
            return jsonify(ok=True, trial=record, comparison=manager.comparison())
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400

    @app.post("/api/ingest")
    @app.post("/api/cameras/<camera_id>/ingest")
    @protected
    def ingest(camera_id: str | None = None) -> Any:
        uploaded = request.files.get("image")
        if uploaded is None:
            return jsonify(error="Multipart field 'image' is required"), 400
        encoded = np.frombuffer(uploaded.read(), dtype=np.uint8)
        try:
            import cv2

            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        except ImportError:
            from PIL import Image

            try:
                rgb = np.asarray(Image.open(io.BytesIO(encoded.tobytes())).convert("RGB"))
                frame = rgb[:, :, ::-1].copy()
            except (OSError, ValueError):
                frame = None
        if frame is None:
            return jsonify(error="The uploaded image could not be decoded"), 400

        try:
            metadata = json.loads(request.form.get("metadata", "{}"))
            source = str(metadata.get("source", "edge"))
            target_id = infer_camera_id(source, camera_id or metadata.get("camera_id"))
            station = manager.camera(target_id)
            origin = str(metadata.get("intrinsics_origin", "")).strip()
            intrinsics = CameraIntrinsics.from_dict(metadata.get("intrinsics"))
            uploaded_depth = request.files.get("depth")
            depth = None
            if uploaded_depth is not None:
                with np.load(io.BytesIO(uploaded_depth.read()), allow_pickle=False) as archive:
                    depth = archive["depth_m"].astype(np.float32)
            timestamp = float(metadata.get("timestamp", time.time()))
            station.update_preview(
                frame,
                depth_m=depth,
                intrinsics=intrinsics,
                timestamp=timestamp,
                intrinsics_origin=origin,
            )
            packet = {
                "frame": frame,
                "depth_m": depth,
                "intrinsics": intrinsics,
                "source": source,
                "timestamp": timestamp,
            }
            if request.args.get("sync", "").strip().lower() in {"1", "true", "yes"}:
                result = station.process_frame(**packet)
            else:
                queue = frame_processor.submit(target_id, packet)
                latest = None if station.latest_analysis is None else station.latest_analysis.to_dict()
                if latest is None:
                    latest = {
                        "visible_objects": 0,
                        "automatic_count": station.tracker.total_count,
                        "realsense_volume_l": None,
                        "monocular_volume_l": None,
                        "inference_ms": None,
                    }
                return jsonify(camera_id=target_id, accepted=True, transport=queue, **latest), 202
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            return jsonify(error=str(exc)), 400
        except Exception:
            LOGGER.exception("Cloud inference failed")
            return jsonify(error="Cloud inference failed; inspect the server logs"), 500
        return jsonify(camera_id=target_id, **result.to_dict())

    @app.post("/api/setup/auto")
    @protected
    def auto_setup() -> Any:
        """Capture both empty-scene profiles with one guided dashboard action."""
        payload = request.get_json(silent=True) or {}
        raw_distance = payload.get("logitech_distance_m")
        distance: float | None = None
        if raw_distance is not None and raw_distance != "":
            try:
                distance = float(raw_distance)
            except (TypeError, ValueError):
                return jsonify(error="Logitech distance must be entered in meters"), 400
            if not np.isfinite(distance) or distance <= 0:
                return jsonify(error="Logitech distance must be a finite positive number of meters"), 400
        try:
            result = manager.automatic_empty_setup(distance)
        except ValueError as exc:
            return jsonify(error=str(exc)), 409
        return jsonify(ok=True, **result)

    @app.post("/api/baseline")
    @app.post("/api/cameras/<camera_id>/baseline")
    @protected
    def capture_baseline(camera_id: str = "realsense") -> Any:
        try:
            return jsonify(ok=True, camera_id=camera_id, baseline=manager.camera(camera_id).set_baseline())
        except ValueError as exc:
            return jsonify(error=str(exc)), 409

    @app.post("/api/accept")
    @app.post("/api/cameras/<camera_id>/accept")
    @protected
    def accept(camera_id: str = "realsense") -> Any:
        try:
            return jsonify(ok=True, camera_id=camera_id, record=manager.camera(camera_id).accept_current())
        except ValueError as exc:
            return jsonify(error=str(exc)), 409

    @app.post("/api/commit")
    @app.post("/api/cameras/<camera_id>/commit")
    @protected
    def commit_bag(camera_id: str = "realsense") -> Any:
        try:
            return jsonify(ok=True, camera_id=camera_id, record=manager.camera(camera_id).commit_current_bags())
        except ValueError as exc:
            return jsonify(error=str(exc)), 409

    @app.post("/api/reset")
    @app.post("/api/cameras/<camera_id>/reset")
    @protected
    def reset(camera_id: str = "realsense") -> Any:
        try:
            manager.camera(camera_id).reset_live_tracking()
            return jsonify(ok=True, camera_id=camera_id)
        except ValueError as exc:
            return jsonify(error=str(exc)), 404

    @app.post("/api/cameras/<camera_id>/reset-history")
    @protected
    def reset_camera_history(camera_id: str) -> Any:
        try:
            return jsonify(ok=True, history=manager.camera(camera_id).clear_history())
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @app.post("/api/cameras/<camera_id>/calibrate-volume")
    @protected
    def calibrate_volume(camera_id: str) -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            observed = payload.get("observed_liters")
            calibration = manager.camera(camera_id).calibrate_known_volume(
                float(payload["known_liters"]), None if observed is None else float(observed)
            )
            return jsonify(ok=True, camera_id=camera_id, calibration=calibration)
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400

    @app.post("/api/cameras/logitech/reference-distance")
    @protected
    def set_logitech_distance() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            distance = float(payload["distance_m"])
            if not np.isfinite(distance) or distance <= 0:
                raise ValueError("Measured Logitech reference distance must be a finite positive value")
            station = manager.camera("logitech")
            station.config.logitech_reference_distance_m = distance
            baseline = station.set_baseline()
            station.store.save_json("calibration/reference_distance.json",
                                    {"distance_m": distance, "captured_at": time.time()})
            return jsonify(ok=True, distance_m=distance, baseline=baseline)
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400

    @app.post("/api/roi")
    @app.post("/api/cameras/<camera_id>/roi")
    @protected
    def update_roi(camera_id: str = "realsense") -> Any:
        payload = request.get_json(silent=True) or {}
        values = payload.get("roi")
        if not isinstance(values, list) or len(values) != 4:
            return jsonify(error="Provide roi as [x, y, width, height]"), 400
        try:
            updated = manager.camera(camera_id).update_camera_roi(tuple(float(value) for value in values))
        except (TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(ok=True, **updated)

    @app.get("/video/latest")
    @app.get("/video/cameras/<camera_id>")
    def stream(camera_id: str = "realsense") -> Response | tuple[Any, int]:
        import cv2
        try:
            station = manager.camera(camera_id)
        except ValueError as exc:
            return jsonify(error=str(exc)), 404

        def generate() -> Any:
            while True:
                with station.lock:
                    # Keep overlays and liters tied to the exact analyzed frame.
                    # The raw preview may already be several uploads ahead while
                    # the GPU is processing a large segmentation/depth model.
                    source_frame = (
                        station.latest_processed_frame
                        if station.latest_processed_frame is not None
                        else station.latest_frame
                    )
                    frame = None if source_frame is None else source_frame.copy()
                    if frame is not None:
                        frame = _annotate_frame(frame, station)
                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, f"WAITING FOR {camera_id.upper()} CAMERA", (45, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, .75, (180, 200, 200), 2)
                success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if success:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
                time.sleep(0.15)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/cameras/<camera_id>/raw-snapshot.jpg")
    def raw_snapshot(camera_id: str) -> Response | tuple[Any, int]:
        """One un-annotated JPEG of this camera's most recent frame.

        Unlike `/video/cameras/<camera_id>` (an MJPEG stream with detection
        overlays drawn in), this is a single plain image with nothing drawn
        on it -- exactly what `tools/calibrate_dual_camera.py` needs so a
        checkerboard detector never has to contend with overlay boxes/text,
        and a single request/response is far simpler to keep two cameras'
        captures synchronized against than parsing two live MJPEG streams.
        """
        import cv2

        try:
            station = manager.camera(camera_id)
        except ValueError as exc:
            return jsonify(error=str(exc)), 404
        with station.lock:
            frame = station.latest_frame
            frame = None if frame is None else frame.copy()
        if frame is None:
            return jsonify(error=f"No frame has arrived yet from {camera_id}"), 503
        success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success:
            return jsonify(error="Could not encode the current frame"), 500
        return Response(encoded.tobytes(), mimetype="image/jpeg")

    @app.get("/video/depth/<kind>")
    @app.get("/video/cameras/<camera_id>/depth")
    def depth_stream(kind: str | None = None, camera_id: str | None = None) -> Response | tuple[Any, int]:
        if camera_id is None:
            if kind not in {"realsense", "monocular"}:
                return jsonify(error="Depth kind must be realsense or monocular"), 404
            camera_id = "realsense" if kind == "realsense" else "logitech"
        try:
            station = manager.camera(camera_id)
        except ValueError as exc:
            return jsonify(error=str(exc)), 404
        import cv2

        def generate() -> Any:
            while True:
                with station.lock:
                    current = station.latest_depth if camera_id == "realsense" else station.latest_monocular_depth
                    depth = None if current is None else current.copy()
                label = "RealSense" if camera_id == "realsense" else "Logitech Depth Anything"
                image = _render_depth_map(depth, label)
                success, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if success:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
                time.sleep(0.30)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Local Life processing service")
    parser.add_argument("--host", help="Bind address; defaults to localhost for safe SSH tunneling")
    parser.add_argument("--port", type=int, help="HTTP port")
    parser.add_argument("--skip-warmup", action="store_true", help="Load models when the first image arrives")
    parser.add_argument("--disable-sync", action="store_true", help="Disable background bucket synchronization")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = AppConfig.from_env()
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if config.host not in {"127.0.0.1", "localhost", "::1"} and not config.api_token:
        parser.error("Binding outside localhost requires LOCALLIFE_API_TOKEN")

    vision = DualCameraCoordinator(config)
    if not args.skip_warmup:
        LOGGER.info("Preparing the selected processing engine")
        try:
            LOGGER.info("Runtime ready: %s", vision.warmup())
        except Exception:
            # Per-camera warmup() already falls back to the dependency-free
            # local detector on model-loading failure; this outer guard only
            # protects against an unexpected error escaping that fallback so
            # the one-click dashboard still starts instead of the whole
            # process exiting.
            LOGGER.exception(
                "Warmup did not complete cleanly; continuing with whatever engine "
                "loaded successfully so the dashboard still starts"
            )
    app = create_app(config, vision)

    if config.enable_bucket_sync and not args.disable_sync:
        synchronizer = BucketSync(config.results_dir, config.bucket, config.sync_interval_seconds)
        synchronizer.start()
        atexit.register(synchronizer.stop)

    LOGGER.info("Dashboard: http://%s:%s", config.host, config.port)
    try:
        from waitress import serve

        serve(app, host=config.host, port=config.port, threads=12)
    except ImportError:
        LOGGER.warning("waitress is unavailable; starting Flask's development server")
        app.run(host=config.host, port=config.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
