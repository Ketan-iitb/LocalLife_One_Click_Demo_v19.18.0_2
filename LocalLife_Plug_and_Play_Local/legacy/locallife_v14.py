#!/usr/bin/env python3
"""
LocalLife V14 — recreated full codebase

Architecture:
- Raspberry Pi hosts the dashboard and camera ingestion.
- Logitech C920:
    * foreground/object mask
    * dominant object color
    * validated object count
- Intel RealSense D435/D435i:
    * RGB/depth stream
    * empty-scene depth baseline
    * depth-based volume integration
- Depth Anything V2 worker runs on the Windows laptop and posts its result back.

Important:
This is a RECREATED V14 from the workflow we built in chat, not a byte-for-byte
copy of an original source file that is no longer available here.
"""

import time
import json
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, render_template_string

try:
    import pyrealsense2 as rs
    HAVE_RS = True
except Exception:
    HAVE_RS = False

PORT = 5005
RESULT_DIR = Path("/home/locallife/LocalLife/v14_results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
mtx = threading.Lock()

state = {
    "logi_frame": None,
    "rs_frame": None,
    "latest_rs_depth_m": None,
    "empty_logi": None,
    "empty_rs_depth_m": None,
    "logi_mask": None,

    "color": "unknown",
    "count": 0,
    "rs_volume_l": None,
    "da_volume_l": None,

    # x, y, w, h as normalized values
    "logi_roi": [0.20, 0.18, 0.60, 0.62],
    "rs_roi": [0.20, 0.18, 0.60, 0.62],

    "da_last_seen": 0.0,
}


def roi_pixels(shape, roi):
    h, w = shape[:2]
    x, y, rw, rh = roi
    x1 = int(np.clip(x, 0, 1) * w)
    y1 = int(np.clip(y, 0, 1) * h)
    x2 = int(np.clip(x + rw, 0, 1) * w)
    y2 = int(np.clip(y + rh, 0, 1) * h)
    return x1, y1, x2, y2


def keep_largest_component(mask, min_area=350):
    src = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(src, 8)
    if n <= 1:
        return np.zeros_like(mask, dtype=np.uint8)

    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[best, cv2.CC_STAT_AREA])
    if area < min_area:
        return np.zeros_like(mask, dtype=np.uint8)

    return ((labels == best).astype(np.uint8) * 255)


def logitech_foreground(frame):
    with mtx:
        baseline = None if state["empty_logi"] is None else state["empty_logi"].copy()
        roi = list(state["logi_roi"])

    if frame is None:
        return None

    if baseline is None or baseline.shape != frame.shape:
        return np.zeros(frame.shape[:2], np.uint8)

    # Baseline differencing in LAB space is more robust than raw BGR.
    current_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    base_lab = cv2.cvtColor(baseline, cv2.COLOR_BGR2LAB)
    diff = cv2.absdiff(current_lab, base_lab)
    score = np.max(diff, axis=2)

    mask = (score > 16).astype(np.uint8) * 255

    x1, y1, x2, y2 = roi_pixels(frame.shape, roi)
    roi_mask = np.zeros_like(mask)
    roi_mask[y1:y2, x1:x2] = 255
    mask = cv2.bitwise_and(mask, roi_mask)

    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)

    return keep_largest_component(mask, 350)


def detect_color(frame, mask):
    if frame is None or mask is None or int((mask > 0).sum()) < 150:
        return "unknown"

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    px = hsv[mask > 0]
    if len(px) < 100:
        return "unknown"

    # Prefer useful saturated pixels when available.
    sat = px[:, 1]
    val = px[:, 2]
    useful = (sat > 45) & (val > 35)
    if useful.sum() > 60:
        px = px[useful]

    h = float(np.median(px[:, 0]))
    s = float(np.median(px[:, 1]))
    v = float(np.median(px[:, 2]))

    if v < 45 and s < 70:
        return "black"

    if s < 35:
        if v >= 190:
            return "white"
        if v >= 95:
            return "grey"
        return "black"

    if h < 8 or h >= 172:
        return "red"
    if h < 18:
        return "orange"
    if h < 32:
        return "yellow" if v >= 145 else "brown"
    if h < 85:
        return "green"
    if h < 105:
        return "cyan"
    if h < 135:
        return "blue"
    if h < 160:
        return "purple"
    return "red"


def draw_roi(frame, roi, label="ROI"):
    if frame is None:
        return None
    out = frame.copy()
    x1, y1, x2, y2 = roi_pixels(out.shape, roi)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(
        out, label, (x1 + 6, max(22, y1 + 22)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
    )
    return out


def compute_realsense_volume_l(depth_m, baseline_m, intrinsics, rgb_shape, roi):
    if depth_m is None or baseline_m is None:
        return None, None
    if depth_m.shape != baseline_m.shape:
        return None, None

    # Object is closer than the empty scene.
    delta = baseline_m - depth_m

    x1, y1, x2, y2 = roi_pixels(rgb_shape, roi)
    valid = (
        (delta > 0.015) &
        (delta < 0.60) &
        (depth_m > 0.12) &
        (baseline_m > 0.12)
    )

    roi_bool = np.zeros_like(valid, dtype=bool)
    roi_bool[y1:y2, x1:x2] = True
    valid &= roi_bool

    mask = keep_largest_component(valid.astype(np.uint8) * 255, 300)
    ys, xs = np.where(mask > 0)
    if len(xs) < 300:
        return None, mask

    z = depth_m[ys, xs]
    height = np.clip(delta[ys, xs], 0.0, 0.60)

    # Perspective-corrected physical area represented by each pixel.
    pixel_area_m2 = (z * z) / (intrinsics.fx * intrinsics.fy)

    volume_m3 = float(np.sum(height * pixel_area_m2))
    volume_l = volume_m3 * 1000.0
    return volume_l, mask


def camera_loop():
    # Logitech C920
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    pipeline = None
    align = None

    if HAVE_RS:
        try:
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(cfg)
            align = rs.align(rs.stream.color)
            print("RealSense online")
        except Exception as exc:
            print("RealSense init failed:", exc)
            pipeline = None
    else:
        print("pyrealsense2 unavailable — RealSense disabled")

    while True:
        # Logitech
        ok, logi = cap.read()
        if ok:
            mask = logitech_foreground(logi)
            color = detect_color(logi, mask)
            with mtx:
                state["logi_frame"] = logi
                state["logi_mask"] = mask
                if color != "unknown":
                    state["color"] = color

        # RealSense
        if pipeline is not None:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1500)
                frames = align.process(frames)

                color_f = frames.get_color_frame()
                depth_f = frames.get_depth_frame()
                if color_f and depth_f:
                    rs_rgb = np.asanyarray(color_f.get_data())
                    depth_raw = np.asanyarray(depth_f.get_data()).astype(np.float32)

                    sensor = frames.get_profile().get_device().first_depth_sensor()
                    scale = sensor.get_depth_scale()
                    depth_m = depth_raw * scale

                    intr = depth_f.profile.as_video_stream_profile().intrinsics

                    with mtx:
                        baseline = None if state["empty_rs_depth_m"] is None else state["empty_rs_depth_m"].copy()
                        roi = list(state["rs_roi"])

                    volume_l, rs_mask = compute_realsense_volume_l(
                        depth_m, baseline, intr, rs_rgb.shape, roi
                    )

                    if rs_mask is not None and int((rs_mask > 0).sum()) > 0:
                        contours, _ = cv2.findContours(
                            rs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(rs_rgb, contours, -1, (0, 0, 255), 2)

                    with mtx:
                        state["rs_frame"] = rs_rgb
                        state["latest_rs_depth_m"] = depth_m
                        state["rs_volume_l"] = volume_l

            except Exception:
                pass

        time.sleep(0.01)


def make_stream(which):
    while True:
        with mtx:
            if which == "logi":
                frame = None if state["logi_frame"] is None else state["logi_frame"].copy()
                roi = list(state["logi_roi"])
                mask = None if state["logi_mask"] is None else state["logi_mask"].copy()
            else:
                frame = None if state["rs_frame"] is None else state["rs_frame"].copy()
                roi = list(state["rs_roi"])
                mask = None

        if frame is None:
            frame = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(
                frame, "WAITING FOR CAMERA", (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

        if which == "logi" and mask is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

        frame = draw_roi(frame, roi, "ACTIVE ROI")

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpg.tobytes() +
                b"\r\n"
            )
        time.sleep(0.08)


HTML = r"""
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LocalLife V14</title>
<style>
body{font-family:Arial;margin:0;padding:18px;background:#0d1114;color:#eef2f4}
h1{margin:0 0 4px}.sub{color:#9dafba;margin-bottom:14px}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0}
.metric,.card{background:#182027;border:1px solid #2c3942;border-radius:12px;padding:12px}
.metric{text-align:center}.label{font-size:11px;color:#8fa2ae}.value{font-size:28px;font-weight:700}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
img{width:100%;background:black;border-radius:8px}
button{padding:11px 14px;border:0;border-radius:8px;margin:4px;font-weight:700;cursor:pointer}
.blue{background:#1565c0;color:white}.green{background:#2e7d32;color:white}.grey{background:#455a64;color:white}
input{width:65px;padding:7px;background:#111;color:white;border:1px solid #58656d;border-radius:4px}
.small{font-size:12px;color:#9dafba}
@media(max-width:900px){.metrics,.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<h1>LocalLife V14 — Recreated</h1>
<div class="sub">Logitech = color + validated count • RealSense = hardware-depth volume • DA V2 = laptop worker</div>

<div class="metrics">
 <div class="metric"><div class="label">LOGITECH COLOR</div><div class="value" id="color">-</div></div>
 <div class="metric"><div class="label">OBJECT COUNT</div><div class="value" id="count">0</div></div>
 <div class="metric"><div class="label">REALSENSE VOLUME</div><div class="value" id="rsvol">-</div></div>
 <div class="metric"><div class="label">DA V2 VOLUME</div><div class="value" id="davol">-</div></div>
</div>

<div>
 <button class="blue" onclick="setEmpty()">SET EMPTY SCENE</button>
 <button class="green" onclick="acceptObject()">ACCEPT / COUNT OBJECT</button>
 <button class="grey" onclick="resetCount()">RESET COUNT</button>
 <span id="worker" class="small"></span>
</div>

<div class="grid">
 <div class="card">
   <h3>Intel RealSense D435/D435i</h3>
   <img src="/video/rs">
 </div>

 <div class="card">
   <h3>Logitech C920 — Color + Count</h3>
   <img src="/video/logi">
 </div>

 <div class="card">
   <h3>ROI controls</h3>
   <div class="small">Values are percentages of the frame.</div><br>

   <b>RealSense ROI</b><br>
   X <input id="rx" value="20">
   Y <input id="ry" value="18">
   W <input id="rw" value="60">
   H <input id="rh" value="62"><br><br>

   <b>Logitech ROI</b><br>
   X <input id="lx" value="20">
   Y <input id="ly" value="18">
   W <input id="lw" value="60">
   H <input id="lh" value="62"><br><br>

   <button class="blue" onclick="saveROI()">SAVE ROI</button>
 </div>

 <div class="card">
   <h3>Recommended V14 test</h3>
   <ol>
     <li>Remove the object.</li>
     <li>Press <b>SET EMPTY SCENE</b>.</li>
     <li>Place one object fully inside both ROIs.</li>
     <li>Step away and let the feeds stabilise.</li>
     <li>Check the Logitech red contour and color.</li>
     <li>Check RealSense volume.</li>
     <li>Press <b>ACCEPT / COUNT OBJECT</b> once.</li>
   </ol>
   <div class="small">
   Count is intentionally validated/manual in this recreated V14 so it does not climb while nothing is happening.
   </div>
 </div>
</div>

<script>
function p(id){return parseFloat(document.getElementById(id).value)/100.0}

async function saveROI(){
 await fetch('/api/roi',{
   method:'POST',
   headers:{'Content-Type':'application/json'},
   body:JSON.stringify({
     rs:[p('rx'),p('ry'),p('rw'),p('rh')],
     logi:[p('lx'),p('ly'),p('lw'),p('lh')]
   })
 });
}

async function setEmpty(){
 const r=await fetch('/api/set_empty',{method:'POST'});
 alert((await r.json()).message);
}

async function acceptObject(){
 const r=await fetch('/api/accept',{method:'POST'});
 const j=await r.json();
 if(!r.ok) alert(j.error);
}

async function resetCount(){
 await fetch('/api/reset_count',{method:'POST'});
}

async function tick(){
 try{
   const s=await (await fetch('/api/state')).json();
   document.getElementById('color').innerText=s.color;
   document.getElementById('count').innerText=s.count;
   document.getElementById('rsvol').innerText=s.rs_volume_l==null?'-':s.rs_volume_l.toFixed(3)+' L';
   document.getElementById('davol').innerText=s.da_volume_l==null?'-':s.da_volume_l.toFixed(3)+' L';
   document.getElementById('worker').innerText=s.da_online?'DA WORKER ONLINE':'DA WORKER OFFLINE';
 }catch(e){}
}
setInterval(tick,700); tick();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video/<which>")
def video(which):
    if which not in ("rs", "logi"):
        which = "logi"
    return Response(
        make_stream(which),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/state")
def api_state():
    with mtx:
        return jsonify(
            color=state["color"],
            count=state["count"],
            rs_volume_l=state["rs_volume_l"],
            da_volume_l=state["da_volume_l"],
            da_online=(time.time() - state["da_last_seen"]) < 5.0,
        )


@app.route("/api/roi", methods=["POST"])
def api_roi():
    payload = request.get_json(force=True)
    with mtx:
        if "rs" in payload:
            state["rs_roi"] = [float(v) for v in payload["rs"]]
        if "logi" in payload:
            state["logi_roi"] = [float(v) for v in payload["logi"]]
    return jsonify(ok=True)


@app.route("/api/set_empty", methods=["POST"])
def api_set_empty():
    with mtx:
        if state["logi_frame"] is not None:
            state["empty_logi"] = state["logi_frame"].copy()
        if state["latest_rs_depth_m"] is not None:
            state["empty_rs_depth_m"] = state["latest_rs_depth_m"].copy()

        state["color"] = "unknown"
        state["rs_volume_l"] = None
        state["da_volume_l"] = None

    return jsonify(ok=True, message="Empty scene captured.")


@app.route("/api/reset_count", methods=["POST"])
def api_reset_count():
    with mtx:
        state["count"] = 0
    return jsonify(ok=True)


@app.route("/api/accept", methods=["POST"])
def api_accept():
    # Count source = Logitech foreground.
    with mtx:
        frame = None if state["logi_frame"] is None else state["logi_frame"].copy()
        mask = None if state["logi_mask"] is None else state["logi_mask"].copy()

    if frame is None or mask is None or int((mask > 0).sum()) < 350:
        return jsonify(ok=False, error="No valid Logitech object in the ROI."), 400

    color = detect_color(frame, mask)

    with mtx:
        state["count"] += 1
        state["color"] = color

        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "count": state["count"],
            "color": state["color"],
            "rs_volume_l": state["rs_volume_l"],
            "da_volume_l": state["da_volume_l"],
        }

    with open(RESULT_DIR / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return jsonify(ok=True, **record)


@app.route("/api/da_input")
def api_da_input():
    # Windows DA worker fetches latest Logitech RGB.
    with mtx:
        frame = None if state["logi_frame"] is None else state["logi_frame"].copy()

    if frame is None:
        return Response(status=404)

    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return Response(status=500)

    return Response(jpg.tobytes(), mimetype="image/jpeg")


@app.route("/api/da_result", methods=["POST"])
def api_da_result():
    payload = request.get_json(force=True)

    with mtx:
        state["da_last_seen"] = time.time()
        value = payload.get("volume_l")
        state["da_volume_l"] = None if value is None else float(value)

    return jsonify(ok=True)


def main():
    threading.Thread(target=camera_loop, daemon=True).start()
    print(f"LocalLife V14 dashboard: http://0.0.0.0:{PORT}/")
    app.run(host="0.0.0.0", port=PORT, threaded=True)


if __name__ == "__main__":
    main()
