"""Capture physical cameras locally and stream aligned data to the cloud GPU."""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import requests

from .camera_recovery import run_resilient_camera


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CapturedFrame:
    image: np.ndarray
    depth_m: np.ndarray | None
    intrinsics: dict[str, Any] | None
    source: str
    camera_id: str = "realsense"
    intrinsics_origin: str = "factory-calibrated"


def _hardware_reset_realsense(rs: Any) -> bool:
    """Issue a firmware-level reset to every attached RealSense device.

    "xioctl(...) failed, errno=16: Device or resource busy" from
    `pipeline.start()` means the device is stuck in a busy/streaming state at
    the USB level -- almost always left over from an earlier process that
    stopped without a clean `pipeline.stop()` (a killed process, a closed
    launcher window, a dropped SSH connection). Killing that old process only
    closes its file descriptor; it does not by itself clear the RealSense
    device's own internal USB/firmware state, so a plain retry after the kill
    can still hit the same busy error. `device.hardware_reset()` is
    librealsense's own documented recovery for exactly this: a real reset of
    the device, after which it briefly disappears and re-enumerates on the
    USB bus. Returns True if at least one device was reset.
    """
    reset_any = False
    try:
        devices = rs.context().query_devices()
    except RuntimeError as exc:
        LOGGER.warning("Could not enumerate RealSense devices for a hardware reset: %s", exc)
        return False
    for device in devices:
        try:
            device.hardware_reset()
            reset_any = True
            LOGGER.info(
                "Issued a hardware reset to RealSense device %s",
                device.get_info(rs.camera_info.serial_number),
            )
        except RuntimeError as exc:
            LOGGER.warning("RealSense hardware reset failed: %s", exc)
    return reset_any


# Every real D400-series RealSense unit reports a depth scale in this band
# (Intel's own SDK default is 0.001 m/unit -- 1mm per raw depth count -- and
# real units vary only slightly around that for different firmware/depth
# tables). A `depth_scale` outside this band means either a driver returned
# something other than metres-per-unit (a stale/mocked SDK, a unit mix-up
# upstream), or a genuine factor-of-10/100/1000 bug -- silently multiplying
# raw depth counts by a wrong-by-orders-of-magnitude scale would then flow,
# undetected, into every downstream metre-based calculation (plane fit,
# height, volume) as a wrong-but-plausible-looking number, exactly the class
# of bug the build spec's "a factor-of-1000 error must fail loudly" requires
# guarding against with an explicit runtime check rather than trusting the
# SDK call silently.
_MIN_PLAUSIBLE_DEPTH_SCALE_M = 1e-5
_MAX_PLAUSIBLE_DEPTH_SCALE_M = 1e-2


def _validate_depth_scale(depth_scale: float) -> float:
    """Fail loudly if the RealSense-reported depth scale is implausible.

    Returns `depth_scale` unchanged when it passes, so this can be used
    inline. Kept as its own function (rather than inlined into
    `iter_realsense`) so it can be unit-tested directly with a known-bad
    scale, without needing a real or even a stubbed RealSense pipeline.
    """
    if not math.isfinite(depth_scale) or depth_scale <= 0:
        raise RuntimeError(
            f"RealSense reported a non-positive/non-finite depth scale ({depth_scale!r}); "
            "refusing to convert raw depth counts to metres with it."
        )
    if not (_MIN_PLAUSIBLE_DEPTH_SCALE_M <= depth_scale <= _MAX_PLAUSIBLE_DEPTH_SCALE_M):
        raise RuntimeError(
            f"RealSense reported an implausible depth scale ({depth_scale:.8f} m/unit); "
            f"expected roughly {_MIN_PLAUSIBLE_DEPTH_SCALE_M:g}-{_MAX_PLAUSIBLE_DEPTH_SCALE_M:g} "
            "m/unit for a real D400-series sensor. This looks like a unit/scale bug (e.g. a "
            "factor-of-1000 error), not a real hardware reading -- refusing to silently produce "
            "wrong-by-orders-of-magnitude depth/height/volume numbers downstream."
        )
    return depth_scale


def iter_realsense(
    width: int, height: int, fps: int, *, filter_depth: bool = True
) -> Iterator[CapturedFrame]:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is required on the physical camera device") from exc

    pipeline = rs.pipeline()
    configuration = rs.config()
    configuration.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    configuration.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    try:
        profile = pipeline.start(configuration)
    except RuntimeError as exc:
        LOGGER.warning(
            "RealSense pipeline.start() failed (%s); attempting a hardware reset and one retry",
            exc,
        )
        if _hardware_reset_realsense(rs):
            # The device drops off and re-enumerates on the USB bus after a
            # hardware reset; this typically takes a couple of seconds.
            time.sleep(3.0)
        profile = pipeline.start(configuration)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = _validate_depth_scale(depth_sensor.get_depth_scale())
    try:
        if depth_sensor.supports(rs.option.visual_preset):
            depth_sensor.set_option(rs.option.visual_preset, float(rs.rs400_visual_preset.high_accuracy))
            LOGGER.info("Enabled RealSense high-accuracy depth preset")
    except (AttributeError, RuntimeError, ValueError) as exc:
        LOGGER.info("RealSense high-accuracy preset unavailable; retaining the camera default: %s", exc)
    align = rs.align(rs.stream.color)
    # Post-processing filter chain, per the SDK's own documented order
    # ("Depth Frame >> Decimation >> Depth2Disparity >> Spatial >> Temporal
    # >> Disparity2Depth >> Hole Filling >> Filtered Depth", confirmed
    # against dev.realsenseai.com's own post-processing-filters page). Two
    # deliberate deviations from that literal order, both because this
    # project's own pixel-correspondence architecture depends on it:
    # (1) Decimation is intentionally NOT applied -- it downsamples the
    # depth frame's resolution, and every downstream consumer (mask
    # indexing, ROI pixels, pinhole backprojection) assumes depth and the
    # color-aligned frame share the same pixel grid and the color frame's
    # own intrinsics. Adding decimation without also re-deriving intrinsics
    # and re-sizing every mask consumer everywhere would silently
    # misalign depth and color rather than improve anything -- a much
    # larger, separately-scoped change, not done here.
    # (2) Alignment runs BEFORE this filter chain (not after, as the SDK
    # order table would imply), matching Intel's own official
    # align-depth2color.py reference example, since the SDK's own
    # post-processing documentation (checked directly) does not actually
    # specify where alignment belongs relative to filtering -- there is no
    # documented basis to move it, and moving it without one risks a
    # regression rather than a confirmed improvement.
    # Spatial and temporal filtering IS moved into the disparity domain
    # (the SDK's own recommendation for those two filters specifically),
    # and hole-filling is added -- both were previously entirely missing.
    to_disparity = rs.disparity_transform(True) if filter_depth else None
    to_depth = rs.disparity_transform(False) if filter_depth else None
    spatial_filter = rs.spatial_filter() if filter_depth else None
    temporal_filter = rs.temporal_filter() if filter_depth else None
    hole_filling_filter = rs.hole_filling_filter() if filter_depth else None
    LOGGER.info("RealSense camera online; depth scale is %.8f m", depth_scale)

    try:
        while True:
            frames = align.process(pipeline.wait_for_frames(timeout_ms=5000))
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue
            if (
                to_disparity is not None
                and spatial_filter is not None
                and temporal_filter is not None
                and to_depth is not None
                and hole_filling_filter is not None
            ):
                depth = to_disparity.process(depth)
                depth = spatial_filter.process(depth)
                depth = temporal_filter.process(depth)
                depth = to_depth.process(depth)
                depth = hole_filling_filter.process(depth)
            image = np.asanyarray(color.get_data())
            aligned_depth = np.asanyarray(depth.get_data()).astype(np.float32) * depth_scale
            camera = color.profile.as_video_stream_profile().intrinsics
            yield CapturedFrame(
                image=image,
                depth_m=aligned_depth,
                intrinsics={
                    "fx": camera.fx,
                    "fy": camera.fy,
                    "ppx": camera.ppx,
                    "ppy": camera.ppy,
                    "width": camera.width,
                    "height": camera.height,
                },
                source="realsense-aligned-rgb-depth",
                camera_id="realsense",
            )
    finally:
        pipeline.stop()


# A C920's sensor is 16:9. Its 70.42 deg horizontal field of view only holds
# in a 16:9 mode; a 4:3 mode (640x480, 800x600, 1600x1200) crops the sensor
# horizontally and keeps the vertical field of view, so deriving fx from the
# horizontal figure there underestimates it by about a third -- and a focal
# length that is too short makes every back-projected footprint too large.
NATIVE_ASPECT = 16.0 / 9.0
DEFAULT_VERTICAL_FOV_DEG = 43.3


def camera_intrinsics_from_fov(
    width: int, height: int, *, horizontal_fov_deg: float = 70.42,
    vertical_fov_deg: float = DEFAULT_VERTICAL_FOV_DEG,
    fx: float | None = None, fy: float | None = None,
) -> dict[str, Any]:
    if width < 1 or height < 1 or not 1 < horizontal_fov_deg < 179:
        raise ValueError("Camera dimensions and horizontal field of view must be valid")
    if not 1 < vertical_fov_deg < 179:
        raise ValueError("Camera vertical field of view must be valid")
    if abs(width / height - NATIVE_ASPECT) <= 0.02:
        estimated_fx = width / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
    else:
        # Square pixels: fx == fy, and the vertical field of view survives a
        # horizontal crop, so it is the reliable one outside 16:9.
        estimated_fx = height / (2.0 * math.tan(math.radians(vertical_fov_deg) / 2.0))
    actual_fx = estimated_fx if fx is None else float(fx)
    actual_fy = actual_fx if fy is None else float(fy)
    if not math.isfinite(actual_fx) or not math.isfinite(actual_fy) or min(actual_fx, actual_fy) <= 0:
        raise ValueError("Camera focal lengths must be finite positive values")
    return {"fx": actual_fx, "fy": actual_fy, "ppx": (width - 1) / 2,
            "ppy": (height - 1) / 2, "width": width, "height": height}


def discover_logitech_source() -> str | int:
    """Prefer the named Logitech V4L node over RealSense's own USB video nodes."""
    by_id = Path("/dev/v4l/by-id")
    if by_id.is_dir():
        for entry in sorted(by_id.glob("*video-index0")):
            if "logitech" in entry.name.lower() or "c920" in entry.name.lower():
                return str(entry)
    sysfs = Path("/sys/class/video4linux")
    if sysfs.is_dir():
        for entry in sorted(sysfs.glob("video*")):
            try:
                label = (entry / "name").read_text(encoding="utf-8").strip().lower()
            except OSError:
                continue
            if "logitech" in label or "c920" in label:
                return str(Path("/dev") / entry.name)
    LOGGER.warning("No named Logitech camera was found; trying USB camera index 0")
    return 0


def iter_video(
    source: str | int, width: int, height: int, *, horizontal_fov_deg: float = 70.42,
    fx: float | None = None, fy: float | None = None,
) -> Iterator[CapturedFrame]:
    import cv2

    capture = cv2.VideoCapture(source)
    announced = False
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open camera/video source: {source}")
    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                if isinstance(source, str):
                    return
                raise RuntimeError("USB camera stopped returning frames")
            real_height, real_width = image.shape[:2]
            intrinsics = camera_intrinsics_from_fov(
                real_width, real_height, horizontal_fov_deg=horizontal_fov_deg, fx=fx, fy=fy
            )
            if not announced:
                LOGGER.info(
                    "Logitech capture %dx%d, intrinsics fx=%.1f fy=%.1f ppx=%.1f ppy=%.1f (%s)",
                    real_width, real_height, intrinsics["fx"], intrinsics["fy"],
                    intrinsics["ppx"], intrinsics["ppy"],
                    "measured lens calibration" if fx is not None else "field-of-view estimate",
                )
                announced = True
            yield CapturedFrame(
                image=image, depth_m=None,
                intrinsics=intrinsics,
                source=f"logitech-video:{source}", camera_id="logitech",
                intrinsics_origin="measured-lens-calibration" if fx is not None else "estimated-horizontal-fov",
            )
    finally:
        capture.release()


def send_frame(
    session: requests.Session,
    endpoint: str,
    frame: CapturedFrame,
    *,
    jpeg_quality: int = 88,
    timeout_seconds: float = 90.0,
) -> dict[str, Any]:
    import cv2

    success, encoded = cv2.imencode(".jpg", frame.image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not success:
        raise RuntimeError("Could not JPEG-encode the camera frame")

    metadata = {"source": frame.source, "timestamp": time.time(), "intrinsics": frame.intrinsics,
                "camera_id": frame.camera_id, "intrinsics_origin": frame.intrinsics_origin}
    files: dict[str, Any] = {"image": ("frame.jpg", encoded.tobytes(), "image/jpeg")}
    if frame.depth_m is not None:
        compressed = io.BytesIO()
        np.savez_compressed(compressed, depth_m=frame.depth_m.astype(np.float32))
        files["depth"] = ("depth.npz", compressed.getvalue(), "application/octet-stream")

    response = session.post(
        endpoint.rstrip("/") + "/api/ingest",
        files=files,
        data={"metadata": json.dumps(metadata)},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream physical cameras to the Local Life processing service")
    parser.add_argument("--cloud", default="http://127.0.0.1:8000", help="Processing API through the private SSH bridge")
    parser.add_argument("--source", default="realsense",
                        help="dual, realsense, logitech, a USB camera index, or a video file")
    parser.add_argument("--logitech-source", default="auto", help="Logitech USB index/device for --source dual")
    parser.add_argument("--logitech-fov-deg", type=float, default=70.42,
                        help="Horizontal field of view; replace with a measured lens calibration when possible")
    parser.add_argument("--logitech-fx", type=float, default=None, help="Measured Logitech focal length fx in pixels")
    parser.add_argument("--logitech-fy", type=float, default=None, help="Measured Logitech focal length fy in pixels")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--upload-fps", type=float, default=4.0)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--token", default="", help="Bearer token when the cloud API is not tunneled")
    parser.add_argument("--baseline-on-start", action="store_true", help="Capture the first empty frame as baseline")
    parser.add_argument("--raw-depth", action="store_true", help="Disable RealSense spatial/temporal noise filters")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if args.upload_fps <= 0:
        parser.error("--upload-fps must be positive")

    def stream_frames(frames: Iterator[CapturedFrame]) -> None:
        session = requests.Session()
        if args.token:
            session.headers["Authorization"] = f"Bearer {args.token}"
        baseline_pending = args.baseline_on_start
        for frame in frames:
            _stream_one_frame(session, frame, args, baseline_pending)
            if baseline_pending:
                try:
                    response = session.post(
                        args.cloud.rstrip("/") + f"/api/cameras/{frame.camera_id}/baseline", timeout=90
                    )
                    response.raise_for_status()
                    baseline_pending = False
                    LOGGER.info("%s empty-scene baseline captured", frame.camera_id)
                except requests.RequestException as exc:
                    LOGGER.warning("%s baseline capture will be retried: %s", frame.camera_id, exc)

    if args.source.lower() == "dual":
        def realsense_factory() -> Iterator[CapturedFrame]:
            return iter_realsense(
                args.width,
                args.height,
                args.camera_fps,
                filter_depth=not args.raw_depth,
            )

        def logitech_factory() -> Iterator[CapturedFrame]:
            # Re-run discovery on every retry because Linux can assign a new
            # /dev/video number after a camera is unplugged and reconnected.
            choice = discover_logitech_source() if args.logitech_source == "auto" else args.logitech_source
            usb_source: str | int = int(choice) if isinstance(choice, str) and choice.isdigit() else choice
            return iter_video(
                usb_source,
                args.width,
                args.height,
                horizontal_fov_deg=args.logitech_fov_deg,
                fx=args.logitech_fx,
                fy=args.logitech_fy,
            )

        camera_factories = {
            "realsense": realsense_factory,
            "logitech": logitech_factory,
        }
        threads = [threading.Thread(
            target=run_resilient_camera,
            args=(camera_id, factory, stream_frames),
            name=f"camera-{camera_id}",
            daemon=False,
        ) for camera_id, factory in camera_factories.items()]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    elif args.source.lower() == "realsense":
        stream_frames(iter_realsense(args.width, args.height, args.camera_fps, filter_depth=not args.raw_depth))
    else:
        choice = discover_logitech_source() if args.source.lower() in {"logitech", "auto"} else args.source
        source: str | int = int(choice) if isinstance(choice, str) and choice.isdigit() else choice
        stream_frames(iter_video(source, args.width, args.height, horizontal_fov_deg=args.logitech_fov_deg,
                                 fx=args.logitech_fx, fy=args.logitech_fy))


def _stream_one_frame(session: requests.Session, frame: CapturedFrame, args: Any, baseline_pending: bool) -> None:
        started = time.monotonic()
        try:
            result = send_frame(session, args.cloud, frame, jpeg_quality=args.jpeg_quality)
            valid_depth_pixels = (
                int(np.count_nonzero(np.isfinite(frame.depth_m) & (frame.depth_m > 0.10)))
                if frame.depth_m is not None
                else 0
            )
            LOGGER.info(
                "source=%s depth_pixels=%s intrinsics=%s objects=%s total=%s hardware_l=%s monocular_l=%s inference_ms=%s",
                frame.source,
                valid_depth_pixels,
                frame.intrinsics is not None,
                result["visible_objects"],
                result["automatic_count"],
                result["realsense_volume_l"],
                result["monocular_volume_l"],
                result["inference_ms"],
            )
        except requests.RequestException as exc:
            LOGGER.warning("Laptop processing service unavailable; reconnecting: %s", exc)
            time.sleep(2)
            return

        delay = (1.0 / args.upload_fps) - (time.monotonic() - started)
        if delay > 0:
            time.sleep(delay)


if __name__ == "__main__":
    main()
