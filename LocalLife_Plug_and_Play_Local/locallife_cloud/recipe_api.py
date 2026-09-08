"""FastAPI cloud endpoint for the dual-camera recipe pipeline (recipe Step 6).

This runs as a separate, optional service ALONGSIDE the existing Flask
dashboard in `server.py` -- it never replaces it. `server.py` is this
project's own tracking/ledger/waste-stream/calibration system, already
tested against real hardware over many rounds; this module is the recipe's
own narrow "Cloud (Main Brain)" contract instead -- one HTTP call in, the
recipe's exact JSON schema out -- for a caller (a script, a notebook, a
different Pi client) that wants only that contract rather than the full
dashboard experience. `pipeline.py`'s dashboard-facing measurement path is
untouched by this module.

Wire format deliberately mirrors this project's existing Pi -> cloud ingest
convention (`edge_client.py`'s `stream_frames()`, `server.py`'s
`/api/ingest`) rather than inventing a new one: each color frame is a JPEG
file upload, depth is an `application/octet-stream` `.npz` file with one
array named `depth_m` (meters, float32), and `metadata` is a JSON string
form field carrying the RealSense intrinsics (and anything else the caller
wants to pass, e.g. `confidence_threshold`). Reusing this convention means
`edge_client.py`'s existing capture/encode helpers could feed this endpoint
too, with no new client-side serialization code.

Start with:
    python -m locallife_cloud.recipe_api [--host HOST] [--port PORT]
or via the launcher's dedicated "Recipe API" role (see
`Start-LocalLife-Demo.ps1`). Auth reuses LOCALLIFE_API_TOKEN -- the same
token `/api/ingest` already requires -- so one credential covers both
services; binding outside localhost without a token is refused for the same
reason `server.py` refuses it (see `main()` below).

fastapi/starlette are imported at module level but inside a try/except, so
the rest of this package (and its test suite) keeps working unchanged in an
environment where fastapi is not installed -- this endpoint is additive
(tasks #11/#13 of the recipe-build work), not a new hard dependency of the
existing dashboard. They are imported at module level rather than deferred
inside `create_app()` because FastAPI resolves each endpoint parameter's
type annotation (UploadFile, Header, ...) against this module's own global
namespace -- with `from __future__ import annotations` in effect, an import
local to a factory function is invisible to that resolution and raises
`PydanticUserError: ... is not fully defined` at app-creation time, so
`UploadFile`/`File`/`Form`/`Header` in particular must be real module
globals by the time `process()` below is defined, not names bound inside
`create_app()`. uvicorn stays a deferred import inside `main()`, since
nothing else needs it at module-load time.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import numpy as np

from .config import AppConfig
from .recipe_detect import DEFAULT_RECIPE_DETECTOR_MODEL, RecipeDetector
from .recipe_material import RecipeMaterialClassifier
from .recipe_pipeline import process_object
from .types import CameraIntrinsics

LOGGER = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
    from fastapi.responses import JSONResponse

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when fastapi is missing
    _FASTAPI_AVAILABLE = False

# Deliberately distinct from AppConfig.port (the main dashboard's default,
# 8000) so both services can run on the same host without a port clash when
# a caller has not set LOCALLIFE_RECIPE_PORT explicitly.
DEFAULT_RECIPE_PORT = 8100

# Recipe's own exact output schema, restated here for readers of this file
# (process_object() in recipe_pipeline.py is the single source of truth for
# the actual values -- this comment is documentation, not a second schema).
RECIPE_RESULT_KEYS = (
    "volume_liters", "volume_confidence", "color", "color_confidence",
    "material", "material_confidence", "object_type", "timestamp",
)


def _decode_jpeg(raw: bytes) -> np.ndarray:
    import cv2

    encoded = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("uploaded image could not be decoded")
    return frame


def _load_depth_npz(raw: bytes) -> np.ndarray:
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if "depth_m" not in archive:
            raise ValueError("depth .npz upload must contain an array named 'depth_m'")
        return archive["depth_m"].astype(np.float32)


def create_app(config: AppConfig | None = None) -> Any:
    """Build and return the FastAPI app. Call once per process -- the
    detector and material classifier are loaded once here and reused across
    every request, not reloaded per call (loading either from cold is a
    multi-second operation)."""
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "fastapi is required for the recipe API: pip install fastapi uvicorn python-multipart"
        )

    settings = config or AppConfig.from_env()
    settings.validate()

    detector = RecipeDetector(model_name=DEFAULT_RECIPE_DETECTOR_MODEL)
    material_classifier = RecipeMaterialClassifier(settings)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        # Loaded eagerly at startup, not lazily on first request, so the
        # first real request from the Pi does not pay multi-second
        # model-load latency itself. A failed load degrades gracefully --
        # detector.enabled / material_classifier.enabled become False and
        # requests still succeed, just with weaker detections/"Other"
        # material -- rather than crashing the service, matching how the
        # rest of this project treats optional-model load failures (see
        # pipeline.py's own warmup()).
        detector.load()
        material_classifier.load()
        LOGGER.info(
            "Recipe API ready: detector_enabled=%s (%s) material_enabled=%s",
            detector.enabled, detector.model_name, material_classifier.enabled,
        )
        yield

    app = FastAPI(
        title="LocalLife Recipe API",
        description=(
            "Dual-camera (RealSense + Logitech) volume, color, and material "
            "estimation, following the recipe's exact JSON contract."
        ),
        version="1.0.0",
        lifespan=_lifespan,
    )
    app.state.detector = detector
    app.state.material_classifier = material_classifier

    def _check_token(authorization: str | None, x_api_token: str | None) -> None:
        if not settings.api_token:
            return
        bearer = f"Bearer {settings.api_token}"
        if authorization != bearer and x_api_token != settings.api_token:
            raise HTTPException(status_code=401, detail="Invalid or missing API token")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "detector_enabled": detector.enabled,
            "detector_model": detector.model_name,
            "material_enabled": material_classifier.enabled,
        }

    @app.post("/api/recipe/process")
    async def process(
        realsense_image: UploadFile = File(..., description="RealSense color frame, JPEG"),
        realsense_depth: UploadFile = File(
            ..., description="RealSense aligned depth, .npz with one array named 'depth_m' (meters, float32)"
        ),
        metadata: str = Form(
            ...,
            description=(
                'JSON string, e.g. {"intrinsics": {"fx":615.0,"fy":615.0,"ppx":320,"ppy":240,'
                '"width":640,"height":480}, "confidence_threshold": 0.25}. "intrinsics" is required.'
            ),
        ),
        logitech_image: UploadFile | None = File(
            None, description="Logitech color frame, JPEG (optional, but preferred for color/material)"
        ),
        realsense_baseline_depth: UploadFile | None = File(
            None,
            description=(
                "Empty-scene RealSense depth, .npz with 'depth_m' (optional; improves the "
                "heightmap volume method used for bags)"
            ),
        ),
        authorization: str | None = Header(default=None),
        x_api_token: str | None = Header(default=None, alias="X-API-Token"),
    ) -> Any:
        """Recipe Steps 2-6 in one call. Returns the recipe's exact JSON
        schema: volume_liters, volume_confidence, color, color_confidence,
        material, material_confidence, object_type, timestamp."""
        _check_token(authorization, x_api_token)

        try:
            payload = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"metadata is not valid JSON: {exc}") from exc

        intrinsics = CameraIntrinsics.from_dict(payload.get("intrinsics"))
        if intrinsics is None:
            raise HTTPException(status_code=400, detail="metadata.intrinsics is required (at least fx, fy)")

        try:
            realsense_color_bgr = _decode_jpeg(await realsense_image.read())
            realsense_depth_m = _load_depth_npz(await realsense_depth.read())
            logitech_color_bgr = (
                _decode_jpeg(await logitech_image.read()) if logitech_image is not None else None
            )
            baseline_depth_m = (
                _load_depth_npz(await realsense_baseline_depth.read())
                if realsense_baseline_depth is not None else None
            )
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if realsense_color_bgr.shape[:2] != realsense_depth_m.shape[:2]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"realsense_image size {realsense_color_bgr.shape[:2]} does not match "
                    f"realsense_depth size {realsense_depth_m.shape[:2]}"
                ),
            )

        confidence_threshold = float(payload.get("confidence_threshold", 0.25))

        result = process_object(
            realsense_color_bgr,
            realsense_depth_m,
            intrinsics,
            logitech_color_bgr,
            detector=detector,
            material_classifier=material_classifier,
            realsense_baseline_depth_m=baseline_depth_m,
            confidence_threshold=confidence_threshold,
        )
        return JSONResponse(result)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start the LocalLife recipe API (dual-camera volume/color/material, recipe Step 6)"
    )
    parser.add_argument("--host", help="Bind address; defaults to localhost for safe SSH tunneling")
    parser.add_argument("--port", type=int, help=f"HTTP port (default: {DEFAULT_RECIPE_PORT}, or LOCALLIFE_RECIPE_PORT)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = AppConfig.from_env()
    host = args.host or os.environ.get("LOCALLIFE_RECIPE_HOST", "127.0.0.1")
    port = args.port or int(os.environ.get("LOCALLIFE_RECIPE_PORT", DEFAULT_RECIPE_PORT))
    if host not in {"127.0.0.1", "localhost", "::1"} and not config.api_token:
        parser.error("Binding outside localhost requires LOCALLIFE_API_TOKEN")

    app = create_app(config)

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "uvicorn is required to run the recipe API: pip install fastapi uvicorn python-multipart"
        ) from exc

    LOGGER.info("Recipe API: http://%s:%s", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
