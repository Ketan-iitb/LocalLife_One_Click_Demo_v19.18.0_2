"""GPU-backed open-vocabulary segmentation and metric monocular depth."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .config import AppConfig
from .geometry import dominant_color, fixed_bin_mask, roi_pixels
from .types import Detection


LOGGER = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is missing. Use the VM's preinstalled PyTorch environment "
            "and run scripts/setup_cloud.sh."
        ) from exc

    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access an NVIDIA GPU")
    return requested


def configure_torch(device: str) -> dict[str, Any]:
    import torch

    report: dict[str, Any] = {
        "torch_version": torch.__version__,
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        properties = torch.cuda.get_device_properties(device)
        report.update(
            gpu_name=properties.name,
            gpu_memory_gb=round(properties.total_memory / (1024**3), 2),
            cuda_version=torch.version.cuda,
        )
    return report


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    try:
        import cv2

        return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    except ImportError:
        from PIL import Image

        image = Image.fromarray(mask.astype(np.uint8) * 255)
        return np.asarray(image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)) > 0


class AdaptiveForegroundSegmenter:
    """CPU-only fixed-camera bag segmenter for the offline laptop build.

    The first stable empty frames become an independent reference for each
    camera. Later foreground components are segmented after compensating for
    global brightness changes. This labels foreground as a bag only inside the
    controlled waste-bag experiment; it is not a general object recognizer.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = "cpu"
        self.active_model_name = "local-opencv-background"
        self.runtime: dict[str, Any] = {
            "device": "Local laptop CPU",
            "active_detector_model": self.active_model_name,
            "prompt_mode": "fixed-camera-foreground",
            "cloud_required": False,
        }
        self._references: dict[str, np.ndarray] = {}
        self._samples: dict[str, list[np.ndarray]] = {}

    def load(self) -> None:
        LOGGER.info("Local OpenCV detector ready; no cloud, CUDA, or model download is required")

    def detect_camera_batch(self, frames: dict[str, np.ndarray]) -> list[list[Detection]]:
        return [self._detect(camera_id, frame) for camera_id, frame in frames.items()]

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        return [self._detect(f"slot-{index}", frame) for index, frame in enumerate(frames)]

    def _detect(self, camera_id: str, frame: np.ndarray) -> list[Detection]:
        import cv2

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab = cv2.GaussianBlur(lab, (5, 5), 0)
        region = fixed_bin_mask(frame.shape, self.config.roi, self.config.bin_polygon)
        reference = self._references.get(camera_id)
        required = max(5, min(12, self.config.automatic_baseline_frames))

        if reference is None:
            samples = self._samples.setdefault(camera_id, [])
            if samples:
                previous = samples[-1]
                motion = cv2.absdiff(lab[:, :, 0], previous[:, :, 0])
                changed = float(np.mean(motion[region] > max(7, self.config.foreground_threshold // 2)))
                if changed > 0.025:
                    samples.clear()
            samples.append(lab)
            if len(samples) >= required:
                self._references[camera_id] = np.median(np.stack(samples[-required:]), axis=0).astype(np.uint8)
                samples.clear()
                LOGGER.info("%s local empty-scene detector reference is ready", camera_id)
            return []

        # Remove a uniform exposure shift before foreground thresholding. The
        # A/B channels retain coloured bags while compensated luminance remains
        # sensitive to grey, white, and black bags.
        current = lab.astype(np.int16)
        base = reference.astype(np.int16)
        light_delta = int(np.median(current[:, :, 0][region] - base[:, :, 0][region]))
        light_delta = int(np.clip(light_delta, -35, 35))
        light = np.abs((current[:, :, 0] - light_delta) - base[:, :, 0])
        chroma = np.maximum(
            np.abs(current[:, :, 1] - base[:, :, 1]),
            np.abs(current[:, :, 2] - base[:, :, 2]),
        )
        score = np.maximum(light, chroma * 2)
        foreground = ((score >= self.config.foreground_threshold) & region).astype(np.uint8) * 255

        foreground = cv2.medianBlur(foreground, 5)
        foreground = cv2.morphologyEx(
            foreground,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        foreground = cv2.morphologyEx(
            foreground,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=2,
        )

        region_area = max(1, int(np.count_nonzero(region)))
        minimum = max(
            self.config.min_component_pixels,
            int(region_area * self.config.min_detection_area_fraction),
        )
        maximum = int(region_area * self.config.max_detection_area_fraction)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        candidates: list[tuple[int, Detection]] = []
        for component in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[component])
            if area < minimum or area > maximum:
                continue
            if width < frame.shape[1] * self.config.min_detection_side_fraction:
                continue
            if height < frame.shape[0] * self.config.min_detection_side_fraction:
                continue
            mask = labels == component
            contrast = float(np.median(score[mask])) if np.any(mask) else 0.0
            confidence = float(np.clip(0.62 + contrast / 255.0, 0.62, 0.96))
            candidates.append((area, Detection(
                label="garbage bag",
                confidence=confidence,
                box=(x, y, x + width, y + height),
                mask=mask,
                source="local-background-segmentation",
                color=dominant_color(frame, mask),
            )))
        return [item for _, item in sorted(candidates, key=lambda value: value[0], reverse=True)[:3]]


def create_segmenter(config: AppConfig) -> Any:
    if config.detector_model.strip().lower() in {
        "local-opencv", "local-opencv-background", "opencv-background"
    }:
        return AdaptiveForegroundSegmenter(config)
    return YoloSegmenter(config)


class YoloSegmenter:
    """Prompt waste categories directly instead of relying on COCO-only classes."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.runtime = configure_torch(self.device)
        self.model: Any | None = None
        self.active_model_name = config.detector_model

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is missing. Install requirements-cloud.txt first") from exc

        model_name = self.config.detector_model
        model_basename = Path(model_name).name.lower()
        is_promptable = "yoloe" in model_basename and "-pf" not in model_basename
        if is_promptable:
            try:
                from ultralytics import YOLOE

                self.model = YOLOE(model_name)
            except ImportError:
                self.model = YOLO(model_name)
            if not hasattr(self.model, "set_classes"):
                raise RuntimeError(
                    "The configured YOLOE model does not support text prompts. "
                    "Upgrade ultralytics or set LOCALLIFE_DETECTOR_MODEL to a trained best.pt."
                )
            try:
                prompt_bank = tuple(dict.fromkeys(
                    (*self.config.prompts, *self.config.negative_prompts)
                ))
                self.model.set_classes(list(prompt_bank))
                self.runtime["prompt_mode"] = "waste-text-prompts"
                self.runtime["accepted_prompt_count"] = len(self.config.prompts)
                self.runtime["negative_prompt_count"] = len(self.config.negative_prompts)
                LOGGER.info(
                    "Configured %s accepted and %s negative/lookalike prompts",
                    len(self.config.prompts), len(self.config.negative_prompts),
                )
            except Exception as exc:
                fallback_name = model_name.removesuffix(".pt") + "-pf.pt"
                LOGGER.warning(
                    "YOLOE text prompt initialization failed (%s). "
                    "Loading the prompt-free open-vocabulary fallback %s.",
                    exc,
                    fallback_name,
                )
                self.model = YOLO(fallback_name)
                self.active_model_name = fallback_name
                self.runtime["prompt_mode"] = "prompt-free-fallback"
                self.runtime["prompt_initialization_error"] = str(exc)
        else:
            self.model = YOLO(model_name)
            self.runtime["prompt_mode"] = "trained-or-standard-model"

        self.runtime["active_detector_model"] = self.active_model_name
        LOGGER.info("Detection model %s loaded on %s", self.active_model_name, self.device)

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []
        if self.model is None:
            self.load()

        predict_kwargs: dict[str, Any] = {
            "source": frames,
            "conf": self.config.detector_confidence,
            "iou": self.config.detector_iou,
            "imgsz": self.config.image_size,
            "device": self.device,
            "batch": min(len(frames), self.config.batch_size),
            "retina_masks": True,
            "verbose": False,
        }
        # Newer Ultralytics releases renamed `half` to `quantize` and log a
        # deprecation warning (once per call) whenever `half` is passed at
        # all, even as False -- which was happening on every single frame
        # batch and drowning real warnings in log noise. Only pass it when
        # actually requesting half precision.
        if self.config.half_precision and self.device.startswith("cuda"):
            predict_kwargs["half"] = True
        results = self.model.predict(**predict_kwargs)
        return [self._parse_result(frame, result) for frame, result in zip(frames, results, strict=True)]

    def _parse_result(self, frame: np.ndarray, result: Any) -> list[Detection]:
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        mask_data = None if result.masks is None else result.masks.data.detach().cpu().numpy()
        rx1, ry1, rx2, ry2 = roi_pixels(frame.shape, self.config.roi)
        detections: list[Detection] = []

        for index, (raw_box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids, strict=True)):
            x1, y1, x2, y2 = (int(round(value)) for value in raw_box)
            x1, x2 = max(0, x1), min(frame.shape[1], x2)
            y1, y2 = max(0, y1), min(frame.shape[0], y2)
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            if not (rx1 <= center_x < rx2 and ry1 <= center_y < ry2):
                continue

            if mask_data is not None and index < len(mask_data):
                mask = _resize_mask(mask_data[index] > 0.5, frame.shape[:2])
            else:
                mask = np.zeros(frame.shape[:2], dtype=bool)
                mask[y1:y2, x1:x2] = True

            if np.count_nonzero(mask) < self.config.min_component_pixels:
                continue
            names = result.names
            label = names.get(int(class_id), str(class_id)) if isinstance(names, dict) else names[int(class_id)]
            detections.append(
                Detection(
                    label=str(label),
                    confidence=float(confidence),
                    box=(x1, y1, x2, y2),
                    mask=mask,
                    source="yoloe-segmentation" if "yoloe" in self.config.detector_model.lower() else "yolo-segmentation",
                    color=dominant_color(frame, mask),
                )
            )
        return detections


class MetricDepthEstimator:
    """Run the larger indoor metric-depth model directly on the cloud GPU."""

    def __init__(self, config: AppConfig, device: str | None = None) -> None:
        self.config = config
        self.device = device or resolve_device(config.device)
        self.processor: Any | None = None
        self.model: Any | None = None

    def load(self) -> None:
        import torch

        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise RuntimeError("Transformers is missing. Install requirements-cloud.txt first") from exc

        self.processor = AutoImageProcessor.from_pretrained(self.config.depth_model)
        dtype = torch.float16 if self.config.half_precision and self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForDepthEstimation.from_pretrained(self.config.depth_model, torch_dtype=dtype)
        self.model.to(self.device).eval()
        LOGGER.info("Depth model %s loaded on %s using %s", self.config.depth_model, self.device, dtype)

    def estimate_batch(self, frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
        if not frames_bgr:
            return []
        if self.model is None or self.processor is None:
            self.load()

        import torch
        from PIL import Image

        images = [Image.fromarray(frame[:, :, ::-1]) for frame in frames_bgr]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {
            key: value.to(self.device, non_blocking=self.device.startswith("cuda"))
            for key, value in inputs.items()
        }

        use_half = self.config.half_precision and self.device.startswith("cuda")
        autocast_device = "cuda" if self.device.startswith("cuda") else "cpu"
        autocast_dtype = torch.float16 if autocast_device == "cuda" else torch.bfloat16
        with torch.inference_mode(), torch.autocast(
            device_type=autocast_device,
            dtype=autocast_dtype,
            enabled=use_half,
        ):
            predictions = self.model(**inputs).predicted_depth

        output: list[np.ndarray] = []
        for index, frame in enumerate(frames_bgr):
            resized = torch.nn.functional.interpolate(
                predictions[index][None, None].float(),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            output.append(resized[0, 0].detach().cpu().numpy().astype(np.float32))
        return output
