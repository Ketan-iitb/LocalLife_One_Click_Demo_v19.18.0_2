"""Configuration shared by the cloud service, batch jobs, and edge bridge."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


DEFAULT_PROMPTS = (
    # Keep this bank compact and material-specific.  The live image set from
    # 2026-09-10 showed that many near-synonymous shape/colour prompts
    # independently proposed the same object, while prompts such as "black
    # garbage bag" biased a visibly green bag toward black.  Colour is
    # measured from pixels, never encoded in the detector prompt.
    "plastic garbage bag",
    "plastic trash bag",
    "filled plastic waste bag",
    "polythene waste bag",
    "transparent plastic waste bag",
    "paper waste bag",
    "kraft paper bag",
    "paper shopping bag",
    "cardboard box",
    "cardboard shipping box",
    "carton box",
    "milk carton",
    "drink carton",
)

# Competing classes are shown to the open-vocabulary detector so it can call
# a backpack a backpack instead of being forced to choose the nearest waste
# label.  They remain separate from ``prompts`` so installations that already
# override LOCALLIFE_PROMPTS do not accidentally treat negatives as accepted
# application classes.
DEFAULT_NEGATIVE_PROMPTS = (
    # Explicit negative/lookalike prompts.  YOLOE can only reject a shoe or
    # backpack as a competing class if it was allowed to name it; with only
    # positive waste prompts it was forced to choose the nearest bag label.
    "backpack",
    "rucksack",
    "laptop bag",
    "briefcase",
    "duffel bag",
    "sports bag",
    "handbag",
    "purse",
    "shoe",
    "sneaker",
    "sandal",
    "slipper",
    "boot",
    "pillow",
    "cushion",
    "blanket",
    "bedding",
    "clothing",
    "bottle",
    "lotion bottle",
    "soda can",
    "aluminium drink can",
    "tin can",
    "chair",
    "furniture",
    "person",
    "hand",
    "foot",
    # Lookalikes observed in the 2026-09-10 hardware screenshots.
    "laundry basket",
    "laundry hamper",
    "fabric storage basket",
    "curtain",
    "drape",
    "chair cover",
    "floor mat",
    "rug",
    "power cable",
    "power adapter",
    "power strip",
    "charger",
    "door",
)

# Temporary, opt-in prompt bank for measuring known household objects while
# the RealSense geometry is being validated.  This is intentionally separate
# from the production waste prompt bank: switching back to ``waste`` restores
# the strict plastic-bag/paper-bag/cardboard-box contract without editing code.
DEFAULT_GEOMETRY_VALIDATION_PROMPTS = tuple(dict.fromkeys((
    *DEFAULT_PROMPTS,
    "backpack",
    "rucksack",
    "laptop bag",
    "briefcase",
    "duffel bag",
    "handbag",
    "shoe",
    "sneaker",
    "slipper",
    "pillow",
    "cushion",
    "folded clothing",
    "plastic bottle",
    "drink can",
    "laundry basket",
    "laundry hamper",
    "fabric storage basket",
    "storage container",
    "parcel",
    "package",
    "book",
    "toy",
    # Electronics and electrical items: absent from the bank, so a charger or a
    # pair of headphones could only be named as something it is not.
    "headphones", "earphones", "headset", "charger", "power adapter",
    "charging cable", "electrical cable", "mobile phone", "laptop",
    "computer mouse", "keyboard", "remote control", "battery", "circuit board",
    "electronic device", "light bulb", "LED bulb", "lamp", "table lamp",
    "torch", "flashlight", "electric drill", "power drill", "power tool",
    # Household and personal objects.
    "handbag", "shoe", "clothing", "textile", "painting", "picture frame",
    "framed artwork", "decorative object", "cosmetic bottle", "cream bottle",
    "glass bottle", "aluminium can", "metal can", "food container", "jar",
    "cup", "packet", "wrapper", "milk carton", "beverage carton",
    # Parent categories: a truthful fallback when the specific name is weak.
    "packaging object", "electronic item", "electrical item", "textile item",
    "rigid household object", "flexible household object", "unknown deposited object",
)))

# Scene classes the detector may name so it can reject them, never measure them.
BACKGROUND_PROMPTS = (
    "floor", "mat", "carpet", "sofa", "chair", "table", "furniture", "wall",
    "foot", "hand", "person", "shadow", "empty background", "permanent fixture",
)

# Scene/background classes must not become measurement objects in validation
# mode.  Hands and people are also excluded so placing/removing a reference
# object does not create a physical-measurement track for the operator.
GEOMETRY_VALIDATION_REJECT_LABELS = frozenset({
    "person", "hand", "foot", "floor", "wall", "ceiling", "door", "window",
    "curtain", "drape", "rug", "floor mat", "carpet", "table", "desk",
    "chair", "sofa", "couch", "bed", "furniture", "power strip",
    "empty background", "permanent fixture", "shadow", "mat",
    "unknown", "unclassified object", "foreground object",
})
GEOMETRY_VALIDATION_REJECT_WORDS = frozenset({
    "person", "people", "human", "hand", "hands", "foot", "feet",
    "floor", "wall", "ceiling", "door", "window", "curtain", "drape",
    "rug", "carpet", "table", "desk", "chair", "sofa", "couch", "bed",
    "furniture", "shadow", "fixture",
})


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _tuple_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(name)
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


LOCAL_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


@dataclass(slots=True)
class AppConfig:
    project_id: str = "locallife-thesis-depth"
    bucket: str = "gs://locallife-thesis-depth-data"
    results_dir: Path = field(default_factory=lambda: Path.cwd() / "artifacts")
    host: str = "127.0.0.1"
    port: int = 8000
    api_token: str = ""
    operating_mode: str = "waste"
    detector_model: str = "yoloe-11l-seg.pt"
    detector_confidence: float = 0.24
    detector_iou: float = 0.50
    image_size: int = 960
    prompts: tuple[str, ...] = DEFAULT_PROMPTS
    negative_prompts: tuple[str, ...] = DEFAULT_NEGATIVE_PROMPTS
    geometry_validation_prompts: tuple[str, ...] = DEFAULT_GEOMETRY_VALIDATION_PROMPTS
    depth_model: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    # RealSense is the only metric geometry authority.  Keeping the large
    # Logitech Depth-Anything model on by default made local CPU inference
    # take minutes while also producing a second, non-authoritative volume.
    # It remains opt-in through LOCALLIFE_ENABLE_DEPTH for comparison trials.
    enable_monocular_depth: bool = False
    enable_material_classification: bool = True
    material_model: str = "openai/clip-vit-base-patch32"
    material_labels: tuple[str, ...] = ()
    material_confidence_threshold: float = 0.35
    material_reclassify_frames: int = 15
    device: str = "auto"
    half_precision: bool = True
    batch_size: int = 6
    min_component_pixels: int = 700
    min_detection_area_fraction: float = 0.003
    max_detection_area_fraction: float = 0.62
    min_detection_side_fraction: float = 0.035
    foreground_threshold: int = 18
    allow_unclassified_foreground: bool = False
    bag_only: bool = False
    baseline_window_frames: int = 9
    volume_window_frames: int = 5
    minimum_depth_coverage: float = 0.70
    minimum_foreground_fraction: float = 0.12
    depth_noise_m: float = 0.004
    # Correction (round 16): "ray-frustum" was defaulted here in round 12 on
    # the claim that it is exact regardless of camera mounting tilt. That
    # claim was wrong in a way that mattered in practice, and real-hardware
    # testing (60+ L reported for a 1.5 L milk box) confirmed it. Look at
    # `estimate_volume()`'s ray-frustum branch: its volume sum is
    # `(reference_depth**3 - object_depth**3) / (3 * fx * fy)`, computed
    # directly from raw camera-Z depth at each pixel -- it never reads
    # `effective_height` (the plane-perpendicular, tilt-corrected height
    # from `_plane_perpendicular_height()`) at all. Only the *displayed*
    # height and the valid-pixel gating used the corrected value; the actual
    # reported liters for the camera's own default mode did not. So the
    # round-13 tilt fix was real and tested, but never actually reached the
    # number this project ships by default.
    # "reference-plane" is the mode whose volume sum genuinely uses the
    # corrected height: `contributions_m3 = object_height(effective) *
    # pixel_area_m2(at the table/baseline depth)` -- perpendicular height
    # times footprint-on-the-table area, which is the same principle as the
    # table-relative cuboid method (height above a fitted table plane, times
    # footprint) rather than raw camera-axis frustum geometry. This project's
    # actual rig is a wall/bracket mount "pointing downward into the bin"
    # (DUAL_CAMERA_THESIS.md / the thesis pilot deck), not a calibrated
    # overhead gantry, so a mode that actually uses the tilt correction
    # matters here, not just in theory. `ray-frustum`/`surface-columns` are
    # kept available (env override) for the documented thesis sensitivity
    # comparison; box-shaped detections additionally get the more robust
    # `estimate_box_volume_cuboid()` table-relative L*W*H measurement (see
    # volume.py and pipeline.py), which this default feeds as its fallback.
    # "height-map-grid" (the Final Implementation Playbook's locked method,
    # sections 3-7) supersedes "reference-plane" as the default. Every mode
    # above integrates once per *pixel*, weighting each pixel by z^2/(fx*fy) --
    # the footprint of a ray meeting a surface square-on. A crumpled polythene
    # bag, this system's actual target, is mostly oblique micro-facets where the
    # true footprint is larger by 1/cos(theta), and every pixel's own depth
    # noise enters the sum at full weight. "height-map-grid" instead bins the
    # backprojected 3-D points into fixed 10 mm cells on the calibrated bin
    # floor and takes a robust median height per cell, so the cell area is an
    # exact constant and tens of thousands of noisy samples collapse into a few
    # hundred medians. Measured against closed-form synthetic ground truth (a
    # rigid box at 0/15/30 degrees of mounting tilt, a smooth dome, and a
    # wrinkled dome with 4 mm depth noise plus 12% dropout), mean absolute
    # percentage error falls from about 15% to about 1%, the noisy wrinkled
    # cases improving most. The per-pixel modes stay available by env override
    # for the documented thesis sensitivity comparison.
    # Whether depositing an object rewrites the scene reference (and, on
    # Logitech, permanently marks its pixels occupied) so the object counts as
    # background from then on. That made sense when it was the only way to
    # measure the next bag incrementally in a bin that is never emptied -- but
    # it mutates the very reference that `detect_scene_objects` and
    # `estimate_volume` subtract from, with the object still physically in
    # frame. Present an object, deposit it, take it away, and the reference now
    # holds an object that is not there; after several sequential objects the
    # reference is a collage of all of them and the largest "changed" region is
    # background rather than the new object. Playbook sections 4 and 10 forbid
    # it outright ("the empty-bin calibration is the fixed coordinate
    # reference"), and the incremental measurement it existed to provide now
    # comes from `volume_before_l`/`volume_after_l`/`added_volume_l`, which
    # difference against that fixed baseline instead of moving it. Off by
    # default; set LOCALLIFE_ADVANCE_REFERENCE_ON_DEPOSIT=true only for a bin
    # whose contents genuinely stay put.
    # Cloud deployment (Windows launcher -> gpu.py -> GPU VM). The backend
    # itself always runs the same pipeline; these describe which deployment it
    # was started under so the operator page can show it, and are reported as
    # unknown rather than guessed when the launcher did not set them.
    cloud_enabled: bool = False
    # Blank means "use project_id"; resolved by `gcp_project`, so the cloud
    # target is configured in one place rather than two.
    cloud_project: str = ""
    cloud_vm_name: str = "depth-l4"
    cloud_zone: str = ""
    cloud_vm_status: str = ""
    pi_host: str = "locallife@locallife.local"
    # Welcome-page / run-mode selector (launcher_service.py).
    local_api_port: int = 8765
    default_run_mode: str = "ask"
    allow_local_fallback: bool = True
    cloud_startup_timeout_seconds: int = 90
    # Per-stage budgets. Defaults come from measured startup, not from guesses:
    # a real run brought the VM up in 79 s with capacity in the usual zone, so
    # 300 s covers that with room for a slow boot while still failing long
    # before an operator gives up. Host-key verification is two authenticated
    # API calls, so it is quick or it is broken. The frame timeout is the one
    # that catches a camera that never starts sending -- previously an
    # indefinite wait on a screen that just said "starting".
    # Comparison / Benchmark Mode. Off by default: collecting per-frame samples
    # is cheap but not free, and a demonstration should not pay for it.
    benchmark_mode: bool = False
    # Names the recorded clip both modes process. Without it the comparison
    # refuses to declare a winner, because two different live moments are not
    # the same input.
    benchmark_input_id: str = ""
    # Benchmark evidence recording (evidence.py). Off by default.
    record_benchmark_evidence: bool = False
    vm_start_timeout_seconds: int = 300
    ssh_verify_timeout_seconds: int = 60
    backend_ready_timeout_seconds: int = 180
    tunnel_ready_timeout_seconds: int = 60
    pi_connect_timeout_seconds: int = 90
    first_frame_timeout_seconds: int = 120
    cloud_request_timeout_seconds: int = 30
    bin_waste_stream: str = ""
    advance_reference_on_deposit: bool = False
    # A support plane is only valid for the pose it was fitted at. Beyond these
    # tolerances the calibration is marked invalid rather than silently reused
    # against a camera that has been moved, tilted or repositioned.
    camera_move_max_tilt_deg: float = 3.0
    camera_move_max_distance_m: float = 0.05
    volume_geometry: str = "height-map-grid"
    # Playbook section 27 tunables. These are its own suggested starting points,
    # not validated constants -- record the final values after tuning on the
    # real camera and bin.
    volume_grid_size_m: float = 0.010
    volume_min_points_per_cell: int = 3
    volume_cell_height_percentile: float = 50.0
    volume_calibration_factor: float = 1.0
    systematic_error_fraction: float = 0.025
    depth_noise_sigma: float = 3.0
    reject_depth_outliers: bool = True
    logitech_reference_distance_m: float = 0.0
    # Thesis research mode: "paired" (RealSense and Logitech side by side),
    # "realsense_only" or "logitech_only". Fusion is never implied by any.
    research_mode: str = "paired"
    # Two finalised camera measurements this close in time are one physical object.
    comparison_pair_window_s: float = 60.0
    # Smallest Logitech object mask accepted (pixels). The shared
    # min_component_pixels (700) rejected cans and cream jars outright; the
    # Logitech mask is already foreground-gated and opened, so noise is not the
    # limit there.
    logitech_min_object_pixels: int = 150
    # Logitech detector confidence; same as detector_confidence unless tuned
    # from the /api/cameras/logitech/diagnose-detector raw predictions.
    logitech_detector_confidence: float = 0.24
    # Flat packets and thin cartons: the RealSense minimum (25 mm) hid them.
    logitech_min_object_height_m: float = 0.010
    # Support-plane cell size for the Logitech height map (metres).
    logitech_height_map_cell_m: float = 0.005
    # Pixels trimmed from the Logitech mask before volume only: monocular depth
    # bleeds across an object's rim, and that rim is floor.
    logitech_volume_erode_px: int = 2
    # A confirmed Logitech track waits this many frames for a stable volume;
    # after that the median of what it has is finalised rather than staying
    # pending for ever.
    logitech_measurement_timeout_frames: int = 12
    # A confirmed object that has not settled after this many frames is
    # recorded once as rejected (unstable_volume / no_valid_measurement).
    finalise_max_frames: int = 45
    # Save per-measurement and periodic scene bundles for real-hardware debugging.
    hardware_diagnostic: bool = False
    hardware_diagnostic_interval_s: float = 5.0
    # Minimum seconds between Logitech Depth Anything V2 inferences; the last
    # prediction is reused in between. 0 = every frame (cloud GPU). A local CPU
    # run throttles so the shared inference lock never starves RealSense.
    logitech_depth_interval_s: float = 0.0
    logitech_horizontal_fov_deg: float = 70.42
    logitech_roi: tuple[float, float, float, float] | None = None
    logitech_max_scene_fraction: float = 0.45
    logitech_max_mask_expansion: float = 2.0
    # 120 L was sized for a full wheelie bin, not a single tracked bag/box --
    # loose enough that a false detection spanning almost the whole camera
    # frame (a hand or a pillow held close to the lens during testing, or a
    # monocular-depth hallucination on a shadow) could integrate to ~116 L
    # and slip through as "plausible" instead of being rejected. 90 L still
    # comfortably covers a large real waste bag/box (a 35 cm tall, 50x50 cm
    # footprint box is ~87 L) while catching near-full-frame false reads.
    logitech_max_item_volume_l: float = 90.0
    realsense_max_item_volume_l: float = 90.0
    logitech_stabilize_depth: bool = True
    logitech_require_reference: bool = True
    logitech_allow_provisional_metric: bool = False
    logitech_provisional_systematic_error_fraction: float = 0.35
    logitech_require_overhead: bool = True
    logitech_max_tilt_degrees: float = 35.0
    logitech_hard_max_tilt_degrees: float = 65.0
    logitech_max_tilt_uncertainty_fraction: float = 0.50
    logitech_duplicate_overlap: float = 0.55
    logitech_min_valid_height_fraction: float = 0.35
    # Depth-Anything-V2's monocular metric depth is a learned estimate, not a
    # real stereo/ToF measurement -- its frame-to-frame jitter on an
    # untextured, unchanged background is routinely well above the ~1-2 cm
    # noise floor a real depth sensor (RealSense) has. Reusing the shared,
    # RealSense-tuned min_object_height_m (2.5 cm) as the depth_changed gate
    # in detect_scene_objects() let that jitter register as "changed" across
    # large, DENSE swaths of background -- dense enough to pass the
    # solid-fill density gate (which targets sparse-bridged artifacts, not
    # uniformly-noisy-but-densely-changed regions) -- producing the
    # consistent full-frame Logitech bounding box seen on real hardware
    # while RealSense (real depth) stayed tight. Only the scene-object
    # detection that feeds phantom recovery (fuse_scene_detections) uses
    # this; it does not touch actual volume/height integration for already-
    # confirmed detections.
    logitech_scene_min_height_m: float = 0.05
    # Off by default: the recipe pipeline (pointcloud_volume.py/recipe_*.py)
    # is an additive, separately-documented result -- a second, independent
    # volume/color/material estimate shown alongside (never replacing) the
    # dashboard's own tracked/ledgered measurement above. It uses a generic,
    # uncurated YOLOv8n/11n-seg detector rather than this project's curated
    # YOLOE prompt vocabulary (see recipe_detect.py's own docstring), so
    # enabling it trades some of that vocabulary's false-positive rejection
    # for the recipe's exact spec; it also loads its own separate CLIP
    # instance for material classification even when the dashboard's own
    # MaterialClassifier is already loaded, using extra memory. Both are
    # acceptable once explicitly opted into, but neither is appropriate to
    # turn on silently for every existing installation.
    recipe_enabled: bool = False
    # Recomputing the recipe pipeline (a full YOLO segmentation pass plus a
    # CLIP material classification) on every dashboard poll (~550ms) would
    # be far more inference work than the existing measurement path does per
    # poll. This throttles it to at most once per this many seconds; state()
    # serves the cached result in between.
    recipe_refresh_seconds: float = 3.0
    comparison_match_seconds: float = 8.0
    bin_capacity_l: float = 0.0
    bin_polygon: tuple[tuple[float, float], ...] = ()
    color_waste_streams: dict[str, str] = field(default_factory=dict)
    auto_deposit: bool = True
    # Retained so existing callers and saved settings keep loading, but it no
    # longer switches anything off: `ledger_active` is unconditionally True.
    # See its docstring for why recording stopped being optional.
    ledger_enabled: bool | None = None
    # Validation/geometry mode is a developer diagnostic, not a run mode. When
    # set, rows are still persisted -- they are marked, so a diagnostic session
    # can never be mistaken for a measurement one.
    diagnostic_mode: bool = False
    # Stability window before a measurement may be finalised (stable_identity.py).
    # Sized from the cameras in use, not from round numbers: at 848x480 a 40 px
    # centroid drift is ~5% of frame width, and 25 mm sits above the D435's own
    # depth noise at bin distance without accepting a lift.
    # Frames of median smoothing for reported dimensions (footprint.py).
    dimension_smoothing_frames: int = 9
    stability_window_frames: int = 12
    min_valid_stable_frames: int = 6
    max_centroid_shift_px: float = 40.0
    min_mask_iou: float = 0.45
    max_depth_change_mm: float = 25.0
    max_volume_variation_percent: float = 12.0
    finalisation_hold_seconds: float = 1.0
    settle_frames: int = 5
    settle_volume_tolerance: float = 0.12
    volume_stability_frames: int = 3
    # Table-relative box cuboid: multi-frame track aggregation (Revised
    # Dual-Camera Volume Estimation recipe, section 13). Once a track has at
    # least `box_aggregation_min_frames` accepted `estimate_box_volume_cuboid()`
    # results, the reported L/W/H/volume come from the median across the
    # most recent `box_aggregation_window_frames` of them (never a per-frame
    # sum) instead of the single current frame -- the PDF's own recommended
    # 20-30-frame capture is the upper end of `box_aggregation_window_frames`;
    # the smaller default here keeps a live dashboard responsive while still
    # rejecting single noisy frames. A track with fewer accepted frames than
    # the minimum still reports its current single-frame cuboid result (the
    # pre-existing behaviour), so early frames are never hidden.
    box_aggregation_min_frames: int = 3
    box_aggregation_window_frames: int = 20
    history_limit: int = 100
    min_object_height_m: float = 0.025
    # The annotated laptop sleeve is 20 mm thick, below the production
    # waste-noise gate. Validation mode lowers this only for reference-object
    # geometry; waste mode keeps the safer 25 mm threshold unchanged.
    geometry_validation_min_object_height_m: float = 0.010
    max_object_height_m: float = 0.80
    roi: tuple[float, float, float, float] = (0.05, 0.05, 0.90, 0.90)
    auto_count: bool = True
    tracker_confirm_frames: int = 2
    tracker_max_missing_frames: int = 24
    tracker_phantom_max_missing_frames: int = 2
    tracker_live_prediction_frames: int = 6
    tracker_minimum_iou: float = 0.08
    tracker_max_center_distance: float = 0.38
    automatic_baseline: bool = True
    automatic_baseline_frames: int = 9
    automatic_baseline_motion_threshold: float = 3.0
    sync_interval_seconds: int = 180
    enable_bucket_sync: bool = True
    max_upload_mb: int = 24
    restore_saved_baseline: bool = False
    saved_baseline_validation_frames: int = 3
    saved_baseline_rgb_threshold: int = 24
    saved_baseline_max_changed_fraction: float = 0.65
    record_only_measured_objects: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        defaults = cls()
        roi_string = os.environ.get("LOCALLIFE_ROI", "")
        roi = defaults.roi
        if roi_string:
            values = tuple(float(value.strip()) for value in roi_string.split(","))
            if len(values) != 4:
                raise ValueError("LOCALLIFE_ROI must contain x,y,width,height")
            roi = values

        logitech_roi_string = os.environ.get("LOCALLIFE_LOGITECH_ROI", "").strip()
        logitech_roi = None
        if logitech_roi_string:
            values = tuple(float(value.strip()) for value in logitech_roi_string.split(","))
            if len(values) != 4:
                raise ValueError("LOCALLIFE_LOGITECH_ROI must contain x,y,width,height")
            logitech_roi = values

        polygon_string = os.environ.get("LOCALLIFE_BIN_POLYGON", "").strip()
        polygon: tuple[tuple[float, float], ...] = ()
        if polygon_string:
            try:
                polygon = tuple(
                    tuple(float(coordinate) for coordinate in vertex.split(":"))
                    for vertex in polygon_string.split(",")
                )
            except ValueError as exc:
                raise ValueError("LOCALLIFE_BIN_POLYGON must use x:y,x:y,x:y coordinates") from exc
            if any(len(vertex) != 2 for vertex in polygon):
                raise ValueError("LOCALLIFE_BIN_POLYGON vertices must each contain x:y")

        color_streams: dict[str, str] = {}
        for pair in os.environ.get("LOCALLIFE_COLOR_MAP", "").split(","):
            if not pair.strip():
                continue
            if ":" not in pair:
                raise ValueError("LOCALLIFE_COLOR_MAP must use color:stream,color:stream entries")
            color, stream = pair.split(":", 1)
            if not color.strip() or not stream.strip():
                raise ValueError("Each color mapping requires both a color and waste stream")
            color_streams[color.strip().lower()] = stream.strip()

        return cls(
            project_id=os.environ.get("LOCALLIFE_GCP_PROJECT", defaults.project_id),
            bucket=os.environ.get("LOCALLIFE_BUCKET", defaults.bucket).rstrip("/"),
            results_dir=Path(os.environ.get("LOCALLIFE_RESULTS_DIR", str(defaults.results_dir))),
            host=os.environ.get("LOCALLIFE_HOST", defaults.host),
            port=int(os.environ.get("LOCALLIFE_PORT", defaults.port)),
            api_token=os.environ.get("LOCALLIFE_API_TOKEN", ""),
            operating_mode=os.environ.get(
                "LOCALLIFE_OPERATING_MODE", defaults.operating_mode,
            ).strip().lower().replace("-", "_"),
            detector_model=os.environ.get("LOCALLIFE_DETECTOR_MODEL", defaults.detector_model),
            detector_confidence=float(os.environ.get("LOCALLIFE_CONF", defaults.detector_confidence)),
            detector_iou=float(os.environ.get("LOCALLIFE_IOU", defaults.detector_iou)),
            image_size=int(os.environ.get("LOCALLIFE_IMAGE_SIZE", defaults.image_size)),
            prompts=_tuple_env("LOCALLIFE_PROMPTS", DEFAULT_PROMPTS),
            negative_prompts=_tuple_env(
                "LOCALLIFE_NEGATIVE_PROMPTS", DEFAULT_NEGATIVE_PROMPTS,
            ),
            geometry_validation_prompts=_tuple_env(
                "LOCALLIFE_VALIDATION_PROMPTS", DEFAULT_GEOMETRY_VALIDATION_PROMPTS,
            ),
            # The Logitech comparison needs Depth Anything V2 in every mode, so
            # a launched app enables it; local CPU runs get the Small metric
            # checkpoint (the Large one took minutes per frame on a laptop CPU)
            # and a throttled rate, the cloud GPU gets Large at full rate.
            depth_model=os.environ.get(
                "LOCALLIFE_DEPTH_MODEL",
                defaults.depth_model if _bool_env("CLOUD_ENABLED", False) else LOCAL_DEPTH_MODEL,
            ),
            enable_monocular_depth=_bool_env("LOCALLIFE_ENABLE_DEPTH", True),
            logitech_depth_interval_s=float(os.environ.get(
                "LOCALLIFE_LOGITECH_DEPTH_INTERVAL_S", "0" if _bool_env("CLOUD_ENABLED", False) else "1.0",
            )),
            enable_material_classification=_bool_env(
                "LOCALLIFE_ENABLE_MATERIAL", defaults.enable_material_classification,
            ),
            material_model=os.environ.get("LOCALLIFE_MATERIAL_MODEL", defaults.material_model),
            material_labels=_tuple_env("LOCALLIFE_MATERIAL_LABELS", defaults.material_labels),
            material_confidence_threshold=float(os.environ.get(
                "LOCALLIFE_MATERIAL_CONF", defaults.material_confidence_threshold,
            )),
            material_reclassify_frames=int(os.environ.get(
                "LOCALLIFE_MATERIAL_RECLASSIFY_FRAMES", defaults.material_reclassify_frames,
            )),
            device=os.environ.get("LOCALLIFE_DEVICE", "auto"),
            half_precision=_bool_env("LOCALLIFE_HALF", True),
            batch_size=int(os.environ.get("LOCALLIFE_BATCH_SIZE", defaults.batch_size)),
            min_component_pixels=int(os.environ.get("LOCALLIFE_MIN_PIXELS", defaults.min_component_pixels)),
            min_detection_area_fraction=float(os.environ.get(
                "LOCALLIFE_MIN_DETECTION_AREA_FRACTION", defaults.min_detection_area_fraction,
            )),
            max_detection_area_fraction=float(os.environ.get(
                "LOCALLIFE_MAX_DETECTION_AREA_FRACTION", defaults.max_detection_area_fraction,
            )),
            min_detection_side_fraction=float(os.environ.get(
                "LOCALLIFE_MIN_DETECTION_SIDE_FRACTION", defaults.min_detection_side_fraction,
            )),
            foreground_threshold=int(os.environ.get("LOCALLIFE_FOREGROUND_THRESHOLD", defaults.foreground_threshold)),
            allow_unclassified_foreground=_bool_env("LOCALLIFE_ALLOW_UNCLASSIFIED", False),
            bag_only=_bool_env("LOCALLIFE_BAG_ONLY", False),
            baseline_window_frames=int(os.environ.get("LOCALLIFE_BASELINE_FRAMES", defaults.baseline_window_frames)),
            volume_window_frames=int(os.environ.get("LOCALLIFE_VOLUME_FRAMES", defaults.volume_window_frames)),
            minimum_depth_coverage=float(os.environ.get("LOCALLIFE_MIN_DEPTH_COVERAGE", defaults.minimum_depth_coverage)),
            minimum_foreground_fraction=float(os.environ.get(
                "LOCALLIFE_MIN_FOREGROUND_FRACTION", defaults.minimum_foreground_fraction,
            )),
            depth_noise_m=float(os.environ.get("LOCALLIFE_DEPTH_NOISE_M", defaults.depth_noise_m)),
            camera_move_max_tilt_deg=float(os.environ.get(
                "LOCALLIFE_CAMERA_MOVE_MAX_TILT_DEG", defaults.camera_move_max_tilt_deg,
            )),
            camera_move_max_distance_m=float(os.environ.get(
                "LOCALLIFE_CAMERA_MOVE_MAX_DISTANCE_M", defaults.camera_move_max_distance_m,
            )),
            cloud_enabled=_bool_env("CLOUD_ENABLED", defaults.cloud_enabled),
            cloud_project=os.environ.get("LOCALLIFE_GCP_PROJECT", defaults.cloud_project),
            cloud_vm_name=os.environ.get("LOCALLIFE_VM_NAME", defaults.cloud_vm_name),
            cloud_zone=os.environ.get("LOCALLIFE_VM_ZONE", defaults.cloud_zone),
            cloud_vm_status=os.environ.get("LOCALLIFE_VM_STATUS", defaults.cloud_vm_status),
            pi_host=os.environ.get("PI_HOST", os.environ.get("LOCALLIFE_PI_HOST", defaults.pi_host)),
            local_api_port=int(os.environ.get("LOCAL_API_PORT", defaults.local_api_port)),
            default_run_mode=os.environ.get("DEFAULT_RUN_MODE", defaults.default_run_mode),
            allow_local_fallback=_bool_env("ALLOW_LOCAL_FALLBACK", defaults.allow_local_fallback),
            cloud_startup_timeout_seconds=int(os.environ.get(
                "CLOUD_STARTUP_TIMEOUT_SECONDS", defaults.cloud_startup_timeout_seconds)),
            diagnostic_mode=_bool_env("LOCALLIFE_DIAGNOSTIC_MODE", defaults.diagnostic_mode),
            dimension_smoothing_frames=int(os.environ.get(
                "DIMENSION_SMOOTHING_FRAMES", defaults.dimension_smoothing_frames)),
            stability_window_frames=int(os.environ.get(
                "STABILITY_WINDOW_FRAMES", defaults.stability_window_frames)),
            min_valid_stable_frames=int(os.environ.get(
                "MIN_VALID_STABLE_FRAMES", defaults.min_valid_stable_frames)),
            max_centroid_shift_px=float(os.environ.get(
                "MAX_CENTROID_SHIFT_PX", defaults.max_centroid_shift_px)),
            min_mask_iou=float(os.environ.get("MIN_MASK_IOU", defaults.min_mask_iou)),
            max_depth_change_mm=float(os.environ.get(
                "MAX_DEPTH_CHANGE_MM", defaults.max_depth_change_mm)),
            max_volume_variation_percent=float(os.environ.get(
                "MAX_VOLUME_VARIATION_PERCENT", defaults.max_volume_variation_percent)),
            finalisation_hold_seconds=float(os.environ.get(
                "FINALISATION_HOLD_SECONDS", defaults.finalisation_hold_seconds)),
            benchmark_mode=_bool_env("LOCALLIFE_BENCHMARK_MODE", defaults.benchmark_mode),
            benchmark_input_id=os.environ.get(
                "LOCALLIFE_BENCHMARK_INPUT_ID", defaults.benchmark_input_id),
            record_benchmark_evidence=_bool_env(
                "LOCALLIFE_RECORD_EVIDENCE", defaults.record_benchmark_evidence),
            vm_start_timeout_seconds=int(os.environ.get(
                "VM_START_TIMEOUT_SECONDS", defaults.vm_start_timeout_seconds)),
            ssh_verify_timeout_seconds=int(os.environ.get(
                "SSH_VERIFY_TIMEOUT_SECONDS", defaults.ssh_verify_timeout_seconds)),
            backend_ready_timeout_seconds=int(os.environ.get(
                "BACKEND_READY_TIMEOUT_SECONDS", defaults.backend_ready_timeout_seconds)),
            tunnel_ready_timeout_seconds=int(os.environ.get(
                "TUNNEL_READY_TIMEOUT_SECONDS", defaults.tunnel_ready_timeout_seconds)),
            pi_connect_timeout_seconds=int(os.environ.get(
                "PI_CONNECT_TIMEOUT_SECONDS", defaults.pi_connect_timeout_seconds)),
            first_frame_timeout_seconds=int(os.environ.get(
                "FIRST_FRAME_TIMEOUT_SECONDS", defaults.first_frame_timeout_seconds)),
            cloud_request_timeout_seconds=int(os.environ.get(
                "CLOUD_REQUEST_TIMEOUT_SECONDS", defaults.cloud_request_timeout_seconds)),
            bin_waste_stream=os.environ.get("BIN_WASTE_STREAM", defaults.bin_waste_stream),
            advance_reference_on_deposit=_bool_env(
                "LOCALLIFE_ADVANCE_REFERENCE_ON_DEPOSIT", defaults.advance_reference_on_deposit,
            ),
            volume_geometry=os.environ.get("LOCALLIFE_VOLUME_GEOMETRY", defaults.volume_geometry),
            volume_grid_size_m=float(os.environ.get(
                "LOCALLIFE_VOLUME_GRID_SIZE_M", defaults.volume_grid_size_m,
            )),
            volume_min_points_per_cell=int(os.environ.get(
                "LOCALLIFE_VOLUME_MIN_POINTS_PER_CELL", defaults.volume_min_points_per_cell,
            )),
            volume_cell_height_percentile=float(os.environ.get(
                "LOCALLIFE_VOLUME_CELL_PERCENTILE", defaults.volume_cell_height_percentile,
            )),
            volume_calibration_factor=float(os.environ.get("LOCALLIFE_VOLUME_CALIBRATION_FACTOR", defaults.volume_calibration_factor)),
            systematic_error_fraction=float(os.environ.get("LOCALLIFE_SYSTEMATIC_ERROR_FRACTION", defaults.systematic_error_fraction)),
            depth_noise_sigma=float(os.environ.get("LOCALLIFE_DEPTH_NOISE_SIGMA", defaults.depth_noise_sigma)),
            reject_depth_outliers=_bool_env("LOCALLIFE_REJECT_DEPTH_OUTLIERS", True),
            logitech_reference_distance_m=float(os.environ.get("LOCALLIFE_LOGITECH_REFERENCE_DISTANCE_M", defaults.logitech_reference_distance_m)),
            research_mode=os.environ.get("LOCALLIFE_RESEARCH_MODE", defaults.research_mode).strip().lower(),
            comparison_pair_window_s=float(os.environ.get("LOCALLIFE_COMPARISON_PAIR_WINDOW_S", defaults.comparison_pair_window_s)),
            finalise_max_frames=int(os.environ.get("LOCALLIFE_FINALISE_MAX_FRAMES", defaults.finalise_max_frames)),
            logitech_detector_confidence=float(os.environ.get(
                "LOCALLIFE_LOGITECH_DETECTOR_CONFIDENCE",
                os.environ.get("LOCALLIFE_DETECTOR_CONFIDENCE", defaults.logitech_detector_confidence))),
            logitech_measurement_timeout_frames=int(os.environ.get(
                "LOCALLIFE_LOGITECH_MEASUREMENT_TIMEOUT_FRAMES", defaults.logitech_measurement_timeout_frames)),
            logitech_volume_erode_px=int(os.environ.get(
                "LOCALLIFE_LOGITECH_VOLUME_ERODE_PX", defaults.logitech_volume_erode_px)),
            logitech_height_map_cell_m=float(os.environ.get(
                "LOCALLIFE_LOGITECH_HEIGHT_MAP_CELL_M", defaults.logitech_height_map_cell_m)),
            logitech_min_object_height_m=float(os.environ.get(
                "LOCALLIFE_LOGITECH_MIN_OBJECT_HEIGHT_M", defaults.logitech_min_object_height_m)),
            logitech_min_object_pixels=int(os.environ.get(
                "LOCALLIFE_LOGITECH_MIN_OBJECT_PIXELS", defaults.logitech_min_object_pixels)),
            hardware_diagnostic=_bool_env("LOCALLIFE_HARDWARE_DIAGNOSTIC", defaults.hardware_diagnostic),
            hardware_diagnostic_interval_s=float(os.environ.get(
                "LOCALLIFE_HARDWARE_DIAGNOSTIC_INTERVAL_S", defaults.hardware_diagnostic_interval_s)),
            logitech_horizontal_fov_deg=float(os.environ.get("LOCALLIFE_LOGITECH_HORIZONTAL_FOV_DEG", defaults.logitech_horizontal_fov_deg)),
            logitech_roi=logitech_roi,
            logitech_max_scene_fraction=float(os.environ.get("LOCALLIFE_LOGITECH_MAX_SCENE_FRACTION", defaults.logitech_max_scene_fraction)),
            logitech_max_mask_expansion=float(os.environ.get("LOCALLIFE_LOGITECH_MAX_MASK_EXPANSION", defaults.logitech_max_mask_expansion)),
            logitech_max_item_volume_l=float(os.environ.get("LOCALLIFE_LOGITECH_MAX_ITEM_VOLUME_L", defaults.logitech_max_item_volume_l)),
            realsense_max_item_volume_l=float(os.environ.get(
                "LOCALLIFE_REALSENSE_MAX_ITEM_VOLUME_L", defaults.realsense_max_item_volume_l,
            )),
            logitech_stabilize_depth=_bool_env("LOCALLIFE_LOGITECH_STABILIZE_DEPTH", True),
            logitech_require_reference=_bool_env("LOCALLIFE_LOGITECH_REQUIRE_REFERENCE", True),
            logitech_allow_provisional_metric=_bool_env(
                "LOCALLIFE_LOGITECH_ALLOW_PROVISIONAL_METRIC",
                defaults.logitech_allow_provisional_metric,
            ),
            logitech_provisional_systematic_error_fraction=float(os.environ.get(
                "LOCALLIFE_LOGITECH_PROVISIONAL_SYSTEMATIC_ERROR_FRACTION",
                defaults.logitech_provisional_systematic_error_fraction,
            )),
            logitech_require_overhead=_bool_env("LOCALLIFE_LOGITECH_REQUIRE_OVERHEAD", True),
            logitech_max_tilt_degrees=float(os.environ.get("LOCALLIFE_LOGITECH_MAX_TILT_DEG", defaults.logitech_max_tilt_degrees)),
            logitech_hard_max_tilt_degrees=float(os.environ.get(
                "LOCALLIFE_LOGITECH_HARD_MAX_TILT_DEG", defaults.logitech_hard_max_tilt_degrees,
            )),
            logitech_max_tilt_uncertainty_fraction=float(os.environ.get(
                "LOCALLIFE_LOGITECH_MAX_TILT_UNCERTAINTY_FRACTION",
                defaults.logitech_max_tilt_uncertainty_fraction,
            )),
            logitech_duplicate_overlap=float(os.environ.get("LOCALLIFE_LOGITECH_DUPLICATE_OVERLAP", defaults.logitech_duplicate_overlap)),
            logitech_min_valid_height_fraction=float(os.environ.get("LOCALLIFE_LOGITECH_MIN_VALID_HEIGHT_FRACTION", defaults.logitech_min_valid_height_fraction)),
            logitech_scene_min_height_m=float(os.environ.get(
                "LOCALLIFE_LOGITECH_SCENE_MIN_HEIGHT_M", defaults.logitech_scene_min_height_m,
            )),
            recipe_enabled=_bool_env("LOCALLIFE_RECIPE_ENABLED", defaults.recipe_enabled),
            recipe_refresh_seconds=float(os.environ.get(
                "LOCALLIFE_RECIPE_REFRESH_SECONDS", defaults.recipe_refresh_seconds,
            )),
            comparison_match_seconds=float(os.environ.get("LOCALLIFE_COMPARISON_MATCH_SECONDS", defaults.comparison_match_seconds)),
            bin_capacity_l=float(os.environ.get("LOCALLIFE_BIN_CAPACITY_L", defaults.bin_capacity_l)),
            bin_polygon=polygon,
            color_waste_streams=color_streams,
            auto_deposit=_bool_env("LOCALLIFE_AUTO_DEPOSIT", True),
            settle_frames=int(os.environ.get("LOCALLIFE_SETTLE_FRAMES", defaults.settle_frames)),
            settle_volume_tolerance=float(os.environ.get("LOCALLIFE_SETTLE_TOLERANCE", defaults.settle_volume_tolerance)),
            volume_stability_frames=int(os.environ.get(
                "LOCALLIFE_VOLUME_STABILITY_FRAMES", defaults.volume_stability_frames,
            )),
            box_aggregation_min_frames=int(os.environ.get(
                "LOCALLIFE_BOX_AGGREGATION_MIN_FRAMES", defaults.box_aggregation_min_frames,
            )),
            box_aggregation_window_frames=int(os.environ.get(
                "LOCALLIFE_BOX_AGGREGATION_WINDOW_FRAMES", defaults.box_aggregation_window_frames,
            )),
            history_limit=int(os.environ.get("LOCALLIFE_HISTORY_LIMIT", defaults.history_limit)),
            min_object_height_m=float(os.environ.get("LOCALLIFE_MIN_HEIGHT_M", defaults.min_object_height_m)),
            geometry_validation_min_object_height_m=float(os.environ.get(
                "LOCALLIFE_VALIDATION_MIN_HEIGHT_M",
                defaults.geometry_validation_min_object_height_m,
            )),
            max_object_height_m=float(os.environ.get("LOCALLIFE_MAX_HEIGHT_M", defaults.max_object_height_m)),
            roi=roi,
            auto_count=_bool_env("LOCALLIFE_AUTO_COUNT", True),
            tracker_confirm_frames=int(os.environ.get("LOCALLIFE_TRACK_CONFIRM", defaults.tracker_confirm_frames)),
            tracker_max_missing_frames=int(os.environ.get("LOCALLIFE_TRACK_MISSING", defaults.tracker_max_missing_frames)),
            tracker_phantom_max_missing_frames=int(os.environ.get(
                "LOCALLIFE_TRACK_PHANTOM_MISSING", defaults.tracker_phantom_max_missing_frames,
            )),
            tracker_live_prediction_frames=int(os.environ.get(
                "LOCALLIFE_TRACK_PREDICTION_FRAMES", defaults.tracker_live_prediction_frames,
            )),
            tracker_minimum_iou=float(os.environ.get(
                "LOCALLIFE_TRACK_MIN_IOU", defaults.tracker_minimum_iou,
            )),
            tracker_max_center_distance=float(os.environ.get(
                "LOCALLIFE_TRACK_MAX_CENTER_DISTANCE", defaults.tracker_max_center_distance,
            )),
            automatic_baseline=_bool_env("LOCALLIFE_AUTOMATIC_BASELINE", defaults.automatic_baseline),
            automatic_baseline_frames=int(os.environ.get(
                "LOCALLIFE_AUTOMATIC_BASELINE_FRAMES", defaults.automatic_baseline_frames,
            )),
            automatic_baseline_motion_threshold=float(os.environ.get(
                "LOCALLIFE_AUTOMATIC_BASELINE_MOTION", defaults.automatic_baseline_motion_threshold,
            )),
            sync_interval_seconds=int(os.environ.get("LOCALLIFE_SYNC_INTERVAL", defaults.sync_interval_seconds)),
            enable_bucket_sync=_bool_env("LOCALLIFE_BUCKET_SYNC", True),
            max_upload_mb=int(os.environ.get("LOCALLIFE_MAX_UPLOAD_MB", defaults.max_upload_mb)),
            restore_saved_baseline=_bool_env("LOCALLIFE_RESTORE_SAVED_BASELINE", True),
            saved_baseline_validation_frames=int(os.environ.get(
                "LOCALLIFE_SAVED_BASELINE_VALIDATION_FRAMES",
                defaults.saved_baseline_validation_frames,
            )),
            saved_baseline_rgb_threshold=int(os.environ.get(
                "LOCALLIFE_SAVED_BASELINE_RGB_THRESHOLD",
                defaults.saved_baseline_rgb_threshold,
            )),
            saved_baseline_max_changed_fraction=float(os.environ.get(
                "LOCALLIFE_SAVED_BASELINE_MAX_CHANGED_FRACTION",
                defaults.saved_baseline_max_changed_fraction,
            )),
            record_only_measured_objects=_bool_env("LOCALLIFE_RECORD_ONLY_MEASURED", True),
        )

    @property
    def gcp_project(self) -> str:
        """The project cloud calls target; `cloud_project` overrides `project_id`."""
        return self.cloud_project or self.project_id

    @property
    def ledger_active(self) -> bool:
        """The measurement history is always on during normal operation.

        It used to follow `operating_mode`, and the launcher defaulted to
        geometry validation, so ordinary runs recorded nothing and downloaded an
        empty CSV. Recording is not a mode, a preference or a toggle: it is what
        the system is for. Nothing in the operator UI can switch it off.

        The one exception is an explicitly isolated developer diagnostic
        (`diagnostic_mode`), and even that does not disable persistence -- it
        only marks the rows it produces, so a diagnostic run can never be
        mistaken for a measurement session.
        """
        return True

    @property
    def processing_mode(self) -> str:
        """Which deployment produced a measurement: "local" or "cloud".

        Recorded on every event row so local and cloud results stay
        distinguishable in one export -- a benchmark comparison is meaningless
        if the rows do not say which side measured them.
        """
        return "cloud" if self.cloud_enabled else "local"

    def validate(self) -> None:
        if self.operating_mode not in {"waste", "geometry_validation"}:
            raise ValueError(
                "LOCALLIFE_OPERATING_MODE must be 'waste' or 'geometry_validation'"
            )
        x, y, width, height = self.roi
        if min(x, y) < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
            raise ValueError("ROI must fit within normalized image coordinates [0, 1]")
        if not 0 < self.detector_confidence < 1:
            raise ValueError("Detection confidence must be between 0 and 1")
        if self.batch_size < 1:
            raise ValueError("Batch size must be at least one")
        if self.max_object_height_m <= self.min_object_height_m:
            raise ValueError("Maximum object height must exceed minimum object height")
        if not 0 < self.geometry_validation_min_object_height_m < self.max_object_height_m:
            raise ValueError("Validation minimum object height must be positive and below the maximum")
        if self.logitech_scene_min_height_m <= 0 or self.max_object_height_m <= self.logitech_scene_min_height_m:
            raise ValueError("Logitech scene-detection minimum height must be positive and below the maximum object height")
        if self.recipe_refresh_seconds <= 0:
            raise ValueError("Recipe refresh interval must be positive")
        if self.baseline_window_frames < 1 or self.volume_window_frames < 1:
            raise ValueError("Baseline and volume smoothing windows must contain at least one frame")
        if self.settle_frames < 2 or self.history_limit < 1:
            raise ValueError("Settlement requires at least two frames and history must retain at least one record")
        if not 0 <= self.settle_volume_tolerance <= 1:
            raise ValueError("Settlement volume tolerance must be within [0, 1]")
        if not 0 < self.minimum_depth_coverage <= 1:
            raise ValueError("Minimum depth coverage must be within (0, 1]")
        if not 0 < self.minimum_foreground_fraction <= 1:
            raise ValueError("Minimum foreground fraction must be within (0, 1]")
        if not 0 < self.min_detection_area_fraction < self.max_detection_area_fraction <= 1:
            raise ValueError("Detection area fractions must be ordered within (0, 1]")
        if not 0 < self.min_detection_side_fraction < 1:
            raise ValueError("Minimum detection side fraction must be within (0, 1)")
        if self.depth_noise_m < 0 or self.bin_capacity_l < 0:
            raise ValueError("Depth noise and bin capacity cannot be negative")
        if self.volume_geometry not in {
            "surface-columns",
            "ray-frustum",
            "reference-plane",
            "triangulated-surface",
            "height-map-grid",
        }:
            raise ValueError(
                "Volume geometry must be height-map-grid, surface-columns, ray-frustum, "
                "reference-plane, or triangulated-surface"
            )
        if not 0.002 <= self.volume_grid_size_m <= 0.20:
            raise ValueError("Volume grid size must be between 2 mm and 200 mm")
        if self.volume_min_points_per_cell < 1:
            raise ValueError("Volume grid cells must require at least one depth point")
        if not 0 <= self.volume_cell_height_percentile <= 100:
            raise ValueError("Volume cell height percentile must be within [0, 100]")
        if not np.isfinite(self.volume_calibration_factor) or self.volume_calibration_factor <= 0:
            raise ValueError("Volume calibration factor must be finite and positive")
        if not 0 <= self.systematic_error_fraction <= 1 or self.depth_noise_sigma < 0:
            raise ValueError("Systematic error must be within [0, 1] and noise sigma cannot be negative")
        if self.research_mode not in ("paired", "realsense_only", "logitech_only"):
            raise ValueError("LOCALLIFE_RESEARCH_MODE must be paired, realsense_only or logitech_only")
        if self.comparison_pair_window_s <= 0:
            raise ValueError("LOCALLIFE_COMPARISON_PAIR_WINDOW_S must be positive")
        if self.logitech_reference_distance_m < 0 or not 1 < self.logitech_horizontal_fov_deg < 179:
            raise ValueError("Logitech reference distance cannot be negative and its field of view must be valid")
        if not 0 <= self.logitech_provisional_systematic_error_fraction <= 1:
            raise ValueError("Logitech provisional systematic error must be within [0, 1]")
        if not 0 < self.logitech_max_scene_fraction <= 1:
            raise ValueError("Logitech maximum object fraction must be within (0, 1]")
        if (self.logitech_max_mask_expansion < 1 or self.logitech_max_item_volume_l <= 0
                or self.realsense_max_item_volume_l <= 0):
            raise ValueError("Logitech mask expansion must be at least one and maximum liters must be positive")
        if not 0 < self.logitech_max_tilt_degrees < 90:
            raise ValueError("Logitech maximum mounting tilt must be between 0 and 90 degrees")
        if not self.logitech_max_tilt_degrees < self.logitech_hard_max_tilt_degrees < 90:
            raise ValueError(
                "Logitech hard maximum mounting tilt must be greater than the confident-zone "
                "maximum and less than 90 degrees"
            )
        if not 0 <= self.logitech_max_tilt_uncertainty_fraction <= 1:
            raise ValueError("Logitech tilt uncertainty fraction must be within [0, 1]")
        if not 0 < self.logitech_duplicate_overlap <= 1:
            raise ValueError("Logitech duplicate overlap must be within (0, 1]")
        if not 0 < self.logitech_min_valid_height_fraction <= 1:
            raise ValueError("Logitech minimum valid-height fraction must be within (0, 1]")
        if self.logitech_roi is not None:
            lx, ly, lw, lh = self.logitech_roi
            if min(lx, ly) < 0 or lw <= 0 or lh <= 0 or lx + lw > 1 or ly + lh > 1:
                raise ValueError("Logitech ROI must fit within normalized image coordinates [0, 1]")
        if self.comparison_match_seconds <= 0:
            raise ValueError("Comparison matching window must be positive")
        if self.bin_polygon and len(self.bin_polygon) < 3:
            raise ValueError("A fixed bin polygon requires at least three vertices")
        if any(not 0 <= coordinate <= 1 for vertex in self.bin_polygon for coordinate in vertex):
            raise ValueError("Fixed bin polygon coordinates must be normalized within [0, 1]")
        if self.saved_baseline_validation_frames < 1:
            raise ValueError("Saved-baseline validation requires at least one frame")
        if self.volume_stability_frames < 2:
            raise ValueError("Volume stabilization requires at least two frames")
        if self.box_aggregation_min_frames < 1:
            raise ValueError("Box dimension aggregation requires at least one frame")
        if self.box_aggregation_window_frames < self.box_aggregation_min_frames:
            raise ValueError("Box dimension aggregation window must be at least the minimum frame count")
        if self.tracker_live_prediction_frames < 0 or self.tracker_live_prediction_frames > self.tracker_max_missing_frames:
            raise ValueError("Live prediction frames must fit within the tracker missing-frame window")
        if not 0 <= self.tracker_minimum_iou <= 1 or self.tracker_max_center_distance <= 0:
            raise ValueError("Tracker IoU and center-distance thresholds are invalid")
        if self.tracker_phantom_max_missing_frames < 0:
            raise ValueError("Tracker phantom missing-frame budget cannot be negative")
        if self.automatic_baseline_frames < 2 or self.automatic_baseline_motion_threshold < 0:
            raise ValueError("Automatic baseline requires stable frames and a non-negative motion threshold")
        if not 0 <= self.saved_baseline_rgb_threshold <= 255:
            raise ValueError("Saved-baseline RGB threshold must be within [0, 255]")
        if not 0 <= self.saved_baseline_max_changed_fraction <= 1:
            raise ValueError("Saved-baseline changed fraction must be within [0, 1]")
        if not 0 <= self.material_confidence_threshold <= 1:
            raise ValueError("Material confidence threshold must be within [0, 1]")
        if self.material_reclassify_frames < 1:
            raise ValueError("Material reclassification interval must be at least one frame")
