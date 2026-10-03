"""LocalLife waste-vision: core Python source for review.

SOURCE
    Every module under locallife_cloud/ on this branch is byte-identical to
    commit e37b9bff97d032cecc7cddec6b3d77ee37375d95 on branch
    improvement/locallife-measurement-integrity (forked from
    V49_box_detection_measurement_fix @ d6fb60556a8d2b286c88a3c99ef21dca3f7c211a).
    Nothing was rewritten for presentation; files were only left out.
    This branch is for reading. It is NOT runnable: modules it imports but does
    not contain (listed at the end), model weights, launchers, configuration
    files (e.g. box_templates.yaml) and requirements are on the source branch.

CAMERA ROLES
    RealSense D435: RGB + factory-calibrated stereo depth, aligned to RGB. It is
    the only metric depth sensor and the only source of the reported volume.
    Logitech C920: RGB only. Its depth is a monocular prediction (Depth Anything
    V2) scaled by its own calibration; its volumes are a separate, independent
    estimate, labelled as such, and are never overwritten by RealSense values.

READING ORDER (camera -> result)
     1. edge_client.py, camera_recovery.py  Raspberry Pi capture/upload, reconnection.
     2. streaming.py                        newest-frame-only processing; dropped-frame counts.
     3. config.py, types.py                 every tunable; Detection / measurement records,
                                            including provenance fields (origin_source,
                                            observation_status fresh|predicted|held,
                                            measured_at, frame_id) and volume provenance.
     4. inference.py                        shared detector/segmenter (YOLOE), monocular depth.
     5. comparison.py                       DualCameraCoordinator: one detector batch for both
                                            cameras, per-camera assembly, live "fused" view
                                            (staleness expiry, explicit association status),
                                            counting summary, deposit matching.
     6. pipeline.py  VisionPipeline._assemble, in stages:
                       validation/filtering -> scene fusion & masks -> reference/support plane
                       -> geometry (estimate_volume, estimate_box_volume_cuboid, Logitech
                       metric_object_volume) -> quality gates -> tracking (tracking.py)
                       -> colour / material / class -> stability -> ledger & deposits
                       (_settle_deposits) -> support-surface estimate and sorting verdict
                       (_apply_support_geometry_and_class) -> output assembly.
     7. geometry.py                         masks, scene fusion, phantom identity
                                            (is_phantom_detection), colour categories.
     8. volume.py, heightmap_volume.py, footprint.py   RealSense geometry (equations below).
     9. logitech.py, logitech_volume.py     Logitech segmentation and monocular volume.
    10. bin_fill.py, object_class.py        support-surface object estimate; per-track class.
    11. colour_evidence.py, material*.py    colour and exterior material evidence.
    12. bin_policy.py, sorting_rules.py     mis-sort verdict (supported / uncertain).
    13. tracking.py, ledger.py, session_deposits.py, storage.py   identity and records.
    14. experiment_log.py, accuracy.py      per-attempt experiment log and evaluation.
    15. server.py, dashboard.py, telemetry.py   HTTP API, research page, timing telemetry.

VOLUME EQUATIONS (RealSense; single view)
    Back-projection: X = (u - cx) Z / fx, Y = (v - cy) Z / fy, Z = depth.
    Support plane fitted from the empty scene (preferred) or, per frame only and
    flagged "live_fitted_support_plane", from the live background around the object.
    Height above plane: h = (a X + b Y + c - Z) / sqrt(a^2 + b^2 + 1).
    Default "height-map-grid": points binned in 10 mm cells ON the plane, median h
    per cell, V = sum(cell_area * fraction * h). Interior cells count fully; a
    boundary cell counts the floor area its elevated samples cover,
    a_i = Z^3 / (fx fy |n . p|)  (patch carried along its ray onto the plane).
    "table_relative_cuboid" (boxes only): H = median of the top-surface cluster;
    L, W = trimmed extents along the minimum-area rectangle orientation, with the
    derived boundary bias added back (trim: /(1 - 2p/100); mask erosion e pixels and
    half a pixel per side, only on sides bounded by a top-face edge); V = L W H.
    Reported volume = raw geometric volume x known-volume factor; dimensions are
    never scaled; raw, factor and relationship are exported with every reading.
    Uncertainty is method-matched: propagated heuristic 1-sigma terms (segmentation
    boundary, erosion compensation, depth noise, support-plane RMSE, frame spread).
    It is NOT a statistically calibrated interval.
    Not observed from one view: the underside and any face hidden from the camera;
    a partly occluded object reads as its visible part (frame-edge clipping is
    flagged as a lower bound, occlusion by another object is not detectable).
    Per-pixel modes ("surface-columns", "reference-plane") assume horizontal
    surfaces and over-count visible walls on tilted views; "ray-frustum" includes
    the occluded shadow volume. They remain only for sensitivity comparison.

MEASUREMENT VS PREDICTION
    Only observation_status == "fresh" readings vote for colour/material, enter the
    stability window, calibrate, or count as experiment trials. Tracker predictions
    and dropout holds are displayed with their age. A depth silhouette stays
    unconfirmed after its display source changes.

CALIBRATION VS EVALUATION
    Known-volume calibration (RealSense only; refused on Logitech, whose path does
    not use it) and the Logitech known-object factor are fitted on calibration
    objects. experiment_log.py records every attempt (success or the failure
    reason), freezes calibration during an evaluation run, excludes objects used
    for both roles, and reports MAE/RMSE/bias/MAPE over valid readings next to the
    failure and timeout rates. Templates (box_templates) and bag class snapping are
    opt-in, flagged, and excluded from geometric accuracy.

KNOWN LIMITATIONS
    * All geometry tests are synthetic (exact ray casting); physical accuracy has to
      be measured with real reference objects through the experiment log.
    * Monocular (Logitech) depth flattens relief; its volumes depend on its own
      known-object factor and carry larger, unquantified error.
    * Material is a zero-shot ranking score plus repeat agreement, not a calibrated
      probability; contents of opaque bags are not inferred.
    * Counting is camera evidence only: no PIR, ToF or load cell is integrated.
      "Stable measurement records" (ledger "deposited") are not physical drops.
    * The two cameras are not spatially registered; cross-camera association is an
      assumption that holds only with one object per view (otherwise "ambiguous").
    * Fusion weights in accuracy.fuse_pair_volume are engineering assumptions.

EXTERNAL DEPENDENCIES (not on this branch)
    Python: numpy, opencv-python, flask; torch, transformers, ultralytics (YOLOE),
    Depth Anything V2 and CLIP/SigLIP weights; pyrealsense2 on the Raspberry Pi.
    Internal modules imported but omitted (on the source branch): bin_fill_panel,
    bin_occupancy, bin_profile, comparison_panel, comparison_store, coordinates,
    cost_accuracy(+_panel), crowded_scene, debug_snapshot, deposit_state,
    depth_edges, diagnostics, duplicate_detections, event_log, excel_export,
    experiment_panel, finalized_record, logitech_* helpers, mask_leak,
    measurement_mask, measurement_zone, operator_dashboard, optional_imports,
    paired_events, readiness, recipe_*, research_layout, shape_geometry,
    shape_router, stable_identity, stable_tracking, waste_bag_names, benchmark.
    Not used by the active path, therefore omitted: calibration.py and fusion.py
    (pixel-level dual-camera fusion, not wired into pipeline.py).
"""
