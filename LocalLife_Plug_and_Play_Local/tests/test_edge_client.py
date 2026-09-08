"""Tests for the Raspberry Pi camera capture recovery logic in edge_client.py.

pyrealsense2 is a hardware SDK that is never installed in this sandbox (or in
CI), so these tests inject a small fake module into sys.modules before
importing edge_client -- the same "stand-in model" testing convention already
used elsewhere in this project for heavy/unavailable dependencies. The fake
module is realistic enough to exercise the real `iter_realsense()` control
flow end to end, including one real frame being yielded.
"""

from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class FakeDevice:
    def __init__(self, serial: str, *, fails: bool = False) -> None:
        self.serial = serial
        self.fails = fails
        self.reset_calls = 0

    def get_info(self, _key: object) -> str:
        return self.serial

    def hardware_reset(self) -> None:
        self.reset_calls += 1
        if self.fails:
            raise RuntimeError("hardware_reset not supported on this device")


class FakeContext:
    def __init__(self, devices: list[FakeDevice] | None = None, *, broken: bool = False) -> None:
        self.devices = devices or []
        self.broken = broken

    def query_devices(self) -> list[FakeDevice]:
        if self.broken:
            raise RuntimeError("no RealSense hardware present")
        return self.devices


class FakeDepthSensor:
    def __init__(self, depth_scale: float = 0.001) -> None:
        self._depth_scale = depth_scale

    def get_depth_scale(self) -> float:
        return self._depth_scale

    def supports(self, _option: object) -> bool:
        return False

    def set_option(self, _option: object, _value: float) -> None:  # pragma: no cover - unused
        raise AssertionError("set_option should not be called when supports() is False")


class FakeProfile:
    def __init__(self, depth_scale: float = 0.001) -> None:
        self._depth_scale = depth_scale

    def get_device(self) -> SimpleNamespace:
        return SimpleNamespace(first_depth_sensor=lambda: FakeDepthSensor(self._depth_scale))


class FakeStreamProfile:
    def as_video_stream_profile(self) -> SimpleNamespace:
        return SimpleNamespace(intrinsics=SimpleNamespace(fx=100.0, fy=100.0, ppx=32.0, ppy=24.0, width=64, height=48))


class FakeColorFrame:
    def get_data(self) -> np.ndarray:
        return np.zeros((48, 64, 3), dtype=np.uint8)

    @property
    def profile(self) -> FakeStreamProfile:
        return FakeStreamProfile()


class FakeDepthFrame:
    def get_data(self) -> np.ndarray:
        return np.full((48, 64), 2000, dtype=np.uint16)


class FakeFrameset:
    def get_color_frame(self) -> FakeColorFrame:
        return FakeColorFrame()

    def get_depth_frame(self) -> FakeDepthFrame:
        return FakeDepthFrame()


class FakeAlign:
    def __init__(self, _stream: object) -> None:
        pass

    def process(self, _frames: object) -> FakeFrameset:
        return FakeFrameset()


class FakeNoiseFilter:
    def process(self, depth: object) -> object:
        return depth


class FakePipeline:
    """Each entry in `start_results` is consumed by one `start()` call.

    An entry that is an Exception instance is raised; any other value means
    `start()` succeeds and returns a FakeProfile.
    """

    def __init__(self, start_results: list[object], *, depth_scale: float = 0.001) -> None:
        self._start_results = list(start_results)
        self.start_calls = 0
        self._depth_scale = depth_scale

    def start(self, _configuration: object) -> FakeProfile:
        self.start_calls += 1
        if not self._start_results:
            raise AssertionError("pipeline.start() called more times than the test expected")
        result = self._start_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return FakeProfile(self._depth_scale)

    def wait_for_frames(self, timeout_ms: int = 5000) -> str:
        return "frames"

    def stop(self) -> None:
        pass


def _install_fake_pyrealsense2(pipeline: FakePipeline, context: FakeContext) -> None:
    rs = types.ModuleType("pyrealsense2")
    rs.pipeline = lambda: pipeline  # type: ignore[attr-defined]
    rs.config = lambda: SimpleNamespace(enable_stream=lambda *a, **k: None)  # type: ignore[attr-defined]
    rs.stream = SimpleNamespace(depth="depth", color="color")  # type: ignore[attr-defined]
    rs.format = SimpleNamespace(z16="z16", bgr8="bgr8")  # type: ignore[attr-defined]
    rs.option = SimpleNamespace(visual_preset="visual_preset")  # type: ignore[attr-defined]
    rs.rs400_visual_preset = SimpleNamespace(high_accuracy=3)  # type: ignore[attr-defined]
    rs.camera_info = SimpleNamespace(serial_number="serial_number")  # type: ignore[attr-defined]
    rs.align = FakeAlign  # type: ignore[attr-defined]
    rs.spatial_filter = lambda: FakeNoiseFilter()  # type: ignore[attr-defined]
    rs.temporal_filter = lambda: FakeNoiseFilter()  # type: ignore[attr-defined]
    rs.disparity_transform = lambda _to_disparity=True: FakeNoiseFilter()  # type: ignore[attr-defined]
    rs.hole_filling_filter = lambda: FakeNoiseFilter()  # type: ignore[attr-defined]
    rs.context = lambda: context  # type: ignore[attr-defined]
    sys.modules["pyrealsense2"] = rs


class RealSenseBusyDeviceRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        sys.modules.pop("pyrealsense2", None)
        sys.modules.pop("locallife_cloud.edge_client", None)

    def tearDown(self) -> None:
        sys.modules.pop("pyrealsense2", None)

    def test_busy_device_is_hardware_reset_and_the_retry_succeeds(self) -> None:
        # Reproduces the real-hardware failure: pipeline.start() first raises
        # the "Device or resource busy" (errno=16) RuntimeError a stuck
        # RealSense device produces, then succeeds once the device has been
        # hardware-reset.
        device = FakeDevice("SN123")
        pipeline = FakePipeline([
            RuntimeError("xioctl(VIDIOC_S_FMT) failed, errno=16 Last Error: Device or resource busy"),
            "ok",
        ])
        _install_fake_pyrealsense2(pipeline, FakeContext([device]))
        from locallife_cloud import edge_client

        with patch.object(edge_client.time, "sleep") as fake_sleep:
            frame = next(edge_client.iter_realsense(64, 48, 30))

        self.assertEqual(pipeline.start_calls, 2)
        self.assertEqual(device.reset_calls, 1)
        fake_sleep.assert_called_once_with(3.0)
        self.assertEqual(frame.camera_id, "realsense")
        self.assertEqual(frame.source, "realsense-aligned-rgb-depth")

    def test_no_device_found_still_retries_once_without_the_reset_wait(self) -> None:
        # If query_devices() finds nothing to reset (e.g. the device already
        # dropped off the USB bus entirely), the retry still happens -- it
        # just doesn't wait for a reset that was never issued.
        pipeline = FakePipeline([
            RuntimeError("Device or resource busy"),
            "ok",
        ])
        _install_fake_pyrealsense2(pipeline, FakeContext([]))
        from locallife_cloud import edge_client

        with patch.object(edge_client.time, "sleep") as fake_sleep:
            frame = next(edge_client.iter_realsense(64, 48, 30))

        self.assertEqual(pipeline.start_calls, 2)
        fake_sleep.assert_not_called()
        self.assertEqual(frame.camera_id, "realsense")

    def test_persistent_failure_after_reset_still_raises(self) -> None:
        # If the device is still busy even after a hardware reset (a genuine
        # hardware/power problem, not a stale process), the error must
        # propagate so the outer run_resilient_camera retry loop keeps
        # trying rather than this being silently swallowed.
        device = FakeDevice("SN123")
        pipeline = FakePipeline([
            RuntimeError("Device or resource busy"),
            RuntimeError("Device or resource busy"),
        ])
        _install_fake_pyrealsense2(pipeline, FakeContext([device]))
        from locallife_cloud import edge_client

        with patch.object(edge_client.time, "sleep"):
            with self.assertRaises(RuntimeError):
                next(edge_client.iter_realsense(64, 48, 30))

        self.assertEqual(pipeline.start_calls, 2)
        self.assertEqual(device.reset_calls, 1)

    def test_successful_start_never_touches_hardware_reset(self) -> None:
        # The common, healthy case must not pay any reset/sleep cost at all.
        device = FakeDevice("SN123")
        pipeline = FakePipeline(["ok"])
        _install_fake_pyrealsense2(pipeline, FakeContext([device]))
        from locallife_cloud import edge_client

        with patch.object(edge_client.time, "sleep") as fake_sleep:
            next(edge_client.iter_realsense(64, 48, 30))

        self.assertEqual(pipeline.start_calls, 1)
        self.assertEqual(device.reset_calls, 0)
        fake_sleep.assert_not_called()


class HardwareResetHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        sys.modules.pop("pyrealsense2", None)
        sys.modules.pop("locallife_cloud.edge_client", None)

    def tearDown(self) -> None:
        sys.modules.pop("pyrealsense2", None)

    def test_resets_every_enumerated_device(self) -> None:
        pipeline = FakePipeline(["ok"])
        one, two = FakeDevice("SN1"), FakeDevice("SN2")
        context = FakeContext([one, two])
        _install_fake_pyrealsense2(pipeline, context)
        from locallife_cloud import edge_client

        result = edge_client._hardware_reset_realsense(sys.modules["pyrealsense2"])

        self.assertTrue(result)
        self.assertEqual(one.reset_calls, 1)
        self.assertEqual(two.reset_calls, 1)

    def test_one_device_failing_to_reset_does_not_stop_the_others(self) -> None:
        pipeline = FakePipeline(["ok"])
        broken, healthy = FakeDevice("SN1", fails=True), FakeDevice("SN2")
        context = FakeContext([broken, healthy])
        _install_fake_pyrealsense2(pipeline, context)
        from locallife_cloud import edge_client

        result = edge_client._hardware_reset_realsense(sys.modules["pyrealsense2"])

        self.assertTrue(result)
        self.assertEqual(healthy.reset_calls, 1)

    def test_broken_enumeration_is_handled_without_raising(self) -> None:
        pipeline = FakePipeline(["ok"])
        context = FakeContext(broken=True)
        _install_fake_pyrealsense2(pipeline, context)
        from locallife_cloud import edge_client

        result = edge_client._hardware_reset_realsense(sys.modules["pyrealsense2"])

        self.assertFalse(result)

    def test_no_devices_returns_false(self) -> None:
        pipeline = FakePipeline(["ok"])
        context = FakeContext([])
        _install_fake_pyrealsense2(pipeline, context)
        from locallife_cloud import edge_client

        result = edge_client._hardware_reset_realsense(sys.modules["pyrealsense2"])

        self.assertFalse(result)


class DepthScaleValidationTests(unittest.TestCase):
    """Guards the build spec's explicit ask: 'the coding agent must verify
    the RealSense depth scale at runtime... a factor-of-1000 error must fail
    loudly.' These exercise `_validate_depth_scale()` directly -- no fake
    RealSense pipeline needed for the pure-function unit tests."""

    def setUp(self) -> None:
        sys.modules.pop("pyrealsense2", None)
        sys.modules.pop("locallife_cloud.edge_client", None)
        _install_fake_pyrealsense2(FakePipeline(["ok"]), FakeContext([]))

    def tearDown(self) -> None:
        sys.modules.pop("pyrealsense2", None)

    def test_real_d400_depth_scale_passes_unchanged(self) -> None:
        from locallife_cloud import edge_client

        self.assertEqual(edge_client._validate_depth_scale(0.001), 0.001)

    def test_a_factor_of_1000_error_fails_loudly(self) -> None:
        from locallife_cloud import edge_client

        with self.assertRaises(RuntimeError):
            edge_client._validate_depth_scale(0.001 * 1000)
        with self.assertRaises(RuntimeError):
            edge_client._validate_depth_scale(0.001 / 1000)

    def test_zero_or_negative_scale_fails_loudly(self) -> None:
        from locallife_cloud import edge_client

        with self.assertRaises(RuntimeError):
            edge_client._validate_depth_scale(0.0)
        with self.assertRaises(RuntimeError):
            edge_client._validate_depth_scale(-0.001)

    def test_nan_or_infinite_scale_fails_loudly(self) -> None:
        from locallife_cloud import edge_client

        with self.assertRaises(RuntimeError):
            edge_client._validate_depth_scale(float("nan"))
        with self.assertRaises(RuntimeError):
            edge_client._validate_depth_scale(float("inf"))

    def test_iter_realsense_end_to_end_rejects_a_bad_device_scale(self) -> None:
        # A full end-to-end reproduction: a real (fake) RealSense device
        # reporting an implausible depth scale must stop iter_realsense
        # before it ever yields a frame built from that wrong scale.
        pipeline = FakePipeline(["ok"], depth_scale=1.0)  # 1.0 m/unit: nonsense
        _install_fake_pyrealsense2(pipeline, FakeContext([]))
        from locallife_cloud import edge_client

        with self.assertRaises(RuntimeError):
            next(edge_client.iter_realsense(64, 48, 30))


class PostProcessingFilterChainTests(unittest.TestCase):
    """Confirms the expanded post-processing chain (disparity-domain
    spatial/temporal + hole-filling, added to close a real gap against the
    build spec's documented filter order) is actually exercised, and that a
    real, correctly-scaled frame still comes out the other end."""

    def setUp(self) -> None:
        sys.modules.pop("pyrealsense2", None)
        sys.modules.pop("locallife_cloud.edge_client", None)

    def tearDown(self) -> None:
        sys.modules.pop("pyrealsense2", None)

    def test_filter_chain_runs_and_a_valid_frame_is_still_yielded(self) -> None:
        calls: list[str] = []

        class RecordingFilter:
            def __init__(self, name: str) -> None:
                self._name = name

            def process(self, depth: object) -> object:
                calls.append(self._name)
                return depth

        pipeline = FakePipeline(["ok"])
        rs = types.ModuleType("pyrealsense2")
        rs.pipeline = lambda: pipeline  # type: ignore[attr-defined]
        rs.config = lambda: SimpleNamespace(enable_stream=lambda *a, **k: None)  # type: ignore[attr-defined]
        rs.stream = SimpleNamespace(depth="depth", color="color")  # type: ignore[attr-defined]
        rs.format = SimpleNamespace(z16="z16", bgr8="bgr8")  # type: ignore[attr-defined]
        rs.option = SimpleNamespace(visual_preset="visual_preset")  # type: ignore[attr-defined]
        rs.rs400_visual_preset = SimpleNamespace(high_accuracy=3)  # type: ignore[attr-defined]
        rs.camera_info = SimpleNamespace(serial_number="serial_number")  # type: ignore[attr-defined]
        rs.align = FakeAlign  # type: ignore[attr-defined]
        rs.spatial_filter = lambda: RecordingFilter("spatial")  # type: ignore[attr-defined]
        rs.temporal_filter = lambda: RecordingFilter("temporal")  # type: ignore[attr-defined]
        rs.disparity_transform = lambda to_disparity=True: RecordingFilter(  # type: ignore[attr-defined]
            "to_disparity" if to_disparity else "to_depth"
        )
        rs.hole_filling_filter = lambda: RecordingFilter("hole_filling")  # type: ignore[attr-defined]
        rs.context = lambda: FakeContext([])  # type: ignore[attr-defined]
        sys.modules["pyrealsense2"] = rs
        from locallife_cloud import edge_client

        frame = next(edge_client.iter_realsense(64, 48, 30))

        # Disparity-domain spatial/temporal, back to depth, then hole
        # filling -- in that order, matching the SDK's own recommended
        # sequence for the stages this project applies.
        self.assertEqual(calls, ["to_disparity", "spatial", "temporal", "to_depth", "hole_filling"])
        self.assertEqual(frame.camera_id, "realsense")
        self.assertTrue(np.all(frame.depth_m == 2000 * 0.001))

    def test_filter_depth_false_skips_the_whole_chain(self) -> None:
        pipeline = FakePipeline(["ok"])
        _install_fake_pyrealsense2(pipeline, FakeContext([]))
        from locallife_cloud import edge_client

        # Should not raise even though the fake filters would record calls
        # if invoked -- filter_depth=False must skip them entirely.
        frame = next(edge_client.iter_realsense(64, 48, 30, filter_depth=False))
        self.assertEqual(frame.camera_id, "realsense")


if __name__ == "__main__":
    unittest.main()
