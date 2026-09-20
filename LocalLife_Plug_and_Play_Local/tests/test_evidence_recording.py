"""Opt-in benchmark evidence recording: off by default, bounded, non-blocking."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from locallife_cloud.evidence import EvidenceRecorder, RecordingSettings


class _Sink:
    """Counts writes instead of encoding images, so no OpenCV is needed."""

    def __init__(self, size: int = 1000) -> None:
        self.size = size
        self.paths: list[Path] = []

    def __call__(self, path: Path, frame) -> int:
        path.write_bytes(b"\x00" * self.size)
        self.paths.append(path)
        return self.size


class EvidenceRecordingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.sink = _Sink()

    def _recorder(self, **settings) -> EvidenceRecorder:
        options = {"enabled": True, "sample_fps": 1000.0}
        options.update(settings)
        recorder = EvidenceRecorder(
            self.root, RecordingSettings(**options), writer=self.sink,
        )
        self.addCleanup(recorder.stop)
        return recorder

    def _settle(self, recorder: EvidenceRecorder, expected: int) -> None:
        deadline = time.time() + 5.0
        while recorder.status.frames_written < expected and time.time() < deadline:
            time.sleep(0.01)

    def test_recording_is_off_unless_explicitly_enabled(self) -> None:
        recorder = EvidenceRecorder(self.root, RecordingSettings(), writer=self.sink)
        status = recorder.start("s1")
        self.assertFalse(status.active)
        self.assertIn("switched off", status.stopped_reason)
        self.assertFalse(recorder.offer(object(), frame_id=1, processing_mode="local"))
        self.assertEqual(self.sink.paths, [])

    def test_an_active_recording_is_visible(self) -> None:
        recorder = self._recorder()
        status = recorder.start("s1").to_dict()
        self.assertTrue(status["active"])
        self.assertIn("s1", status["session_directory"])

    def test_frames_are_written_with_metadata_linking_them_to_events(self) -> None:
        recorder = self._recorder()
        recorder.start("s1")
        recorder.offer(object(), frame_id=7, processing_mode="cloud", event_id="abc123")
        self._settle(recorder, 1)
        records = recorder.records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["event_id"], "abc123")
        self.assertEqual(records[0]["processing_mode"], "cloud")
        self.assertIn("00000007", records[0]["file"])

    def test_ordinary_frames_are_sampled_not_all_kept(self) -> None:
        # 2 FPS: a burst of frames inside half a second must yield one file.
        recorder = self._recorder(sample_fps=2.0)
        recorder.start("s1")
        accepted = [
            recorder.offer(object(), frame_id=index, processing_mode="local")
            for index in range(30)
        ]
        self.assertEqual(accepted.count(True), 1)

    def test_an_event_frame_is_never_sampled_away(self) -> None:
        recorder = self._recorder(sample_fps=0.01)
        recorder.start("s1")
        recorder.offer(object(), frame_id=1, processing_mode="local")
        # An ordinary frame now would be sampled out; an event frame must not be.
        self.assertFalse(recorder.offer(object(), frame_id=2, processing_mode="local"))
        self.assertTrue(
            recorder.offer(object(), frame_id=3, processing_mode="local", event_id="e1")
        )

    def test_the_size_limit_stops_recording(self) -> None:
        recorder = self._recorder(max_megabytes=0.002)  # 2 kB: two 1 kB frames.
        recorder.start("s1")
        for index in range(6):
            recorder.offer(object(), frame_id=index, processing_mode="local", event_id=f"e{index}")
            self._settle(recorder, min(index + 1, 2))
        self.assertFalse(recorder.status.active)
        self.assertIn("maximum size", recorder.status.stopped_reason)

    def test_the_duration_limit_stops_recording(self) -> None:
        # start() takes the first tick as started_at; the offer below then sees
        # a clock far past the budget. event_id keeps it out of the sampler, so
        # this exercises the duration limit and nothing else.
        ticks = iter([0.0] + [999.0] * 10)
        recorder = EvidenceRecorder(
            self.root, RecordingSettings(enabled=True, max_duration_seconds=10.0),
            writer=self.sink, clock=lambda: next(ticks),
        )
        self.addCleanup(recorder.stop)
        recorder.start("s1")
        self.assertFalse(
            recorder.offer(object(), frame_id=1, processing_mode="local", event_id="e1")
        )
        self.assertFalse(recorder.status.active)
        self.assertIn("maximum duration", recorder.status.stopped_reason)

    def test_a_full_queue_drops_rather_than_blocking_the_pipeline(self) -> None:
        # A writer that never returns: the inference loop must not wait on it.
        stalled = threading.Event()
        self.addCleanup(stalled.set)

        def blocking_writer(path: Path, frame) -> int:
            stalled.wait(timeout=10.0)
            return 0

        recorder = EvidenceRecorder(
            self.root, RecordingSettings(enabled=True, sample_fps=10_000.0),
            writer=blocking_writer, queue_size=2,
        )
        self.addCleanup(recorder.stop, "test teardown")
        recorder.start("s1")
        started = time.time()
        results = [
            recorder.offer(object(), frame_id=index, processing_mode="local", event_id=f"e{index}")
            for index in range(50)
        ]
        # Offering 50 frames into a stalled writer must return promptly...
        self.assertLess(time.time() - started, 2.0)
        # ...and the overflow must be reported as dropped, not lost silently.
        self.assertIn(False, results)
        self.assertGreater(recorder.status.frames_dropped, 0)

    def test_disk_usage_is_reported(self) -> None:
        recorder = self._recorder()
        recorder.start("s1")
        recorder.offer(object(), frame_id=1, processing_mode="local", event_id="e1")
        self._settle(recorder, 1)
        usage = recorder.disk_usage()
        self.assertIn("s1", usage["path"])
        self.assertGreaterEqual(usage["files"], 1)

    def test_stopping_flushes_and_closes_the_metadata(self) -> None:
        recorder = self._recorder()
        recorder.start("s1")
        recorder.offer(object(), frame_id=1, processing_mode="local", event_id="e1")
        self._settle(recorder, 1)
        status = recorder.stop("demonstration finished")
        self.assertFalse(status.active)
        self.assertEqual(status.stopped_reason, "demonstration finished")
        self.assertEqual(len(recorder.records()), 1)

    def test_uploads_are_off_unless_explicitly_enabled(self) -> None:
        self.assertFalse(RecordingSettings().upload_to_cloud)

    def test_invalid_limits_are_refused(self) -> None:
        for bad in ({"sample_fps": 0}, {"max_megabytes": 0}, {"max_duration_seconds": -1}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    RecordingSettings(enabled=True, **bad).validate()


import threading  # noqa: E402 - used by the stalled-writer test above

if __name__ == "__main__":
    unittest.main()
