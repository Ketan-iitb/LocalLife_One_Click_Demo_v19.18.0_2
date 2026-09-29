"""v41 cloud fixes: supervised dashboard tunnel and Depth Anything despite a broken torchaudio."""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from locallife_cloud import optional_imports

ROOT = Path(__file__).resolve().parents[1]


def _powershell() -> str | None:
    return os.environ.get("LOCALLIFE_PWSH") or shutil.which("pwsh") or shutil.which("powershell")


class CloudTunnelSupervisorTests(unittest.TestCase):
    """Runs tests/ps/test_cloud_tunnel.ps1 against the real launcher functions."""

    @unittest.skipUnless(_powershell(), "PowerShell not available")
    def test_tunnel_supervisor_scenarios(self) -> None:
        result = subprocess.run(
            [_powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
             str(ROOT / "tests" / "ps" / "test_cloud_tunnel.ps1"),
             "-Launcher", str(ROOT.parent / "Start-LocalLife-Demo.ps1")],
            capture_output=True, text=True, timeout=300, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ALL PASSED", result.stdout)
        for scenario in ("delayed readiness", "zone change", "stale plink", "no READY without HTTP",
                         "staged terminal error", "local-mode server"):
            self.assertIn(scenario, result.stdout)


class BrokenTorchaudioGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.saved_modules = {k: v for k, v in sys.modules.items() if k == "torchaudio" or k.startswith("torchaudio.")}
        for key in self.saved_modules:
            sys.modules.pop(key)
        sys.path.insert(0, self.directory)
        optional_imports._state.clear()

    def tearDown(self) -> None:
        sys.path.remove(self.directory)
        for key in [k for k in sys.modules if k == "torchaudio" or k.startswith("torchaudio.")]:
            sys.modules.pop(key)
        sys.modules.update(self.saved_modules)
        optional_imports._state.clear()
        importlib.invalidate_caches()
        shutil.rmtree(self.directory, ignore_errors=True)

    def _package(self, body: str) -> None:
        package = Path(self.directory) / "torchaudio"
        package.mkdir()
        (package / "__init__.py").write_text(textwrap.dedent(body), encoding="utf-8")
        importlib.invalidate_caches()

    def test_broken_shared_library_becomes_a_clean_import_error(self) -> None:
        self._package('raise OSError("Could not load this library: .../_torchaudio.abi3.so")\n')
        with self.assertRaises(OSError):
            try:
                import torchaudio  # noqa: F401
            except ImportError:  # the guard real import paths use -- it misses OSError
                pass
        sys.modules.pop("torchaudio", None)
        error = optional_imports.disable_broken_torchaudio()
        self.assertIn("_torchaudio.abi3.so", error)
        with self.assertRaises(ImportError):
            import torchaudio  # noqa: F401,F811

    def test_working_torchaudio_is_left_alone(self) -> None:
        self._package("VALUE = 1\n")
        self.assertIsNone(optional_imports.disable_broken_torchaudio())
        self.assertEqual(importlib.import_module("torchaudio").VALUE, 1)

    def test_absent_torchaudio_is_not_touched(self) -> None:
        if importlib.util.find_spec("torchaudio") is not None:
            self.skipTest("a real torchaudio is installed here")
        self.assertIsNone(optional_imports.disable_broken_torchaudio())
        self.assertNotIn("torchaudio", sys.modules)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                         "torch/transformers not installed")
    def test_depth_model_starts_and_predicts_with_broken_torchaudio(self) -> None:
        import numpy as np
        from transformers import (DepthAnythingConfig, DepthAnythingForDepthEstimation, Dinov2Config,
                                  DPTImageProcessor)

        from locallife_cloud.config import AppConfig
        from locallife_cloud.inference import MetricDepthEstimator

        self._package('raise OSError("Could not load this library: .../_torchaudio.abi3.so")\n')
        model_dir = Path(self.directory) / "tiny_depth_anything"
        backbone = Dinov2Config(hidden_size=48, num_hidden_layers=4, num_attention_heads=2, intermediate_size=96,
                                image_size=518, patch_size=14, out_indices=[1, 2, 3, 4],
                                reshape_hidden_states=False, apply_layernorm=True)
        config = DepthAnythingConfig(backbone_config=backbone, neck_hidden_sizes=[12, 24, 48, 48],
                                     fusion_hidden_size=16, head_hidden_size=8, reassemble_hidden_size=48,
                                     depth_estimation_type="metric", max_depth=20)
        DepthAnythingForDepthEstimation(config).save_pretrained(model_dir)
        DPTImageProcessor(size={"height": 518, "width": 518}, keep_aspect_ratio=True,
                          ensure_multiple_of=14).save_pretrained(model_dir)
        from locallife_cloud.comparison import DualCameraCoordinator

        config = AppConfig(depth_model=str(model_dir), device="cpu", detector_model="local-opencv-background",
                           enable_monocular_depth=True, enable_bucket_sync=False, results_dir=Path(self.directory))
        coordinator = DualCameraCoordinator(config)   # the real construction path used by the server
        self.assertIn("abi3.so", optional_imports.disable_broken_torchaudio())
        estimator = coordinator.camera("logitech").depth_estimator
        self.assertIsInstance(estimator, MetricDepthEstimator)
        estimator.load()
        depth = estimator.estimate_batch([np.zeros((120, 160, 3), dtype=np.uint8)])[0]
        self.assertEqual(depth.shape, (120, 160))
        self.assertTrue(np.isfinite(depth).all())


if __name__ == "__main__":
    unittest.main()
