"""Pytest-session-only setup.

`test_material_transformers_compat.py` exercises `transformers`' CLIP
loading machinery against tiny stand-in configs, not real downloaded
weights -- but without `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` set,
`transformers` still tries an actual network call to huggingface.co first
(to check for updates) before falling back to any local/offline path. On a
network that blocks or firewalls that host, this doesn't fail fast -- it
hangs for the library's own retry/backoff window, which can stall (or, in a
sandboxed CI runner, time out) an otherwise-fast, fully offline test suite.
Every test in this suite is written to run without internet access (see
this project's own test-writing convention: synthetic data and stand-in
models throughout), so forcing offline mode here removes a flaky, slow
failure mode without changing what any test actually verifies. Scoped to
the pytest process only (`os.environ`, not the machine/user's shell) --
running the real app outside pytest is unaffected.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
