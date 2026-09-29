"""Keep a broken optional audio library from taking image depth down with it.

On the cloud VM, torchaudio is installed in the system site-packages but its
compiled extension (_torchaudio.abi3.so) does not load against the installed
torch, so `import torchaudio` raises OSError. Depth Anything V2 is an image
model and never uses torchaudio, but import paths that probe optional audio
support only catch ImportError, so the OSError escaped and switched Logitech
depth off.

`disable_broken_torchaudio()` imports torchaudio once. Only if the package is
present *and* fails to load does it mark it unimportable (a clean ImportError
from then on) and tell transformers it is unavailable. A working torchaudio,
or none at all, is left exactly as it was; torch and CUDA are never touched.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys

LOGGER = logging.getLogger(__name__)

_state: dict[str, str | None] = {}


def disable_broken_torchaudio() -> str | None:
    """Return the load error if torchaudio was broken and has been disabled, else None."""
    if "result" in _state:
        return _state["result"]
    result = None
    if sys.modules.get("torchaudio", False) is not None and importlib.util.find_spec("torchaudio") is not None:
        try:
            importlib.import_module("torchaudio")
        except (OSError, ImportError, RuntimeError) as exc:
            result = f"{type(exc).__name__}: {exc}"
            for name in [key for key in sys.modules if key == "torchaudio" or key.startswith("torchaudio.")]:
                sys.modules.pop(name, None)
            sys.modules["torchaudio"] = None  # type: ignore[assignment]  # import now raises ImportError
            try:
                import_utils = importlib.import_module("transformers.utils.import_utils")
                import_utils._torchaudio_available = False  # type: ignore[attr-defined]
            except (ImportError, AttributeError):
                pass
            LOGGER.warning("torchaudio is installed but cannot load (%s); disabled for this process -- "
                           "image depth does not use it", result)
    _state["result"] = result
    return result
