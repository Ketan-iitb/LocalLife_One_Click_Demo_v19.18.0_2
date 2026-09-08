"""Small dependency-free recovery loop for physical camera streams."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from typing import TypeVar


LOGGER = logging.getLogger(__name__)
FrameT = TypeVar("FrameT")


def run_resilient_camera(
    camera_id: str,
    frame_factory: Callable[[], Iterator[FrameT]],
    stream_consumer: Callable[[Iterator[FrameT]], None],
    *,
    retry_seconds: float = 3.0,
    max_attempts: int | None = None,
) -> None:
    """Keep reopening one physical camera after USB or capture failures.

    USB cameras can remain busy briefly after a demonstration closes, and a
    RealSense device can disappear while Linux recreates its video nodes.  The
    edge process therefore retries each camera independently instead of ending
    both streams after one transient device error.
    """

    attempts = 0
    while max_attempts is None or attempts < max_attempts:
        attempts += 1
        try:
            LOGGER.info("Opening %s camera (attempt %d)", camera_id, attempts)
            stream_consumer(frame_factory())
            LOGGER.warning("%s camera stream ended; reopening it", camera_id)
        except KeyboardInterrupt:
            raise
        except Exception:
            LOGGER.exception(
                "%s camera could not stream; retrying in %.1f seconds",
                camera_id,
                retry_seconds,
            )

        if max_attempts is not None and attempts >= max_attempts:
            return
        time.sleep(max(0.0, retry_seconds))
