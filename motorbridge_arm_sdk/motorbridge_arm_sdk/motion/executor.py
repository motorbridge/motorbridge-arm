from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from ..errors import ArmError, ArmErrorCode

logger = logging.getLogger(__name__)


class JointMotionExecutor:
    def __init__(self, dt_s: float = 0.02) -> None:
        if dt_s <= 0:
            raise ValueError("dt_s must be > 0")
        self._dt_s = dt_s

    def interpolate_linear(self, q0: list[float], q1: list[float], steps: int) -> list[list[float]]:
        if len(q0) != len(q1):
            raise ArmError(ArmErrorCode.ERR_CONFIG, "q0/q1 dimension mismatch")
        if steps < 2:
            steps = 2
        points: list[list[float]] = []
        for k in range(steps):
            a = k / (steps - 1)
            points.append([(1.0 - a) * s + a * t for s, t in zip(q0, q1)])
        return points

    def run(self, points: list[list[float]], send_fn: Callable[[list[float], float], None], vlim: float, abort_event: threading.Event | None = None, timeout_s: float | None = None) -> None:
        """Execute a sequence of joint-space waypoints.

        Args:
            points: Joint-space waypoints to send.
            send_fn: Function ``(q, vlim) -> None`` that sends a command.
            vlim: Velocity limit.
            abort_event: Event to signal early termination.
            timeout_s: Maximum wall-clock time for the entire motion.
                If ``None`` a per-point timeout of ``dt_s * 2 * len(points)``
                is used to prevent indefinite blocking.
        """
        logger.info("executor.run() start — %d points, dt=%.4fs", len(points), self._dt_s)
        t_start = time.monotonic()
        if timeout_s is None:
            timeout_s = self._dt_s * 2.0 * len(points)
        deadline = t_start + timeout_s
        for i, q in enumerate(points):
            if abort_event is not None and abort_event.is_set():
                logger.warning("Motion aborted at point %d/%d", i, len(points))
                return
            now = time.monotonic()
            if now >= deadline:
                logger.warning(
                    "Motion timed out after %.2fs at point %d/%d (timeout=%.2fs)",
                    now - t_start, i, len(points), timeout_s,
                )
                if abort_event is not None:
                    abort_event.set()
                return
            t_pre = time.monotonic()
            send_fn(q, vlim)
            elapsed = time.monotonic() - t_pre
            if elapsed > self._dt_s:
                logger.warning("timing overrun: step took %.4fs (> dt=%.4fs)", elapsed, self._dt_s)
            remaining = max(0.0, self._dt_s - elapsed)
            if remaining > 0:
                time.sleep(remaining)
        logger.info("executor.run() done — %.3fs elapsed", time.monotonic() - t_start)
