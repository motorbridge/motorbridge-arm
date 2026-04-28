from __future__ import annotations

import time

from ..errors import ArmError, ArmErrorCode


class JointMotionExecutor:
    def __init__(self, dt_s: float = 0.02) -> None:
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

    def run(self, points: list[list[float]], send_fn, vlim: float) -> None:
        for q in points:
            send_fn(q, vlim)
            time.sleep(self._dt_s)
