from __future__ import annotations

import math


def estimate_steps(q0: list[float], q1: list[float], step_rad: float = 0.02) -> int:
    if len(q0) != len(q1):
        raise ValueError("q0/q1 dimension mismatch")
    max_delta = max((abs(a - b) for a, b in zip(q0, q1)), default=0.0)
    return max(2, int(math.ceil(max_delta / max(step_rad, 1e-6))) + 1)
