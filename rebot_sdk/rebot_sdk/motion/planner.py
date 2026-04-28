from __future__ import annotations

import math
from dataclasses import dataclass

from ..types import Pose6D


def estimate_steps(q0: list[float], q1: list[float], step_rad: float = 0.02) -> int:
    if len(q0) != len(q1):
        raise ValueError("q0/q1 dimension mismatch")
    max_delta = max((abs(a - b) for a, b in zip(q0, q1)), default=0.0)
    return max(2, int(math.ceil(max_delta / max(step_rad, 1e-6))) + 1)


def _apply_profile(t: float, profile: str) -> float:
    p = profile.lower().strip()
    t = max(0.0, min(1.0, t))
    if p == "linear":
        return t
    if p == "min_jerk":
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        return 10.0 * t3 - 15.0 * t4 + 6.0 * t5
    # Placeholder for full SE(3) geodesic timing profile.
    # Path geometry remains interpolated, time scaling follows min-jerk.
    if p == "geodesic":
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        return 10.0 * t3 - 15.0 * t4 + 6.0 * t5
    return t


def interpolate_joint_linear(q0: list[float], q1: list[float], steps: int, profile: str = "linear") -> list[list[float]]:
    if len(q0) != len(q1):
        raise ValueError("q0/q1 dimension mismatch")
    if steps < 2:
        steps = 2
    points: list[list[float]] = []
    for k in range(steps):
        a = _apply_profile(k / (steps - 1), profile)
        points.append([(1.0 - a) * s + a * t for s, t in zip(q0, q1)])
    return points


def interpolate_pose_linear(start: Pose6D, end: Pose6D, steps: int, profile: str = "linear") -> list[Pose6D]:
    if steps < 2:
        steps = 2
    points: list[Pose6D] = []
    for k in range(steps):
        a = _apply_profile(k / (steps - 1), profile)
        points.append(
            Pose6D(
                x=(1.0 - a) * start.x + a * end.x,
                y=(1.0 - a) * start.y + a * end.y,
                z=(1.0 - a) * start.z + a * end.z,
                roll=(1.0 - a) * start.roll + a * end.roll,
                pitch=(1.0 - a) * start.pitch + a * end.pitch,
                yaw=(1.0 - a) * start.yaw + a * end.yaw,
            )
        )
    return points


@dataclass(frozen=True, slots=True)
class ArcSpec:
    center_x: float
    center_y: float
    normal_z: float = 1.0


def interpolate_pose_circular(start: Pose6D, end: Pose6D, arc: ArcSpec, steps: int, profile: str = "linear") -> list[Pose6D]:
    if steps < 2:
        steps = 2
    sx = start.x - arc.center_x
    sy = start.y - arc.center_y
    ex = end.x - arc.center_x
    ey = end.y - arc.center_y
    rs = math.hypot(sx, sy)
    re = math.hypot(ex, ey)
    if rs < 1e-8 or re < 1e-8:
        return interpolate_pose_linear(start, end, steps)

    r = 0.5 * (rs + re)
    a0 = math.atan2(sy, sx)
    a1 = math.atan2(ey, ex)
    d = a1 - a0
    while d <= -math.pi:
        d += 2.0 * math.pi
    while d > math.pi:
        d -= 2.0 * math.pi
    if arc.normal_z < 0 and d > 0:
        d -= 2.0 * math.pi
    if arc.normal_z >= 0 and d < 0:
        d += 2.0 * math.pi

    points: list[Pose6D] = []
    for k in range(steps):
        t = _apply_profile(k / (steps - 1), profile)
        a = a0 + d * t
        x = arc.center_x + r * math.cos(a)
        y = arc.center_y + r * math.sin(a)
        points.append(
            Pose6D(
                x=x,
                y=y,
                z=(1.0 - t) * start.z + t * end.z,
                roll=(1.0 - t) * start.roll + t * end.roll,
                pitch=(1.0 - t) * start.pitch + t * end.pitch,
                yaw=(1.0 - t) * start.yaw + t * end.yaw,
            )
        )
    return points
