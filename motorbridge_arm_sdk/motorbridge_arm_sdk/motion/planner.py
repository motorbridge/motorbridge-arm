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
    # Geodesic typically pairs with smooth timing.
    if p == "geodesic":
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        return 10.0 * t3 - 15.0 * t4 + 6.0 * t5
    return t


def _rpy_to_rot(roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def interpolate_pose_geodesic(start: Pose6D, end: Pose6D, steps: int, profile: str = "geodesic") -> list[Pose6D]:
    if steps < 2:
        steps = 2

    try:
        import pinocchio as pin
        import numpy as np
    except Exception:
        # Safe fallback: linear pose interpolation if pinocchio is unavailable.
        return interpolate_pose_linear(start, end, steps, profile=profile)

    R0 = np.array(_rpy_to_rot(start.roll, start.pitch, start.yaw), dtype=float)
    R1 = np.array(_rpy_to_rot(end.roll, end.pitch, end.yaw), dtype=float)
    T0 = pin.SE3(R0, np.array([start.x, start.y, start.z], dtype=float))
    T1 = pin.SE3(R1, np.array([end.x, end.y, end.z], dtype=float))
    xi = pin.log6(T0.inverse() * T1).vector

    points: list[Pose6D] = []
    for k in range(steps):
        a = _apply_profile(k / (steps - 1), profile)
        Ta = T0 * pin.exp6(xi * a)
        t = Ta.translation
        R = Ta.rotation
        # Keep RPY convention consistent with rest of package.
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0.0
        points.append(
            Pose6D(
                x=float(t[0]),
                y=float(t[1]),
                z=float(t[2]),
                roll=float(roll),
                pitch=float(pitch),
                yaw=float(yaw),
            )
        )
    return points


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
