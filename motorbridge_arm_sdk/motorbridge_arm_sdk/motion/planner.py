from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..types import Pose6D
from ..model.kinematics import _rot_to_rpy


def se3_interpolate(a: Any, b: Any, s: float) -> Any:
    """Interpolate between two SE(3) poses on the geodesic.

    Computes ``a * exp6(log6(a^{-1} * b) * s)`` using Pinocchio.  Accepts
    ``pin.SE3`` objects or (4, 4) ndarrays as input.

    Args:
        a: Start pose (pin.SE3 or 4x4 ndarray).
        b: End pose (pin.SE3 or 4x4 ndarray).
        s: Interpolation parameter in ``[0, 1]``.

    Returns:
        A ``pin.SE3`` object at parameter *s* along the geodesic from *a*
        to *b*.

    Raises:
        ImportError: If Pinocchio is not available.
    """
    import pinocchio as pin
    import numpy as np

    if not isinstance(a, pin.SE3):
        a = pin.SE3(np.array(a, dtype=float))
    if not isinstance(b, pin.SE3):
        b = pin.SE3(np.array(b, dtype=float))
    xi = pin.log6(a.inverse() * b).vector
    return a * pin.exp6(xi * max(0.0, min(1.0, s)))


def estimate_steps(q0: list[float], q1: list[float], step_rad: float = 0.02) -> int:
    if len(q0) != len(q1):
        raise ValueError("q0/q1 dimension mismatch")
    max_delta = max((abs(a - b) for a, b in zip(q0, q1)), default=0.0)
    return max(2, int(math.ceil(max_delta / max(step_rad, 1e-6))) + 1)


def _apply_profile(t: float, profile: str, accel_ratio: float = 0.25) -> float:
    p = profile.lower().strip()
    t = max(0.0, min(1.0, t))
    if p == "linear":
        return t
    if p in ("min_jerk", "geodesic"):
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        return 10.0 * t3 - 15.0 * t4 + 6.0 * t5
    if p == "trapezoid":
        a = max(0.01, min(0.49, accel_ratio))
        if t < a:
            s = 0.5 * t * t / a
        elif t < 1.0 - a:
            s = a / 2.0 + (t - a)
        else:
            dt = t - (1.0 - a)
            s = 1.0 - a / 2.0 + dt - 0.5 * dt * dt / a
        return max(0.0, min(1.0, s))
    if p == "cubic":
        # Cubic Hermite smoothstep: s = -2t^3 + 3t^2
        # Zero velocity at both endpoints, smooth interpolation.
        return -2.0 * t * t * t + 3.0 * t * t
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


def interpolate_pose_geodesic(start: Pose6D, end: Pose6D, steps: int, profile: str = "geodesic", accel_ratio: float = 0.25) -> list[Pose6D]:
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
        a = _apply_profile(k / (steps - 1), profile, accel_ratio)
        Ta = T0 * pin.exp6(xi * a)
        t = Ta.translation
        R = Ta.rotation
        roll, pitch, yaw = _rot_to_rpy(R)
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


def interpolate_joint_linear(q0: list[float], q1: list[float], steps: int, profile: str = "linear", accel_ratio: float = 0.25) -> list[list[float]]:
    if len(q0) != len(q1):
        raise ValueError("q0/q1 dimension mismatch")
    if steps < 2:
        steps = 2
    points: list[list[float]] = []
    for k in range(steps):
        a = _apply_profile(k / (steps - 1), profile, accel_ratio)
        points.append([(1.0 - a) * s + a * t for s, t in zip(q0, q1)])
    return points


def interpolate_pose_linear(start: Pose6D, end: Pose6D, steps: int, profile: str = "linear", accel_ratio: float = 0.25) -> list[Pose6D]:
    if steps < 2:
        steps = 2
    points: list[Pose6D] = []
    for k in range(steps):
        a = _apply_profile(k / (steps - 1), profile, accel_ratio)
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


def interpolate_pose_circular(start: Pose6D, end: Pose6D, arc: ArcSpec, steps: int, profile: str = "linear", accel_ratio: float = 0.25) -> list[Pose6D]:
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
        t = _apply_profile(k / (steps - 1), profile, accel_ratio)
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
