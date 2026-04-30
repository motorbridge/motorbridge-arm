from __future__ import annotations

import math
from dataclasses import dataclass

from ..types import Pose6D


@dataclass(slots=True)
class TrajectoryStats:
    total_points: int
    success_count: int
    success_rate: float
    max_position_error: float
    avg_position_error: float
    max_orientation_error: float
    avg_orientation_error: float


def pose_position_error(a: Pose6D, b: Pose6D) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def pose_orientation_error(a: Pose6D, b: Pose6D) -> float:
    def _wrap(x):
        while x > math.pi:
            x -= 2 * math.pi
        while x < -math.pi:
            x += 2 * math.pi
        return x
    dr = _wrap(a.roll - b.roll)
    dp = _wrap(a.pitch - b.pitch)
    dy = _wrap(a.yaw - b.yaw)
    return (dr * dr + dp * dp + dy * dy) ** 0.5


def compute_trajectory_stats(reference: list[Pose6D], actual: list[Pose6D], success_flags: list[bool] | None = None) -> TrajectoryStats:
    n = min(len(reference), len(actual))
    if n == 0:
        return TrajectoryStats(
            total_points=0,
            success_count=0,
            success_rate=0.0,
            max_position_error=0.0,
            avg_position_error=0.0,
            max_orientation_error=0.0,
            avg_orientation_error=0.0,
        )
    pos_errs = [pose_position_error(reference[i], actual[i]) for i in range(n)]
    ori_errs = [pose_orientation_error(reference[i], actual[i]) for i in range(n)]
    if success_flags is None:
        success_flags = [True] * n
    m = min(n, len(success_flags))
    success_count = sum(1 for i in range(m) if success_flags[i])
    return TrajectoryStats(
        total_points=n,
        success_count=success_count,
        success_rate=(success_count / n),
        max_position_error=max(pos_errs),
        avg_position_error=sum(pos_errs) / n,
        max_orientation_error=max(ori_errs),
        avg_orientation_error=sum(ori_errs) / n,
    )
