from __future__ import annotations

import enum
from dataclasses import dataclass, field

from ..motion.planner import _apply_profile, interpolate_pose_geodesic
from ..types import Pose6D


class TrajProfile(enum.Enum):
    LINEAR = "linear"
    MIN_JERK = "min_jerk"
    GEODESIC = "geodesic"


@dataclass(slots=True)
class TrajPlanParams:
    dt: float = 0.02
    profile: TrajProfile = TrajProfile.MIN_JERK


@dataclass(slots=True)
class CartesianPoint:
    time: float
    pose: Pose6D


@dataclass(slots=True)
class CartesianTrajectory:
    points_: list[CartesianPoint] = field(default_factory=list)

    def add_point(self, t: float, pose: Pose6D) -> None:
        self.points_.append(CartesianPoint(t, pose))

    def duration(self) -> float:
        return self.points_[-1].time if self.points_ else 0.0

    def points(self) -> list[CartesianPoint]:
        return self.points_


@dataclass(slots=True)
class CartesianTrajectoryResult:
    trajectory: CartesianTrajectory
    n_points: int


def plan_cartesian_geodesic_trajectory(
    start_pose: Pose6D,
    end_pose: Pose6D,
    duration: float,
    params: TrajPlanParams | None = None,
) -> CartesianTrajectoryResult:
    if duration <= 0.0:
        raise ValueError("duration must be > 0")
    if params is None:
        params = TrajPlanParams()

    n = max(2, int(duration / max(params.dt, 1e-4)) + 1)
    profile = params.profile.value

    poses = interpolate_pose_geodesic(start_pose, end_pose, n, profile=profile)
    traj = CartesianTrajectory()
    dt = duration / (n - 1)
    for i, pose in enumerate(poses):
        t = i * dt
        # Preserve consistent timing profile mapping even if interpolation backend changes.
        _ = _apply_profile(i / (n - 1), profile)
        traj.add_point(t, pose)
    return CartesianTrajectoryResult(trajectory=traj, n_points=n)
