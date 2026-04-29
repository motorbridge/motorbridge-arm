"""Cartesian trajectory sampling using SE(3) geodesic interpolation.
/ 基于 SE(3) 测地线插值的笛卡尔轨迹采样。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from ..motion.planner import _apply_profile, interpolate_pose_geodesic
from ..types import Pose6D


class TrajProfile(enum.Enum):
    """Motion profile for time-parameterizing the trajectory. / 轨迹时间参数化的运动曲线。

    Attributes:
        LINEAR: Linear interpolation in parameter space.
            / 参数空间线性插值。
        MIN_JERK: Minimum-jerk (5th-order polynomial) profile for smooth motion.
            / 最小加加速度（五阶多项式）曲线，实现平滑运动。
        GEODESIC: Geodesic profile on SE(3).
            / SE(3) 上的测地线曲线。
    """

    LINEAR = "linear"
    MIN_JERK = "min_jerk"
    GEODESIC = "geodesic"


@dataclass(slots=True)
class TrajPlanParams:
    """Parameters governing trajectory sampling. / 轨迹采样参数。

    Attributes:
        dt: Time step between consecutive waypoints in seconds.
            Default is 0.02 s (50 Hz).
            / 连续航点之间的时间步长（秒）。默认 0.02 s（50 Hz）。
        profile: Motion profile used to parameterize the interpolation.
            Default is ``MIN_JERK``.
            / 用于参数化插值的运动曲线。默认 ``MIN_JERK``。
    """

    dt: float = 0.02
    profile: TrajProfile = TrajProfile.MIN_JERK


@dataclass(slots=True)
class CartesianPoint:
    """A single waypoint on a Cartesian trajectory. / 笛卡尔轨迹上的一个航点。

    Attributes:
        time: Time stamp in seconds. / 时间戳（秒）。
        pose: Cartesian pose (position + orientation) at this time.
            / 该时刻的笛卡尔位姿（位置 + 姿态）。
    """

    time: float
    pose: Pose6D


@dataclass(slots=True)
class CartesianTrajectory:
    """Ordered sequence of Cartesian waypoints. / 有序的笛卡尔航点序列。

    Attributes:
        points_: Internal list of ``CartesianPoint`` objects.
            / ``CartesianPoint`` 对象的内部列表。
    """

    points_: list[CartesianPoint] = field(default_factory=list)

    def add_point(self, t: float, pose: Pose6D) -> None:
        """Append a waypoint at time ``t``. / 在时间 ``t`` 追加一个航点。"""
        self.points_.append(CartesianPoint(t, pose))

    def duration(self) -> float:
        """Return the total duration of the trajectory. / 返回轨迹总时长。"""
        return self.points_[-1].time if self.points_ else 0.0

    def points(self) -> list[CartesianPoint]:
        """Return all waypoints. / 返回所有航点。"""
        return self.points_


@dataclass(slots=True)
class CartesianTrajectoryResult:
    """Result of Cartesian trajectory sampling. / 笛卡尔轨迹采样结果。

    Attributes:
        trajectory: The sampled ``CartesianTrajectory``.
            / 采样得到的 ``CartesianTrajectory``。
        n_points: Number of waypoints in the trajectory.
            / 轨迹中的航点数。
    """

    trajectory: CartesianTrajectory
    n_points: int


def plan_cartesian_geodesic_trajectory(
    start_pose: Pose6D,
    end_pose: Pose6D,
    duration: float,
    params: TrajPlanParams | None = None,
) -> CartesianTrajectoryResult:
    """Plan a Cartesian trajectory via SE(3) geodesic interpolation.
    / 通过 SE(3) 测地线插值规划笛卡尔轨迹。

    Generates a sequence of Cartesian poses between ``start_pose`` and
    ``end_pose`` using the specified time step and motion profile.
    Position is linearly interpolated, while orientation follows the
    shortest arc on SO(3) (geodesic).

    Args:
        start_pose: Starting Cartesian pose.
            / 起始笛卡尔位姿。
        end_pose: Ending Cartesian pose.
            / 终止笛卡尔位姿。
        duration: Total trajectory duration in seconds.  Must be > 0.
            / 轨迹总时长（秒）。必须大于 0。
        params: Sampling parameters (time step and motion profile).
            Uses defaults when ``None``.
            / 采样参数（时间步长和运动曲线）。为 ``None`` 时使用默认值。

    Returns:
        ``CartesianTrajectoryResult`` containing the sampled trajectory
        and the number of points.
        / ``CartesianTrajectoryResult``，包含采样轨迹和航点数。

    Raises:
        ValueError: If ``duration`` is not positive.
            / 若 ``duration`` 非正则抛出。
    """
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
