from __future__ import annotations

from dataclasses import dataclass

from ..motion.stats import TrajectoryStats, compute_trajectory_stats
from ..types import Pose6D
from .clik_tracker import IKParams, JointTrajectoryPoint, track_trajectory
from .sampler import TrajPlanParams, plan_cartesian_geodesic_trajectory


@dataclass(slots=True)
class TrajStats:
    total_points: int = 0
    success_count: int = 0
    success_rate: float = 0.0
    max_ik_error: float = 0.0
    avg_ik_error: float = 0.0


def plan_joint_space_trajectory(
    model,
    end_frame_id: int,
    q_start: list[float],
    q_end: list[float],
    duration: float,
    kin,
    params: TrajPlanParams | None = None,
    ik_params: IKParams | None = None,
    null_gain: float = 0.1,
    start_pose: Pose6D | None = None,
    end_pose: Pose6D | None = None,
) -> list[JointTrajectoryPoint]:
    if duration <= 0.0:
        raise ValueError("duration must be > 0")
    if params is None:
        params = TrajPlanParams()
    if ik_params is None:
        ik_params = IKParams()

    T_start = start_pose if start_pose is not None else kin.forward(q_start)
    T_end = end_pose if end_pose is not None else kin.forward(q_end)
    cart_result = plan_cartesian_geodesic_trajectory(T_start, T_end, duration, params)
    return track_trajectory(
        model=model,
        end_frame_id=end_frame_id,
        traj=cart_result.trajectory,
        q_init=q_start,
        kin=kin,
        ik_params=ik_params,
        null_gain=null_gain,
    )


def compute_traj_stats(reference: list[Pose6D], actual: list[Pose6D], success_flags: list[bool] | None = None) -> TrajectoryStats:
    return compute_trajectory_stats(reference, actual, success_flags=success_flags)
