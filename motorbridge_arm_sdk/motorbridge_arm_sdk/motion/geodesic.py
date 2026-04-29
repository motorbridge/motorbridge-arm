from __future__ import annotations

from dataclasses import dataclass

from ..model.inverse_kinematics import IKParams
from ..types import Pose6D
from ..trajectory.clik_tracker import track_trajectory
from ..trajectory.sampler import TrajPlanParams, plan_cartesian_geodesic_trajectory
from ..trajectory.trajectory_planner import compute_traj_stats


# Backward-compatible alias.
CliKParams = IKParams


@dataclass(slots=True)
class JointTrajectoryPoint:
    time: float
    q: list[float]
    ik_success: bool


def plan_se3_geodesic(start: Pose6D, end: Pose6D, duration_s: float, dt_s: float = 0.02) -> list[Pose6D]:
    res = plan_cartesian_geodesic_trajectory(start, end, duration_s, params=TrajPlanParams(dt=dt_s))
    return [p.pose for p in res.trajectory.points()]


def track_with_clik(model, end_frame_id: int, poses: list[Pose6D], q0: list[float], kin, params: IKParams | None = None) -> list[JointTrajectoryPoint]:
    p = params or IKParams()
    class _Pt:
        def __init__(self, t: float, pose: Pose6D) -> None:
            self.time = t
            self.pose = pose

    class _Cart:
        def __init__(self, points: list[_Pt]) -> None:
            self._points = points

        def points(self) -> list[_Pt]:
            return self._points

    denom = max(len(poses) - 1, 1)
    cart = _Cart([_Pt(i / denom, pose) for i, pose in enumerate(poses)])
    out = track_trajectory(
        model=model,
        end_frame_id=end_frame_id,
        traj=cart,
        q_init=q0,
        kin=kin,
        ik_params=IKParams(max_iter=p.max_iter, tolerance=p.tolerance, damping=p.damping, step_size=p.step_size, null_gain=p.null_gain),
        null_gain=p.null_gain,
    )
    return [JointTrajectoryPoint(time=x.time, q=list(x.q), ik_success=bool(x.ik_success)) for x in out]


def compute_geodesic_stats(reference: list[Pose6D], actual: list[Pose6D], success_flags: list[bool] | None = None):
    return compute_traj_stats(reference, actual, success_flags=success_flags)
