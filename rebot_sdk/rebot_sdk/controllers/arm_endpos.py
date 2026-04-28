from __future__ import annotations

from dataclasses import dataclass

from ..arm import Arm
from ..motion.geodesic import compute_geodesic_stats, plan_se3_geodesic
from ..trajectory.clik_tracker import IKParams
from ..trajectory.trajectory_planner import plan_joint_space_trajectory
from ..types import Pose6D


@dataclass(slots=True)
class TrajResult:
    ok: bool
    points: int
    max_pos_err: float
    avg_pos_err: float


class ArmEndPos:
    """Independent high-level end-effector controller."""

    def __init__(self, arm: Arm) -> None:
        self.arm = arm

    def move_to_ik(self, target: Pose6D, vlim: float = 1.0) -> bool:
        q = self.arm.solve_ik(target)
        self.arm.move_j(q, vlim=vlim, profile="min_jerk")
        return True

    def move_to_traj(self, target: Pose6D, duration_s: float = 2.0, vlim: float = 1.0) -> TrajResult:
        start = self.arm.get_pose()
        ref = plan_se3_geodesic(start, target, duration_s=duration_s, dt_s=self.arm._cfg.loop_dt_s)
        q0 = self.arm.get_joint_positions()
        q1 = self.arm.solve_ik(target)
        jt = plan_joint_space_trajectory(
            model=self.arm._kin._model,
            end_frame_id=self.arm._kin._frame_id if self.arm._kin._frame_id is not None else 0,
            q_start=q0,
            q_end=q1,
            duration=duration_s,
            kin=self.arm._kin,
            ik_params=IKParams(),
            null_gain=0.1,
        )
        self.arm._run_joint_points([p.q for p in jt], vlim=vlim, motion_name="move_l_clik")
        actual = [self.arm._kin.forward(p.q) for p in jt]
        s = compute_geodesic_stats(ref, actual, success_flags=[p.ik_success for p in jt])
        return TrajResult(ok=True, points=s.total_points, max_pos_err=s.max_position_error, avg_pos_err=s.avg_position_error)
