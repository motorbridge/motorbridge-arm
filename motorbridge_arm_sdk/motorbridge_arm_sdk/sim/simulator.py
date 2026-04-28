from __future__ import annotations

from dataclasses import dataclass

from ..model.kinematics import Kinematics
from ..motion.planner import ArcSpec, interpolate_pose_circular
from ..trajectory.clik_tracker import IKParams
from ..trajectory.sampler import TrajPlanParams, TrajProfile
from ..trajectory.trajectory_planner import plan_joint_space_trajectory
from ..types import ArmConfig, Pose6D


@dataclass(slots=True)
class SimTrajectoryPoint:
    time: float
    q: list[float]
    pose: Pose6D
    ik_success: bool


@dataclass(slots=True)
class SimTrajectory:
    points: list[SimTrajectoryPoint]
    duration_s: float


class SimArm:
    """Model-only arm simulator without motor hardware."""

    @staticmethod
    def _profile_from_name(profile: str) -> TrajProfile:
        p = str(profile or "min_jerk").strip().lower()
        if p == "linear":
            return TrajProfile.LINEAR
        if p == "geodesic":
            return TrajProfile.GEODESIC
        return TrajProfile.MIN_JERK

    def __init__(self, config: ArmConfig) -> None:
        self._cfg = config
        self._q = list(config.default_home or [0.0 for _ in config.joints])
        self._kin = Kinematics(config.urdf_path, config.ee_frame)

    def set_joint_positions(self, q: list[float]) -> None:
        if len(q) != len(self._cfg.joints):
            raise ValueError("joint length mismatch")
        self._q = list(q)

    def get_joint_positions(self) -> list[float]:
        return list(self._q)

    def get_pose(self) -> Pose6D:
        return self._kin.forward(self._q)

    def move_j(self, q_target: list[float]) -> None:
        self.set_joint_positions(q_target)

    def solve_ik(self, pose: Pose6D) -> list[float]:
        return self._kin.inverse(pose, self._q)

    def plan_l(
        self,
        target: Pose6D,
        duration_s: float = 2.0,
        profile: str = "min_jerk",
        null_gain: float = 0.1,
    ) -> SimTrajectory:
        q_start = self.get_joint_positions()
        q_end = self.solve_ik(target)
        traj = plan_joint_space_trajectory(
            model=self._kin._model,
            end_frame_id=self._kin._frame_id if self._kin._frame_id is not None else 0,
            q_start=q_start,
            q_end=q_end,
            duration=duration_s,
            kin=self._kin,
            params=TrajPlanParams(dt=self._cfg.loop_dt_s, profile=self._profile_from_name(profile)),
            ik_params=IKParams(),
            null_gain=null_gain,
            start_pose=self.get_pose(),
            end_pose=target,
        )
        points = [
            SimTrajectoryPoint(
                time=float(p.time),
                q=[float(v) for v in p.q],
                pose=self._kin.forward(p.q),
                ik_success=bool(p.ik_success),
            )
            for p in traj
        ]
        return SimTrajectory(points=points, duration_s=duration_s)

    def move_l(
        self,
        target: Pose6D,
        duration_s: float = 2.0,
        profile: str = "min_jerk",
        null_gain: float = 0.1,
    ) -> SimTrajectory:
        traj = self.plan_l(target=target, duration_s=duration_s, profile=profile, null_gain=null_gain)
        if traj.points:
            self._q = list(traj.points[-1].q)
        return traj

    def plan_c(
        self,
        target: Pose6D,
        center_x: float,
        center_y: float,
        normal_z: float = 1.0,
        duration_s: float = 2.0,
        steps: int = 80,
        profile: str = "min_jerk",
    ) -> SimTrajectory:
        start = self.get_pose()
        poses = interpolate_pose_circular(
            start,
            target,
            ArcSpec(center_x=center_x, center_y=center_y, normal_z=normal_z),
            steps=max(2, steps),
            profile=profile,
        )
        q = self.get_joint_positions()
        dt = duration_s / max(1, len(poses) - 1)
        out: list[SimTrajectoryPoint] = []
        for i, pose in enumerate(poses):
            q = self.solve_ik(pose)
            out.append(
                SimTrajectoryPoint(
                    time=i * dt,
                    q=list(q),
                    pose=self._kin.forward(q),
                    ik_success=True,
                )
            )
        return SimTrajectory(points=out, duration_s=duration_s)

    def move_c(
        self,
        target: Pose6D,
        center_x: float,
        center_y: float,
        normal_z: float = 1.0,
        duration_s: float = 2.0,
        steps: int = 80,
        profile: str = "min_jerk",
    ) -> SimTrajectory:
        traj = self.plan_c(
            target=target,
            center_x=center_x,
            center_y=center_y,
            normal_z=normal_z,
            duration_s=duration_s,
            steps=steps,
            profile=profile,
        )
        if traj.points:
            self._q = list(traj.points[-1].q)
        return traj

    def playback(
        self,
        trajectory: SimTrajectory,
        realtime: bool = False,
        visualizer=None,
    ) -> None:
        import time

        prev_t = 0.0
        path: list[list[float]] = []
        for pt in trajectory.points:
            self._q = list(pt.q)
            if visualizer is not None:
                visualizer.update(pt.q)
                path.append([pt.pose.x, pt.pose.y, pt.pose.z])
                visualizer.draw_actual_path(path)
            if realtime:
                dt = max(0.0, pt.time - prev_t)
                if dt > 0:
                    time.sleep(dt)
            prev_t = pt.time
