from __future__ import annotations

import logging
from dataclasses import dataclass

from ..model.kinematics import Kinematics

logger = logging.getLogger(__name__)

from ..motion.planner import ArcSpec, interpolate_pose_circular
from ..trajectory.sampler import TrajProfile
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


@dataclass(slots=True)
class SimPoseValidation:
    requested: Pose6D
    solved_pose: Pose6D
    q: list[float]
    error_m: float
    reachable: bool
    iterations: int


class SimArm:
    """Model-only arm simulator without motor hardware.

    仅使用模型的机械臂模拟器，无需电机硬件。
    """

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

    @property
    def loop_dt_s(self) -> float:
        """Control loop period in seconds.

        控制循环周期（秒）。
        """
        return self._cfg.loop_dt_s

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
        return self._kin.inverse_position(pose, self._q)

    def validate_pose(self, pose: Pose6D, tolerance_m: float = 0.003) -> SimPoseValidation:
        """Check whether a Cartesian pose is reachable without moving the simulator."""
        result = self._kin.inverse_position_result(pose, self._q)
        q = result.q if result.q else self.solve_ik(pose)
        solved_pose = self._kin.forward(q)
        error_m = (
            (solved_pose.x - pose.x) ** 2
            + (solved_pose.y - pose.y) ** 2
            + (solved_pose.z - pose.z) ** 2
        ) ** 0.5
        return SimPoseValidation(
            requested=pose,
            solved_pose=solved_pose,
            q=list(q),
            error_m=float(error_m),
            reachable=bool(error_m <= tolerance_m),
            iterations=int(getattr(result, "iterations", 0) or 0),
        )

    def plan_l(
        self,
        target: Pose6D,
        duration_s: float = 2.0,
        profile: str = "min_jerk",
        null_gain: float = 0.1,
    ) -> SimTrajectory:
        q_start = self.get_joint_positions()
        q_end = self.solve_ik(target)
        steps = max(2, int(max(duration_s, self._cfg.loop_dt_s) / self._cfg.loop_dt_s) + 1)

        def ease(u: float) -> float:
            p = self._profile_from_name(profile)
            if p == TrajProfile.LINEAR:
                return u
            # Minimum-jerk profile: smooth start and stop, inexpensive enough
            # for the interactive WebSocket simulation loop.
            return (10 * u**3) - (15 * u**4) + (6 * u**5)

        points: list[SimTrajectoryPoint] = []
        for i in range(steps):
            u = i / max(1, steps - 1)
            s = ease(u)
            q = [float(a + (b - a) * s) for a, b in zip(q_start, q_end)]
            points.append(
                SimTrajectoryPoint(
                    time=float(u * duration_s),
                    q=q,
                    pose=self._kin.forward(q),
                    ik_success=True,
                )
            )
        result = SimTrajectory(points=points, duration_s=duration_s)
        logger.info("plan_l: %d points, duration=%.2fs", len(result.points), result.duration_s)
        return result

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
        num_joints = len(self._cfg.joints)
        for i, pose in enumerate(poses):
            q_new = self.solve_ik(pose)
            # Check IK convergence: if the solved position is far from the
            # requested pose, mark the point as an IK failure.
            solved_pose = self._kin.forward(q_new)
            pos_err = (
                (solved_pose.x - pose.x) ** 2
                + (solved_pose.y - pose.y) ** 2
                + (solved_pose.z - pose.z) ** 2
            ) ** 0.5
            ik_ok = pos_err < 1e-3
            if not ik_ok:
                logger.warning(
                    "plan_c step %d: IK did not converge (pos_err=%.4f m)", i, pos_err
                )
            out.append(
                SimTrajectoryPoint(
                    time=i * dt,
                    q=list(q_new),
                    pose=solved_pose,
                    ik_success=ik_ok,
                )
            )
        result = SimTrajectory(points=out, duration_s=duration_s)
        logger.info("plan_c: %d points, duration=%.2fs", len(result.points), result.duration_s)
        return result

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
        """Play back a trajectory by stepping through its points.

        逐步回放轨迹点。

        Args:
            trajectory: The trajectory to play back.
                        要回放的轨迹。
            realtime: If ``True``, sleep between points to match real time.
                      若为 ``True``，则在点之间休眠以匹配真实时间。
            visualizer: Optional visualizer object with ``update()`` and
                        ``draw_actual_path()`` methods.
                        可选的可视化对象，需提供 ``update()`` 和
                        ``draw_actual_path()`` 方法。

        Raises:
            ValueError: If any trajectory point has a joint vector whose length
                        does not match the configured number of joints.
        """
        import time

        num_joints = len(self._cfg.joints)
        prev_t = 0.0
        path: list[list[float]] = []
        for pt in trajectory.points:
            if len(pt.q) != num_joints:
                raise ValueError(
                    f"trajectory point at t={pt.time:.4f} has {len(pt.q)} "
                    f"joints, expected {num_joints}"
                )
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
