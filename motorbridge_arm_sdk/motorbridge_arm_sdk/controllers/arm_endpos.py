from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from ..arm import Arm
from ..motion.geodesic import compute_geodesic_stats, plan_se3_geodesic
from ..trajectory.clik_tracker import IKParams
from ..trajectory.trajectory_planner import plan_joint_space_trajectory
from ..types import Pose6D

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrajResult:
    ok: bool
    points: int
    max_pos_err: float
    avg_pos_err: float


class ArmEndPos:
    """Independent high-level end-effector controller.

    Provides IK-based and trajectory-based motion commands using the
    parent :class:`Arm` instance's kinematics and motion execution pipeline.

    Supports use as a context manager::

        with ArmEndPos(arm) as ctrl:
            ctrl.move_to_ik(target_pose)

    Args:
        arm: A connected :class:`Arm` instance.
    """

    def __init__(self, arm: Arm) -> None:
        self.arm = arm

    @property
    def _kin(self):
        return self.arm.kinematics

    @property
    def _cfg(self):
        return self.arm.config

    def start(self) -> None:
        """Connect the arm, enable motors, and switch to POS_VEL mode.

        Equivalent to calling ``arm.connect()``, ``arm.enable()``, and
        ``arm.mode_pos_vel()`` in sequence.
        """
        self.arm.connect()
        self.arm.enable()
        self.arm.mode_pos_vel()

    def end(self) -> None:
        """Drive to home position and disconnect.

        Calls :meth:`safe_home` followed by ``arm.close()``.
        """
        try:
            self.safe_home()
        except Exception as exc:
            logger.warning("safe_home() during end() failed: %s", exc)
        self.arm.close()

    def stop(self) -> None:
        """Alias for :meth:`end` to satisfy the LifecycleController protocol.

        :meth:`end` 的别名，满足 LifecycleController 协议。
        """
        self.end()

    def safe_home(self, vlim: float = 0.3, timeout_s: float = 30.0) -> None:
        """Move the arm to its home (zero) position at reduced velocity.

        Uses a timeout to prevent indefinite blocking if motion stalls.
        When a timeout occurs, the motion is cancelled via the arm's abort
        event and the arm is switched to damping mode for safety.

        Args:
            vlim: Velocity limit as a fraction of maximum.  Defaults to
                ``0.3`` (30 % of maximum) for safety.
            timeout_s: Maximum seconds to wait for the motion to complete.
                Defaults to 30 s.
        """
        home = self._cfg.default_home if self._cfg.default_home else [0.0] * self.arm.num_joints
        done = threading.Event()

        def _go():
            try:
                self.arm.move_j(home, vlim=vlim, profile="min_jerk")
            except Exception as exc:
                logger.warning("safe_home move failed: %s", exc)
            finally:
                done.set()

        t = threading.Thread(target=_go, daemon=True)
        t.start()
        if not done.wait(timeout=timeout_s):
            logger.warning("safe_home timed out after %.1f s, cancelling motion", timeout_s)
            self.arm._abort_event.set()
            t.join(timeout=2.0)
            self.arm._abort_event.clear()
            try:
                self.arm.set_to_damping()
            except Exception as exc:
                logger.warning("safe_home: damping fallback after timeout failed: %s", exc)

    def move_to_ik(self, target: Pose6D, vlim: float = 1.0) -> bool:
        """Move to a target pose using IK + joint-space motion.

        Solves IK for *target* using the current joint state as seed, then
        executes a ``move_j`` with a min-jerk profile.

        Args:
            target: Desired end-effector pose.
            vlim: Velocity limit as a fraction of maximum.  Defaults to ``1.0``.

        Returns:
            ``True`` on success.
        """
        q = self.arm.solve_ik(target)
        self.arm.move_j(q, vlim=vlim, profile="min_jerk")
        return True

    def move_to_traj(self, target: Pose6D, duration_s: float = 2.0, vlim: float = 1.0) -> TrajResult:
        """Move to a target pose using CLIK trajectory planning.

        Plans a Cartesian geodesic trajectory, tracks it with a CLIK solver
        that includes joint-limit-aware null-space projection, then executes
        the resulting joint-space waypoints on the hardware.

        Args:
            target: Desired end-effector pose.
            duration_s: Trajectory duration in seconds.  Defaults to ``2.0``.
            vlim: Velocity limit as a fraction of maximum.  Defaults to ``1.0``.

        Returns:
            A :class:`TrajResult` with planning statistics.
        """
        start = self.arm.get_pose()
        ref = plan_se3_geodesic(start, target, duration_s=duration_s, dt_s=self._cfg.loop_dt_s)
        q0 = self.arm.get_joint_positions()
        q1 = self.arm.solve_ik(target)
        jt = plan_joint_space_trajectory(
            model=self._kin.pinocchio_model,
            end_frame_id=self._kin.end_frame_id if self._kin.end_frame_id is not None else 0,
            q_start=q0,
            q_end=q1,
            duration=duration_s,
            kin=self._kin,
            ik_params=IKParams(),
            null_gain=0.1,
        )
        self.arm.execute_joint_trajectory([p.q for p in jt], vlim=vlim, motion_name="move_l_clik")
        actual = [self._kin.forward(p.q) for p in jt]
        s = compute_geodesic_stats(ref, actual, success_flags=[p.ik_success for p in jt])
        return TrajResult(ok=True, points=s.total_points, max_pos_err=s.max_position_error, avg_pos_err=s.avg_position_error)

    def __enter__(self) -> ArmEndPos:
        return self

    def __exit__(self, *args) -> None:
        self.end()
