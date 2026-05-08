"""Real-time Cartesian (task-space) controller with background IK.

Provides :class:`CartesianController` that accepts end-effector pose commands
and runs inverse kinematics in a background loop, sending joint-space PD
commands to the motors.

Inspired by the arx5-sdk ``Arx5CartesianController``.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from ..arm import Arm
from ..trajectory.timestamped_interpolator import Waypoint
from .realtime_controller import RealtimeController
from ..types import Pose6D

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EEFCommand:
    """End-effector command with timestamp.

    Attributes:
        timestamp: Absolute time in seconds.
        x, y, z: Position in metres.
        roll, pitch, yaw: Orientation in radians (RPY).
        gripper_pos: Gripper position.
    """

    timestamp: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    gripper_pos: float = 0.0

    def to_pose6d(self) -> Pose6D:
        return Pose6D(x=self.x, y=self.y, z=self.z, roll=self.roll, pitch=self.pitch, yaw=self.yaw)


class CartesianController:
    """Real-time Cartesian controller with background IK.

    Accepts end-effector pose commands via :meth:`set_eef_cmd` or trajectory
    segments via :meth:`set_eef_traj`.  A background thread runs at the
    configured frequency, solving IK for the desired EEF pose and sending
    joint PD commands to the motors.

    Usage::

        with CartesianController(arm, rate_hz=500) as ctrl:
            ctrl.set_eef_cmd(EEFCommand(
                timestamp=ctrl.time + 0.1,
                x=0.3, y=0.0, z=0.4,
                roll=0, pitch=0, yaw=0,
            ))

    Args:
        arm: A connected :class:`Arm` instance.
        rate_hz: Control loop frequency.  Default 500 Hz.
        gravity_comp: Enable gravity compensation.  Default ``False``.
    """

    def __init__(
        self,
        arm: Arm,
        rate_hz: float = 500.0,
        gravity_comp: bool = False,
    ) -> None:
        self._arm = arm
        self._rt = RealtimeController(arm, rate_hz=rate_hz, gravity_comp=gravity_comp)
        self._dof = arm.num_joints
        self._lock = threading.Lock()

        self._eef_cmd: EEFCommand | None = None
        self._eef_cmd_time: float = 0.0

        self._t0: float = 0.0
        self._running = False

    @property
    def time(self) -> float:
        """Controller time in seconds since start."""
        return time.monotonic() - self._t0 if self._running else 0.0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def gains(self):
        return self._rt.gains

    @gains.setter
    def gains(self, g):
        self._rt.gains = g

    def set_gain(self, kp: list[float], kd: list[float]) -> None:
        self._rt.set_gain(kp, kd)

    def set_eef_cmd(self, cmd: EEFCommand) -> None:
        """Set a single end-effector target.

        The controller will move to this target by the command's timestamp.
        If the timestamp is 0, the controller uses the current time plus
        the default preview time (0.1 s).

        Args:
            cmd: The EEF target command.
        """
        with self._lock:
            self._eef_cmd = cmd
            self._eef_cmd_time = time.monotonic()

    def set_eef_traj(self, traj: list[EEFCommand]) -> None:
        """Set a timestamped EEF trajectory.

        Each element of *traj* is converted to a joint-space waypoint via IK
        and fed into the underlying real-time controller's interpolator.

        Args:
            traj: Ordered list of EEF commands (ascending timestamps).
        """
        if not traj:
            return

        now = self.time
        joint_traj: list[Waypoint] = []
        for cmd in traj:
            if cmd.timestamp <= now:
                continue
            try:
                q = self._arm.solve_ik(cmd.to_pose6d())
                joint_traj.append(Waypoint(
                    timestamp=cmd.timestamp,
                    pos=list(q),
                    gripper_pos=cmd.gripper_pos,
                ))
            except Exception as exc:
                logger.warning("IK failed for EEF traj point at t=%.3f: %s", cmd.timestamp, exc)
                continue

        if not joint_traj:
            return

        interp = self._rt.interpolator
        if not interp.initialized:
            q0 = self._arm.get_joint_positions()
            start = Waypoint(timestamp=now, pos=list(q0))
            if len(joint_traj) == 1:
                end = joint_traj[0]
                interp.init(start, end)
            else:
                interp.init(start, joint_traj[0])
                interp.append_traj(joint_traj[0].timestamp, joint_traj[1:])
        else:
            interp.override_traj(now, joint_traj)

    def get_eef_state(self) -> Pose6D:
        """Return the current end-effector pose."""
        return self._arm.get_pose()

    def get_joint_state(self):
        """Return current joint positions."""
        return self._arm.get_joint_positions()

    def get_home_pose(self) -> Pose6D:
        """Return the home pose (FK at all-zeros)."""
        return self._arm.get_pose()

    def reset_to_home(self) -> None:
        self._rt.reset_to_home()

    def set_to_damping(self) -> None:
        self._rt.set_to_damping()

    def start(self) -> None:
        """Start the Cartesian control loop."""
        if self._running:
            return
        self._t0 = time.monotonic()
        self._running = True

        self._arm.mode_mit()
        if self._rt.gravity_comp_enabled:
            self._arm.enable_gravity_comp()

        q0 = self._arm.get_joint_positions()
        start = Waypoint(timestamp=0.0, pos=list(q0))
        self._rt.interpolator.init_fixed(start)
        self._rt.start()
        logger.info("CartesianController started at %.0f Hz", self._rt.rate_hz)

    def stop(self) -> None:
        """Stop the Cartesian control loop."""
        self._rt.stop()
        self._running = False
        logger.info("CartesianController stopped")

    def _update_ik_target(self) -> None:
        with self._lock:
            cmd = self._eef_cmd
            self._eef_cmd = None

        if cmd is None:
            return

        ts = cmd.timestamp if cmd.timestamp > 0 else self.time + 0.1
        try:
            q = self._arm.solve_ik(cmd.to_pose6d())
        except Exception as exc:
            logger.warning("Cartesian IK failed: %s", exc)
            return

        wp = Waypoint(timestamp=ts, pos=list(q), gripper_pos=cmd.gripper_pos)
        interp = self._rt.interpolator
        now = self.time
        if not interp.initialized:
            q0 = self._arm.get_joint_positions()
            start = Waypoint(timestamp=now, pos=list(q0))
            interp.init(start, wp)
        else:
            try:
                interp.override_waypoint(now, wp)
            except ValueError:
                q0 = self._arm.get_joint_positions()
                start = Waypoint(timestamp=now, pos=list(q0))
                interp.init(start, wp)

    def __enter__(self) -> CartesianController:
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
