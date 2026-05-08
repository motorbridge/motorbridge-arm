"""Real-time background control loop for joint-space PD + gravity compensation.

Provides :class:`RealtimeController` that runs a fixed-frequency background
thread for sending motor commands and receiving state.  Integrates with the
timestamped :class:`JointStateInterpolator` for smooth trajectory execution.

Inspired by the arx5-sdk 500 Hz background send/recv architecture.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from ..arm import Arm
from ..trajectory.timestamped_interpolator import JointStateInterpolator, Waypoint

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ControllerGains:
    """Per-joint PD gains. / 每关节 PD 增益。

    Attributes:
        kp: Position proportional gains (one per joint).
        kd: Velocity derivative gains (one per joint).
    """

    kp: list[float]
    kd: list[float]

    @property
    def dof(self) -> int:
        return len(self.kp)


class RealtimeController:
    """Fixed-frequency background joint PD controller with gravity compensation.

    Runs a real-time control loop at a configurable frequency (default 500 Hz)
    that:
    1. Reads the current joint state from the arm.
    2. Interpolates the desired state from a timestamped trajectory.
    3. Computes a PD torque command: ``tau = kp * (q_d - q) + kd * (qd_d - qd)``.
    4. Optionally adds gravity compensation feedforward.
    5. Sends the command to the motors.

    Usage::

        with RealtimeController(arm, rate_hz=500) as ctrl:
            interp = ctrl.interpolator
            interp.init(start_wp, end_wp)
            # ... later ...
            interp.append_waypoint(t, new_wp)

    Args:
        arm: A connected :class:`Arm` instance.
        rate_hz: Control loop frequency in Hz.  Default 500 Hz.
        gravity_comp: Whether to enable gravity compensation feedforward.
            Default ``False``.
    """

    def __init__(
        self,
        arm: Arm,
        rate_hz: float = 500.0,
        gravity_comp: bool = False,
    ) -> None:
        self._arm = arm
        self._dt = 1.0 / max(rate_hz, 1.0)
        self._rate_hz = rate_hz
        self._dof = arm.num_joints
        self._interpolator = JointStateInterpolator(self._dof, method="cubic")

        self._kp = [80.0] * self._dof
        self._kd = [2.0] * self._dof

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False
        self._gravity_comp = gravity_comp
        self._t0: float = 0.0

    @property
    def interpolator(self) -> JointStateInterpolator:
        """The trajectory interpolator used by this controller."""
        return self._interpolator

    @property
    def gains(self) -> ControllerGains:
        return ControllerGains(kp=list(self._kp), kd=list(self._kd))

    @gains.setter
    def gains(self, g: ControllerGains) -> None:
        if g.dof != self._dof:
            raise ValueError(f"gains DOF {g.dof} != arm DOF {self._dof}")
        self._kp = list(g.kp)
        self._kd = list(g.kd)

    @property
    def running(self) -> bool:
        return self._running

    @property
    def gravity_comp_enabled(self) -> bool:
        """Whether gravity compensation is configured for this controller."""
        return self._gravity_comp

    @property
    def rate_hz(self) -> float:
        """The configured control loop rate in Hz."""
        return self._rate_hz

    def set_gain(self, kp: list[float], kd: list[float]) -> None:
        """Set per-joint PD gains.

        Args:
            kp: Position proportional gains.
            kd: Velocity derivative gains.
        """
        if len(kp) != self._dof or len(kd) != self._dof:
            raise ValueError(f"expected {self._dof} gains")
        self._kp = list(kp)
        self._kd = list(kd)

    def start(self) -> None:
        """Start the background control loop."""
        if self._running:
            return
        self._arm.mode_mit()
        if self._gravity_comp:
            self._arm.enable_gravity_comp()
        self._stop_event.clear()
        self._t0 = time.monotonic()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="rt-ctrl")
        self._thread.start()
        logger.info("RealtimeController started at %.0f Hz", self._rate_hz)

    def stop(self) -> None:
        """Stop the background control loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._running = False
        logger.info("RealtimeController stopped")

    def reset_to_home(self) -> None:
        """Set a trajectory target to the home position."""
        if not self._interpolator.initialized:
            q = self._arm.get_joint_positions()
            now = time.monotonic() - self._t0
            start = Waypoint(timestamp=now, pos=list(q))
            self._interpolator.init_fixed(start)
        home = [0.0] * self._dof
        now = time.monotonic() - self._t0
        q = self._arm.get_joint_positions()
        start = Waypoint(timestamp=now, pos=list(q))
        end = Waypoint(timestamp=now + 2.0, pos=home)
        self._interpolator.init(start, end)

    def set_to_damping(self) -> None:
        """Switch to damping mode (kp=0, kd=low) so the arm is compliant."""
        self._kp = [0.0] * self._dof
        self._kd = [0.5] * self._dof

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                self._tick()
            except Exception as exc:
                logger.warning("realtime tick error: %s", exc)
            elapsed = time.monotonic() - t0
            wait = self._dt - elapsed
            if wait > 0:
                self._stop_event.wait(timeout=wait)

    def _tick(self) -> None:
        now = time.monotonic() - self._t0

        if not self._interpolator.initialized:
            return

        wp = self._interpolator.interpolate(now)
        q_d = wp.pos
        qd_d = wp.vel if wp.vel else [0.0] * self._dof

        try:
            q = self._arm.get_joint_positions()
            qd = self._arm.get_joint_velocities()
        except Exception:
            return

        tau = [0.0] * self._dof
        for i in range(self._dof):
            tau[i] = self._kp[i] * (q_d[i] - q[i]) + self._kd[i] * (qd_d[i] - qd[i])

        self._arm.mit(pos=q_d, vel=qd_d, kp=self._kp, kd=self._kd, tau=tau)

    def __enter__(self) -> RealtimeController:
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
