"""Timestamped joint-state trajectory interpolator with dynamic waypoint management.

Provides :class:`JointStateInterpolator` for real-time interpolation between
timestamped waypoints using linear or cubic Hermite methods, with support for
dynamic trajectory modification (append, override).

Inspired by the arx5-sdk ``JointStateInterpolator`` design.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Waypoint:
    """A single timestamped joint-state waypoint.

    Attributes:
        timestamp: Absolute time in seconds.
        pos: Joint positions (radians).
        vel: Joint velocities (rad/s).  Used by cubic Hermite interpolation.
        torque: Joint torques (Nm).  Linearly interpolated.
        gripper_pos: Gripper position.  Linearly interpolated.
    """

    timestamp: float
    pos: list[float]
    vel: list[float] = field(default_factory=list)
    torque: list[float] = field(default_factory=list)
    gripper_pos: float = 0.0

    @property
    def dof(self) -> int:
        return len(self.pos)


class JointStateInterpolator:
    """Timestamped joint-state trajectory interpolator.

    Stores an ordered sequence of :class:`Waypoint` objects and can interpolate
    the joint state at any given time.  Supports two interpolation methods:

    * ``"linear"`` — linear interpolation of position, velocity, and torque.
    * ``"cubic"`` — cubic Hermite spline for position and velocity, ensuring
      C1 continuity (both position and velocity are continuous at waypoints).
      Torque is linearly interpolated.

    The trajectory can be modified dynamically using :meth:`append_waypoint`,
    :meth:`override_waypoint`, :meth:`append_traj`, and :meth:`override_traj`.

    Usage::

        interp = JointStateInterpolator(dof=6, method="cubic")
        interp.init(start_wp, end_wp)
        state = interp.interpolate(0.5)
        interp.append_waypoint(0.3, new_end_wp)

    Args:
        dof: Number of joints.
        method: Interpolation method — ``"linear"`` or ``"cubic"``.
    """

    def __init__(self, dof: int, method: str = "linear") -> None:
        if method not in ("linear", "cubic"):
            raise ValueError(f"method must be 'linear' or 'cubic', got '{method}'")
        self._dof = dof
        self._method = method
        self._traj: list[Waypoint] = []
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def dof(self) -> int:
        return self._dof

    @property
    def method(self) -> str:
        return self._method

    @property
    def waypoints(self) -> list[Waypoint]:
        return list(self._traj)

    def init(self, start: Waypoint, end: Waypoint) -> None:
        """Initialize with a start and end waypoint.

        Args:
            start: Starting waypoint.
            end: Ending waypoint.  Must have a timestamp >= start timestamp.

        Raises:
            ValueError: If timestamps are invalid or DOF mismatches.
        """
        if end.timestamp < start.timestamp:
            raise ValueError("end.timestamp must be >= start.timestamp")
        if start.timestamp == end.timestamp:
            raise ValueError("start and end timestamps are equal; use init_fixed() instead")
        self._check_dof(start)
        self._check_dof(end)
        self._traj = [start, end]
        self._initialized = True

    def init_fixed(self, wp: Waypoint) -> None:
        """Initialize with a single fixed waypoint (no motion).

        Args:
            wp: The fixed waypoint to hold.
        """
        self._check_dof(wp)
        self._traj = [wp]
        self._initialized = True

    def append_waypoint(self, current_time: float, end: Waypoint) -> None:
        """Append a new waypoint to the end of the trajectory.

        The trajectory is rebuilt from the interpolated state at *current_time*
        to *end*.  Any existing waypoints between *current_time* and
        *end.timestamp* that fall before the new endpoint are preserved;
        waypoints beyond *end.timestamp* are replaced.

        Args:
            current_time: Current time in seconds.
            end: New endpoint waypoint.

        Raises:
            RuntimeError: If the interpolator is not initialized.
            ValueError: If timestamps are invalid.
        """
        self._require_init()
        self._check_dof(end)
        if end.timestamp <= current_time:
            raise ValueError("end.timestamp must be > current_time")

        current_state = self._interpolate_at(current_time)
        prev_traj = self._traj
        self._traj = [current_state]
        for wp in prev_traj:
            if wp.timestamp > current_time:
                if wp.timestamp > end.timestamp:
                    self._traj.append(end)
                    break
                else:
                    self._traj.append(wp)
        else:
            self._traj.append(end)

    def override_waypoint(self, current_time: float, end: Waypoint) -> None:
        """Replace the entire trajectory from *current_time* with a single new endpoint.

        Args:
            current_time: Current time in seconds.
            end: New endpoint waypoint.
        """
        self._require_init()
        self._check_dof(end)
        if end.timestamp <= current_time:
            raise ValueError("end.timestamp must be > current_time")

        current_state = self._interpolate_at(current_time)
        self._traj = [current_state, end]

    def append_traj(self, current_time: float, traj: list[Waypoint]) -> None:
        """Append a new trajectory segment, merging with the existing one.

        Existing waypoints before *current_time* are preserved.  The new
        trajectory is appended after them.

        Args:
            current_time: Current time in seconds.
            traj: New trajectory segment (waypoints in ascending time order).
        """
        self._require_init()
        traj = [wp for wp in traj if wp.timestamp >= current_time]
        if not traj:
            return
        self._validate_traj(traj)

        current_state = self._interpolate_at(current_time)
        prev_traj = self._traj
        self._traj = [current_state]
        new_start = traj[0].timestamp

        for wp in prev_traj:
            if current_time < wp.timestamp < new_start:
                self._traj.append(wp)
        self._traj.extend(traj)

    def override_traj(self, current_time: float, traj: list[Waypoint]) -> None:
        """Replace the trajectory from *current_time* onward with a new segment.

        Args:
            current_time: Current time in seconds.
            traj: New trajectory segment.
        """
        self._require_init()
        traj = [wp for wp in traj if wp.timestamp >= current_time]
        if not traj:
            return
        self._validate_traj(traj)

        current_state = self._interpolate_at(current_time)
        self._traj = [current_state] + traj

    def interpolate(self, time: float) -> Waypoint:
        """Interpolate the joint state at the given time.

        Clamps to the first or last waypoint if *time* is outside the
        trajectory range.

        Args:
            time: Query time in seconds.

        Returns:
            A :class:`Waypoint` with the interpolated state.
        """
        if not self._initialized or not self._traj:
            raise RuntimeError("interpolator not initialized")
        return self._interpolate_at(time)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _interpolate_at(self, time: float) -> Waypoint:
        if not self._traj:
            raise RuntimeError("empty trajectory")

        if len(self._traj) == 1:
            wp = self._traj[0]
            return Waypoint(
                timestamp=time,
                pos=list(wp.pos),
                vel=list(wp.vel) if wp.vel else [0.0] * self._dof,
                torque=list(wp.torque) if wp.torque else [0.0] * self._dof,
                gripper_pos=wp.gripper_pos,
            )

        if time <= self._traj[0].timestamp:
            wp = self._traj[0]
            return Waypoint(time, list(wp.pos), list(wp.vel) if wp.vel else [], list(wp.torque) if wp.torque else [], wp.gripper_pos)

        if time >= self._traj[-1].timestamp:
            wp = self._traj[-1]
            return Waypoint(time, list(wp.pos), list(wp.vel) if wp.vel else [], list(wp.torque) if wp.torque else [], wp.gripper_pos)

        for i in range(len(self._traj) - 1):
            s = self._traj[i]
            e = self._traj[i + 1]
            if s.timestamp <= time <= e.timestamp:
                dt = e.timestamp - s.timestamp
                if dt < 1e-12:
                    return Waypoint(time, list(s.pos), list(s.vel) if s.vel else [], list(s.torque) if s.torque else [], s.gripper_pos)
                t = (time - s.timestamp) / dt
                return self._interp_pair(s, e, t, time)

        wp = self._traj[-1]
        return Waypoint(time, list(wp.pos), list(wp.vel) if wp.vel else [], list(wp.torque) if wp.torque else [], wp.gripper_pos)

    def _interp_pair(self, s: Waypoint, e: Waypoint, t: float, time: float) -> Waypoint:
        if self._method == "cubic" and s.vel and e.vel:
            t2 = t * t
            t3 = t2 * t
            h00 = 2.0 * t3 - 3.0 * t2 + 1.0
            h10 = t3 - 2.0 * t2 + t
            h01 = -2.0 * t3 + 3.0 * t2
            h11 = t3 - t2
            dt = e.timestamp - s.timestamp
            pos = [
                h00 * sp + h10 * sv * dt + h01 * ep + h11 * ev * dt
                for sp, sv, ep, ev in zip(s.pos, s.vel, e.pos, e.vel)
            ]
            vel = [
                (6.0 * t2 - 6.0 * t) * sp / dt
                + (3.0 * t2 - 4.0 * t + 1.0) * sv
                + (-6.0 * t2 + 6.0 * t) * ep / dt
                + (3.0 * t2 - 2.0 * t) * ev
                for sp, sv, ep, ev in zip(s.pos, s.vel, e.pos, e.vel)
            ]
            torque = self._lerp_list(s.torque, e.torque, t) if s.torque and e.torque else []
            gripper = (1.0 - t) * s.gripper_pos + t * e.gripper_pos
            return Waypoint(time, pos, vel, torque, gripper)
        else:
            pos = self._lerp_list(s.pos, e.pos, t)
            vel = self._lerp_list(s.vel, e.vel, t) if s.vel and e.vel else []
            torque = self._lerp_list(s.torque, e.torque, t) if s.torque and e.torque else []
            gripper = (1.0 - t) * s.gripper_pos + t * e.gripper_pos
            return Waypoint(time, pos, vel, torque, gripper)

    @staticmethod
    def _lerp_list(a: list[float], b: list[float], t: float) -> list[float]:
        return [(1.0 - t) * av + t * bv for av, bv in zip(a, b)]

    def _check_dof(self, wp: Waypoint) -> None:
        if len(wp.pos) != self._dof:
            raise ValueError(f"waypoint DOF ({len(wp.pos)}) != expected ({self._dof})")

    def _require_init(self) -> None:
        if not self._initialized:
            raise RuntimeError("interpolator not initialized")

    def _validate_traj(self, traj: list[Waypoint]) -> None:
        for wp in traj:
            self._check_dof(wp)
        for i in range(len(traj) - 1):
            if traj[i].timestamp > traj[i + 1].timestamp:
                raise ValueError("trajectory timestamps must be in ascending order")


def calc_joint_vel_from_traj(
    traj: list[Waypoint],
    avg_window_s: float = 0.05,
) -> list[Waypoint]:
    """Compute joint velocities from a position-only trajectory.

    Uses a two-scale central-difference approach (full window + half window)
    to estimate velocity at each waypoint.  Modifies waypoints in-place.

    Args:
        traj: Ordered list of waypoints (timestamps ascending).
        avg_window_s: Averaging window in seconds.  Default 0.05 s.

    Returns:
        The same list with ``vel`` fields populated.
    """
    if len(traj) < 2:
        return traj
    dof = traj[0].dof
    n = len(traj)

    for i in range(n):
        ti = traj[i].timestamp
        # Find indices for full window and half window
        i0 = i
        while i0 > 0 and traj[i0 - 1].timestamp >= ti - avg_window_s:
            i0 -= 1
        i1 = i
        while i1 > 0 and traj[i1 - 1].timestamp >= ti - avg_window_s / 2:
            i1 -= 1
        i2 = i
        while i2 < n - 1 and traj[i2 + 1].timestamp <= ti + avg_window_s / 2:
            i2 += 1
        i3 = i
        while i3 < n - 1 and traj[i3 + 1].timestamp <= ti + avg_window_s:
            i3 += 1

        dt_full = traj[i3].timestamp - traj[i0].timestamp
        dt_half = traj[i2].timestamp - traj[i1].timestamp
        vel = [0.0] * dof
        if dt_full > 1e-9:
            for j in range(dof):
                v_full = (traj[i3].pos[j] - traj[i0].pos[j]) / dt_full
                if dt_half > 1e-9:
                    v_half = (traj[i2].pos[j] - traj[i1].pos[j]) / dt_half
                    vel[j] = 0.5 * (v_full + v_half)
                else:
                    vel[j] = v_full
        traj[i].vel = vel
    return traj
