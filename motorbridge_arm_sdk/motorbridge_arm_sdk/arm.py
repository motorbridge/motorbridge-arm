from __future__ import annotations

import logging
from dataclasses import asdict
import threading
import time

from .calibration.zeroing import ZeroCalibrator
from .motion.executor import JointMotionExecutor
from .motion.planner import ArcSpec, estimate_steps, interpolate_joint_linear, interpolate_pose_circular, interpolate_pose_geodesic, interpolate_pose_linear
from .trajectory.clik_tracker import IKParams, track_trajectory
from .trajectory.trajectory_planner import plan_joint_space_trajectory
from .trajectory.sampler import CartesianTrajectory, CartesianPoint, TrajPlanParams
from .params.registry import ParamRegistry, create_default_registry
from .runtime import RuntimeStateMachine
from .safety.supervisor import SafetySupervisor
from .session import ModeLike, MotorBridgeSession
from .telemetry.recorder import Recorder
from .telemetry.state_cache import StateCache
from .types import ArmConfig, ArmRunState, ArmState, FaultState, JointState, PayloadConfig, Pose6D, ToolConfig
from .model.kinematics import Kinematics
from .vendors import MotorAdapterRegistry, create_default_adapter_registry

logger = logging.getLogger(__name__)


class Arm:
    """High-level robotic arm controller.

    Provides a unified interface for connecting to, controlling, and monitoring
    a robotic arm built on the MotorBridge platform.  Wraps session management,
    safety checks, kinematics, motion planning, and telemetry recording into a
    single object that can be used as a context manager.

    Typical usage::

        with Arm(config) as arm:
            arm.connect()
            arm.enable()
            arm.home()
            arm.move_j([0.5, -0.3, 0.8, 0.0, 0.4, 0.0])
            arm.close()

    Args:
        config: Arm configuration containing joint definitions, URDF path,
            communication channel, and safety limits.
        registry: Optional parameter registry for vendor-specific motor
            parameters.  Falls back to the default registry if not provided.
        adapter_registry: Optional motor adapter registry for hardware
            communication.  Falls back to the default adapter registry.
    """

    def __init__(
        self,
        config: ArmConfig,
        registry: ParamRegistry | None = None,
        adapter_registry: MotorAdapterRegistry | None = None,
    ) -> None:
        self._cfg = config
        self._runtime = RuntimeStateMachine(ArmRunState.DISCONNECTED)
        self._adapter_registry = adapter_registry or create_default_adapter_registry()
        self._session = MotorBridgeSession(config.channel, adapter_registry=self._adapter_registry)
        self._safety = SafetySupervisor(config)
        self._cache = StateCache(config)
        self._executor = JointMotionExecutor(config.loop_dt_s)
        self._zero = ZeroCalibrator(self._session)
        self._kin = Kinematics(config.urdf_path, config.ee_frame)
        self._registry = registry or create_default_registry()
        self._recorder = Recorder()
        self._tool = ToolConfig()
        self._payload = PayloadConfig()
        self._mode = "pos_vel"
        self._ctrl_thread: threading.Thread | None = None
        self._ctrl_running = False
        self._ctrl_fn = None
        self._abort_event = threading.Event()

    def connect(self) -> None:
        """Establish communication with the robotic arm hardware.

        Opens the underlying MotorBridge session, registers all configured
        joints, and transitions the runtime state to IDLE.

        Raises:
            RuntimeError: If the session cannot be opened or the runtime
                state does not allow a transition to IDLE.
        """
        self._runtime.transition(ArmRunState.IDLE)
        self._session.connect()
        for j in self._cfg.joints:
            self._session.add_joint(j)
        self._cache.update_run_state(ArmRunState.IDLE)
        self._recorder.add("connect", {"channel": self._cfg.channel, "joints": len(self._cfg.joints)})

    @property
    def num_joints(self) -> int:
        return len(self._cfg.joints)

    @property
    def joint_names(self) -> list[str]:
        return [j.name for j in self._cfg.joints]

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def control_loop_active(self) -> bool:
        t = self._ctrl_thread
        return t is not None and t.is_alive()

    def close(self) -> None:
        """Shut down the arm controller and release all resources.

        Stops any running control loop, forces the runtime state to
        DISCONNECTED, and closes the underlying MotorBridge session.
        """
        self.stop_control_loop()
        self._runtime.force(ArmRunState.DISCONNECTED)
        self._session.close()
        self._cache.update_run_state(ArmRunState.DISCONNECTED)

    def reconnect(self) -> None:
        """Close the current connection and re-establish a fresh one.

        Convenience wrapper that calls :meth:`close` followed by
        :meth:`connect`.  Useful for recovering from transient communication
        errors.
        """
        self.close()
        self.connect()

    def enable(self) -> None:
        """Enable all joint motors and transition to the ENABLED state.

        After calling this method the arm is powered and ready to accept
        motion commands.

        Raises:
            RuntimeError: If the runtime is not in a state that allows
                transition to ENABLED (e.g. currently DISCONNECTED).
        """
        self._runtime.transition(ArmRunState.ENABLED)
        self._session.enable_all()
        self._cache.update_run_state(ArmRunState.ENABLED)

    def disable(self) -> None:
        """Disable all joint motors and transition back to IDLE.

        If the arm is currently RUNNING it is first transitioned to ENABLED
        before being disabled.  All joint motors are de-energised.
        """
        if self._runtime.state == ArmRunState.RUNNING:
            self._runtime.transition(ArmRunState.ENABLED)
        self._runtime.transition(ArmRunState.IDLE)
        self._session.disable_all()
        self._cache.update_run_state(ArmRunState.IDLE)

    def estop(self) -> None:
        """Perform an emergency stop.

        Immediately disables all joint motors, aborts any running trajectory,
        and forces the runtime into the FAULT state.  Use :meth:`clear_faults`
        and :meth:`enable` to resume normal operation after an e-stop.
        """
        self._abort_event.set()
        self._session.disable_all()
        self._runtime.force(ArmRunState.FAULT)
        self._cache.update_run_state(ArmRunState.FAULT)

    def refresh_state(self) -> ArmState:
        """Request fresh feedback from all joints and update the state cache.

        Sends a feedback request to every motor, reads position, velocity,
        torque, status, and temperature values, applies zero-offset and
        direction corrections, then stores the results in the internal
        state cache.

        Returns:
            An :class:`ArmState` snapshot reflecting the latest hardware
            readings.
        """
        self._session.request_feedback_all()
        for i, h in enumerate(self._session.joints):
            raw = h.motor.get_state()
            if raw is None:
                continue
            q = (raw.pos - h.config.zero_offset) / h.config.direction
            self._cache.update_joint(
                i,
                JointState(
                    name=h.config.name,
                    pos=q,
                    vel=raw.vel,
                    torq=raw.torq,
                    status_code=raw.status_code,
                    t_mos=raw.t_mos,
                    t_rotor=raw.t_rotor,
                ),
            )
        return self._cache.snapshot()

    def get_state(self) -> ArmState:
        """Return the current arm state without requesting new hardware feedback.

        Returns a cached snapshot.  Call :meth:`refresh_state` first if
        up-to-date readings are required.

        Returns:
            The most recent :class:`ArmState` snapshot from the internal
            cache.
        """
        return self._cache.snapshot()

    def get_joint_positions(self) -> list[float]:
        """Refresh hardware feedback and return joint positions.

        Returns:
            A list of joint position values in radians, one per configured
            joint.  Joints with no reading default to ``0.0``.
        """
        st = self.refresh_state()
        out: list[float] = []
        for j in st.joints:
            out.append(0.0 if j.pos is None else float(j.pos))
        return out

    def get_positions(self, request: bool = True):
        """Return joint positions, optionally refreshing from hardware.

        This is a convenience alias for :meth:`get_joint_positions`.  The
        *request* parameter is currently unused and accepted for API
        compatibility.

        Args:
            request: If ``True`` (default), fresh hardware feedback is
                requested before returning positions.

        Returns:
            A list of joint position values in radians.
        """
        _ = request
        return self.get_joint_positions()

    def get_velocities(self, request: bool = True):
        """Return joint velocities, optionally refreshing from hardware.

        Args:
            request: If ``True`` (default), fresh hardware feedback is
                requested before returning velocities.

        Returns:
            A list of joint velocity values in rad/s.  Joints with no
            reading default to ``0.0``.
        """
        _ = request
        st = self.refresh_state()
        return [0.0 if j.vel is None else float(j.vel) for j in st.joints]

    def get_torques(self, request: bool = True):
        """Return joint torques, optionally refreshing from hardware.

        Args:
            request: If ``True`` (default), fresh hardware feedback is
                requested before returning torques.

        Returns:
            A list of joint torque values in Nm.  Joints with no reading
            default to ``0.0``.
        """
        _ = request
        st = self.refresh_state()
        return [0.0 if j.torq is None else float(j.torq) for j in st.joints]

    def get_state_vectors(self):
        """Return positions, velocities, and torques in a single call.

        Each sub-call refreshes hardware feedback independently.

        Returns:
            A 3-tuple ``(positions, velocities, torques)`` where each
            element is a list of floats with one entry per configured joint.
        """
        return self.get_positions(), self.get_velocities(), self.get_torques()

    def get_joint_state(self, joint: int | str) -> JointState:
        """Return the full state of a single joint after refreshing hardware.

        Args:
            joint: Joint identifier -- either a zero-based integer index or
                the string name of the joint.

        Returns:
            A :class:`JointState` object containing position, velocity,
            torque, status code, and temperature data for the requested
            joint.

        Raises:
            IndexError: If an integer index is out of range.
            KeyError: If a string name does not match any configured joint.
        """
        st = self.refresh_state()
        idx = self._resolve_joint_index(joint)
        return st.joints[idx]

    def get_pose(self) -> Pose6D:
        """Compute the current end-effector pose via forward kinematics.

        Uses the latest joint positions and adds the configured tool offset.

        Returns:
            A :class:`Pose6D` describing the end-effector position (m) and
            orientation (rad) in the base frame.
        """
        q = self.get_joint_positions()
        base = self._kin.forward(q)
        return Pose6D(
            x=base.x + self._tool.x,
            y=base.y + self._tool.y,
            z=base.z + self._tool.z,
            roll=base.roll + self._tool.roll,
            pitch=base.pitch + self._tool.pitch,
            yaw=base.yaw + self._tool.yaw,
        )

    def move_j(self, q_target: list[float], vlim: float = 1.0, profile: str | None = None) -> None:
        """Move all joints to target positions in joint space.

        The target is clamped to the configured joint limits before execution.
        The motion is interpolated from the current positions using the
        specified velocity limit and interpolation profile.

        Args:
            q_target: List of target joint positions in radians, one per
                configured joint.
            vlim: Velocity limit as a fraction of the maximum joint velocity.
                Defaults to ``1.0`` (full speed).
            profile: Interpolation profile name (e.g. ``"linear"``,
                ``"trapezoid"``).  Falls back to the profile specified in
                the arm configuration.

        Raises:
            ValueError: If the target list length does not match the number
                of joints or values are outside joint limits after clamping.
        """
        q_target = self._safety.clamp_joint_targets(q_target)
        self._run_joint_target(q_target, vlim=vlim, profile=profile or self._cfg.motion_profile)

    def mode_mit(self) -> bool:
        """Switch all joints to MIT impedance control mode.

        In MIT mode each joint accepts position, velocity, stiffness (kp),
        damping (kd), and feed-forward torque (tau) parameters.

        Returns:
            ``True`` on success.
        """
        self._mode = "mit"
        self._session.ensure_mode_all(ModeLike.MIT)
        return True

    def mode_pos_vel(self) -> bool:
        """Switch all joints to position-velocity control mode.

        This is the default mode.  Each joint accepts a target position and
        a velocity limit.

        Returns:
            ``True`` on success.
        """
        self._mode = "pos_vel"
        self._session.ensure_mode_all(ModeLike.POS_VEL)
        return True

    def mode_vel(self) -> bool:
        """Switch all joints to pure velocity control mode.

        In velocity mode each joint accepts a target velocity value.

        Returns:
            ``True`` on success.
        """
        self._mode = "vel"
        self._session.ensure_mode_all(ModeLike.VEL)
        return True

    def mit(self, pos, vel=None, kp=None, kd=None, tau=None) -> None:
        """Send an MIT impedance command to all joints simultaneously.

        Automatically switches to MIT mode before sending the command.
        Any omitted parameters default to zero-filled lists.

        Args:
            pos: List of target positions in radians, one per joint.
            vel: List of target velocities in rad/s.  Defaults to zeros.
            kp: List of stiffness gains.  Defaults to zeros.
            kd: List of damping gains.  Defaults to zeros.
            tau: List of feed-forward torques in Nm.  Defaults to zeros.
        """
        n = len(self._cfg.joints)
        vel = vel if vel is not None else [0.0] * n
        kp = kp if kp is not None else [0.0] * n
        kd = kd if kd is not None else [0.0] * n
        tau = tau if tau is not None else [0.0] * n
        self.mode_mit()
        self._session.set_mit_all(list(pos), list(vel), list(kp), list(kd), list(tau))

    def pos_vel(self, pos, vlim=None) -> None:
        """Send a position-velocity command to all joints simultaneously.

        Automatically switches to position-velocity mode before sending the
        command.

        Args:
            pos: List of target positions in radians, one per joint.
            vlim: Velocity limit.  May be a single float, a list/tuple with
                one element, or ``None`` to use the minimum joint velocity
                limit from the arm configuration.
        """
        n = len(self._cfg.joints)
        if vlim is None:
            vmax = min(j.limit_vel for j in self._cfg.joints)
        else:
            if isinstance(vlim, (list, tuple)):
                vmax = float(vlim[0]) if len(vlim) > 0 else 1.0
            else:
                vmax = float(vlim)
        self.mode_pos_vel()
        self._session.set_pos_vel_all(list(pos), vmax)

    def set_vel(self, vel) -> None:
        """Send a velocity command to all joints simultaneously.

        Automatically switches to velocity mode before sending the command.

        Args:
            vel: List of target velocities in rad/s, one per joint.
        """
        self.mode_vel()
        self._session.set_vel_all(list(vel))

    def move_joint(self, joint: int | str, pos: float, vlim: float = 1.0) -> None:
        """Move a single joint to a target position while holding the others.

        Reads the current positions of all joints, updates only the specified
        joint, and executes a joint-space motion.

        Args:
            joint: Joint identifier -- either a zero-based integer index or
                the string name of the joint.
            pos: Target position in radians for the specified joint.
            vlim: Velocity limit as a fraction of maximum.  Defaults to ``1.0``.

        Raises:
            IndexError: If an integer index is out of range.
            KeyError: If a string name does not match any configured joint.
        """
        idx = self._resolve_joint_index(joint)
        q = self.get_joint_positions()
        q[idx] = pos
        self.move_j(q, vlim=vlim)

    def move_joints(self, targets: dict[int | str, float], vlim: float = 1.0) -> None:
        """Move a subset of joints to target positions while holding the rest.

        Args:
            targets: Mapping of joint identifiers (int index or str name) to
                target positions in radians.
            vlim: Velocity limit as a fraction of maximum.  Defaults to ``1.0``.

        Raises:
            IndexError: If an integer index is out of range.
            KeyError: If a string name does not match any configured joint.
        """
        q = self.get_joint_positions()
        for j, p in targets.items():
            idx = self._resolve_joint_index(j)
            q[idx] = p
        self.move_j(q, vlim=vlim)

    def joint_vel(self, joint: int | str, vel: float) -> None:
        """Command a single joint in velocity mode.

        Only the specified joint is switched to velocity mode; the mode of
        all other joints is unchanged.

        Args:
            joint: Joint identifier -- either a zero-based integer index or
                the string name of the joint.
            vel: Target velocity in rad/s.

        Raises:
            IndexError: If an integer index is out of range.
            KeyError: If a string name does not match any configured joint.
        """
        idx = self._resolve_joint_index(joint)
        self._session.ensure_mode_joint(idx, ModeLike.VEL)
        self._session.set_vel_joint(idx, vel)
        self._recorder.add("joint_vel", {"joint": self._cfg.joints[idx].name, "vel": vel})

    def joint_mit(self, joint: int | str, pos: float, vel: float, kp: float, kd: float, tau: float = 0.0) -> None:
        """Command a single joint in MIT impedance mode.

        Only the specified joint is switched to MIT mode; the mode of all
        other joints is unchanged.

        Args:
            joint: Joint identifier -- either a zero-based integer index or
                the string name of the joint.
            pos: Target position in radians.
            vel: Target velocity in rad/s.
            kp: Stiffness gain.
            kd: Damping gain.
            tau: Feed-forward torque in Nm.  Defaults to ``0.0``.

        Raises:
            IndexError: If an integer index is out of range.
            KeyError: If a string name does not match any configured joint.
        """
        idx = self._resolve_joint_index(joint)
        self._session.ensure_mode_joint(idx, ModeLike.MIT)
        self._session.set_mit_joint(idx, pos=pos, vel=vel, kp=kp, kd=kd, tau=tau)
        self._recorder.add(
            "joint_mit",
            {"joint": self._cfg.joints[idx].name, "pos": pos, "vel": vel, "kp": kp, "kd": kd, "tau": tau},
        )

    def home(self, vlim: float = 1.0) -> None:
        """Move all joints to the configured home position.

        If no home position is configured, all joints are moved to zero.

        Args:
            vlim: Velocity limit as a fraction of maximum.  Defaults to ``1.0``.
        """
        home = self._cfg.default_home or [0.0 for _ in self._cfg.joints]
        self.move_j(home, vlim=vlim)

    def solve_ik(self, target: Pose6D) -> list[float]:
        """Compute inverse kinematics for a desired end-effector pose.

        Uses the current joint positions as the seed and clamps the result
        to the configured joint limits.

        Args:
            target: Desired end-effector pose as a :class:`Pose6D`.

        Returns:
            A list of joint positions in radians that achieve the target
            pose, clamped to joint limits.
        """
        q_now = self.get_joint_positions()
        q = self._kin.inverse(target, q_now)
        q = self._safety.clamp_joint_targets(q)
        return q

    def move_l(self, target: Pose6D, vlim: float = 1.0, step_m: float = 0.01, profile: str | None = None) -> None:
        """Move the end-effector in a straight line (linear Cartesian motion).

        Computes a series of Cartesian waypoints from the current pose to
        *target*, solves inverse kinematics at each waypoint, and executes
        the resulting joint-space trajectory.  When Pinocchio is available
        a CLIK-based trajectory planner is used; otherwise a per-waypoint IK
        fallback is applied.

        Args:
            target: Desired end-effector pose as a :class:`Pose6D`.
            vlim: Velocity limit as a fraction of maximum.  Defaults to ``1.0``.
            step_m: Cartesian step size in metres between interpolated
                waypoints.  Smaller values yield smoother paths.  Defaults
                to ``0.01``.
            profile: Interpolation profile name (e.g. ``"linear"``,
                ``"geodesic"``).  Falls back to the arm configuration default.

        Raises:
            ValueError: If IK fails to converge for any waypoint.
        """
        start = self.get_pose()
        profile_name = profile or self._cfg.motion_profile
        dist = ((target.x - start.x) ** 2 + (target.y - start.y) ** 2 + (target.z - start.z) ** 2) ** 0.5
        steps = max(2, int(dist / max(step_m, 1e-4)) + 1)
        q_now = self.get_joint_positions()
        joint_points: list[list[float]] = []

        # Primary path: unified cartesian trajectory + CLIK tracking with joint-limit-aware null space.
        if self._kin.has_pinocchio and self._kin.pinocchio_model is not None and self._kin.end_frame_id is not None:
            try:
                q_goal = self._safety.clamp_joint_targets(self._kin.inverse(target, q_now))
                duration_s = max(self._cfg.loop_dt_s, (steps - 1) * self._cfg.loop_dt_s)
                ik_params = IKParams(
                    max_iter=200,
                    tolerance=1e-4,
                    damping=1e-6,
                    step_size=0.8,
                )
                traj_points = plan_joint_space_trajectory(
                    model=self._kin.pinocchio_model,
                    end_frame_id=self._kin.end_frame_id,
                    q_start=q_now,
                    q_end=q_goal,
                    duration=duration_s,
                    kin=self._kin,
                    ik_params=ik_params,
                    null_gain=0.1,
                    start_pose=start,
                    end_pose=target,
                )
                joint_points = [self._safety.clamp_joint_targets(pt.q) for pt in traj_points]
            except (ValueError, RuntimeError, ImportError) as exc:
                logger.warning("CLIK trajectory pipeline failed, falling back to per-waypoint IK: %s", exc)
                joint_points = []

        # Fallback path: per-waypoint IK.
        if not joint_points:
            if profile_name.lower() == "geodesic":
                poses = interpolate_pose_geodesic(start, target, steps, profile=profile_name)
            else:
                poses = interpolate_pose_linear(start, target, steps, profile=profile_name)
            for pose in poses:
                q_now = self._kin.inverse(pose, q_now)
                joint_points.append(self._safety.clamp_joint_targets(q_now))
        self._run_joint_points(joint_points, vlim=vlim, motion_name="move_l")

    def move_c(
        self,
        target: Pose6D,
        center_x: float,
        center_y: float,
        normal_z: float = 1.0,
        vlim: float = 1.0,
        steps: int = 80,
        profile: str | None = None,
    ) -> None:
        """Move the end-effector along a circular arc (Cartesian arc motion).

        Interpolates circular-arc Cartesian waypoints between the current
        pose and *target*, solves IK at each, and executes the joint-space
        trajectory.

        Args:
            target: Desired end-effector pose at the end of the arc.
            center_x: X coordinate of the arc centre.
            center_y: Y coordinate of the arc centre.
            normal_z: Z component of the arc-plane normal vector.  Defaults
                to ``1.0``.
            vlim: Velocity limit as a fraction of maximum.  Defaults to ``1.0``.
            steps: Number of interpolation steps along the arc.  Defaults to
                ``80``.
            profile: Interpolation profile name.  Falls back to the arm
                configuration default.

        Raises:
            ValueError: If IK fails to converge for any waypoint.
        """
        start = self.get_pose()
        poses = interpolate_pose_circular(
            start,
            target,
            ArcSpec(center_x=center_x, center_y=center_y, normal_z=normal_z),
            steps,
            profile=profile or self._cfg.motion_profile,
        )
        q_now = self.get_joint_positions()
        joint_points: list[list[float]] = []

        # Primary path: CLIK tracker with null-space avoidance.
        if self._kin.has_pinocchio and self._kin.pinocchio_model is not None and self._kin.end_frame_id is not None:
            try:
                duration_s = max(self._cfg.loop_dt_s, (steps - 1) * self._cfg.loop_dt_s)
                dt = duration_s / max(len(poses) - 1, 1)
                cart_traj = CartesianTrajectory()
                for i, pose in enumerate(poses):
                    cart_traj.add_point(i * dt, pose)
                ik_params = IKParams(max_iter=200, tolerance=1e-4, damping=1e-6, step_size=0.8)
                traj_result = track_trajectory(
                    model=self._kin.pinocchio_model,
                    end_frame_id=self._kin.end_frame_id,
                    traj=cart_traj,
                    q_init=q_now,
                    kin=self._kin,
                    ik_params=ik_params,
                    null_gain=0.1,
                )
                joint_points = [self._safety.clamp_joint_targets(pt.q) for pt in traj_result]
            except (ValueError, RuntimeError, ImportError) as exc:
                logger.warning("CLIK pipeline failed for move_c, falling back to per-waypoint IK: %s", exc)
                joint_points = []

        # Fallback path: per-waypoint IK.
        if not joint_points:
            for pose in poses:
                q_now = self._kin.inverse(pose, q_now)
                joint_points.append(self._safety.clamp_joint_targets(q_now))
        self._run_joint_points(joint_points, vlim=vlim, motion_name="move_c")

    def read_param(self, joint_index: int, param_id: int, param_type: str | None = None, timeout_ms: int = 1000):
        """Read a vendor-specific motor parameter from a single joint.

        The parameter type is resolved from the parameter registry when
        available; otherwise it falls back to ``"f32"``.

        Args:
            joint_index: Zero-based index of the target joint.
            param_id: Vendor-specific parameter identifier.
            param_type: Data type string (e.g. ``"f32"``, ``"u32"``).  If
                ``None``, the type is looked up in the parameter registry.
            timeout_ms: Communication timeout in milliseconds.  Defaults to
                ``1000``.

        Returns:
            The parameter value read from the motor, type depends on
            *param_type*.
        """
        spec = self._registry.get(self._cfg.joints[joint_index].vendor, param_id)
        ptype = param_type or (spec.param_type if spec else "f32")
        value = self._session.get_param(joint_index, param_id, ptype, timeout_ms)
        self._recorder.add("read_param", {"joint": joint_index, "param_id": param_id, "type": ptype, "value": value})
        return value

    def write_param(self, joint_index: int, param_id: int, value: int | float, param_type: str | None = None) -> None:
        """Write a vendor-specific motor parameter to a single joint.

        The parameter type is resolved from the parameter registry when
        available; otherwise it falls back to ``"f32"``.

        Args:
            joint_index: Zero-based index of the target joint.
            param_id: Vendor-specific parameter identifier.
            value: Value to write.
            param_type: Data type string (e.g. ``"f32"``, ``"u32"``).  If
                ``None``, the type is looked up in the parameter registry.
        """
        spec = self._registry.get(self._cfg.joints[joint_index].vendor, param_id)
        ptype = param_type or (spec.param_type if spec else "f32")
        self._session.set_param(joint_index, param_id, ptype, value)
        self._recorder.add("write_param", {"joint": joint_index, "param_id": param_id, "type": ptype, "value": value})

    def zero_calibrate(self, scope: str = "all", joint_index: int | None = None):
        """Perform zero-position calibration on one or all joints.

        When *scope* is ``"all"`` every joint is calibrated.  When *scope* is
        ``"joint"``, only the joint specified by *joint_index* is calibrated.

        Args:
            scope: ``"all"`` to calibrate every joint or ``"joint"`` for a
                single joint.  Defaults to ``"all"``.
            joint_index: Required when *scope* is ``"joint"``.  Zero-based
                index of the joint to calibrate.

        Returns:
            A calibration result object with at minimum an ``ok`` boolean
            attribute.

        Raises:
            ValueError: If *scope* is ``"joint"`` and *joint_index* is
                ``None``.
        """
        if scope == "all":
            result = self._zero.zero_all()
        else:
            if joint_index is None:
                raise ValueError("joint_index is required for scope=joint")
            result = self._zero.zero_joint(joint_index)
        self._recorder.add("zero_calibrate", {"scope": scope, "joint_index": joint_index, "ok": result.ok})
        return result

    def set_tool(self, tool: ToolConfig) -> None:
        """Configure the tool (end-effector) offset used for kinematics.

        The tool offset is added to the forward-kinematics result when
        computing the end-effector pose.

        Args:
            tool: A :class:`ToolConfig` describing the tool's positional
                and rotational offset from the flange frame.
        """
        self._tool = tool
        self._recorder.add("set_tool", asdict(tool))

    def set_payload(self, payload: PayloadConfig) -> None:
        """Configure the payload parameters for dynamics compensation.

        Args:
            payload: A :class:`PayloadConfig` describing the payload mass
                and centre of mass.
        """
        self._payload = payload
        self._recorder.add("set_payload", asdict(payload))

    def get_faults(self) -> FaultState:
        """Check all joints for fault conditions.

        Refreshes hardware feedback and inspects the status code of each
        joint.  A status code other than ``1`` is considered a fault.

        Returns:
            A :class:`FaultState` indicating whether any fault is present
            and listing the names of faulted joints.
        """
        st = self.refresh_state()
        bad: list[str] = []
        for j in st.joints:
            if j.status_code is None:
                continue
            if j.status_code != 1:
                bad.append(j.name)
        return FaultState(has_fault=len(bad) > 0, faulted_joints=bad)

    def clear_faults(self) -> None:
        """Clear fault conditions on all joints.

        Sends a clear-fault command to every motor.  After clearing you
        typically need to call :meth:`enable` to resume operation.
        """
        self._session.clear_fault_all()
        self._recorder.add("clear_faults", {"ok": True})

    def gripper_open(self, pos: float = 1.0, vlim: float = 1.0) -> None:
        """Open the gripper.

        Moves the gripper joint to the specified open position.

        Args:
            pos: Target position for the gripper joint.  Defaults to ``1.0``
                (fully open).
            vlim: Velocity limit as a fraction of maximum.  Defaults to ``1.0``.

        Raises:
            ValueError: If no gripper joint is configured.
        """
        self._gripper_move(pos=pos, vlim=vlim)

    def gripper_close(self, pos: float = 0.0, vlim: float = 1.0) -> None:
        """Close the gripper.

        Moves the gripper joint to the specified closed position.

        Args:
            pos: Target position for the gripper joint.  Defaults to ``0.0``
                (fully closed).
            vlim: Velocity limit as a fraction of maximum.  Defaults to ``1.0``.

        Raises:
            ValueError: If no gripper joint is configured.
        """
        self._gripper_move(pos=pos, vlim=vlim)

    def save_trace(self, path: str) -> None:
        """Save the recorded telemetry trace to a JSON file.

        All events logged by the telemetry recorder (connects, moves,
        parameter reads/writes, etc.) are written to *path*.

        Args:
            path: File-system path for the output JSON file.
        """
        self._recorder.save_json(path)

    def start_control_loop(self, callback, rate_hz: float | None = None) -> None:
        """Start a background control loop that calls *callback* at a fixed rate.

        The callback receives the :class:`Arm` instance and the computed loop
        period (``dt``) as arguments.  If a control loop is already running
        it is stopped first.

        Args:
            callback: A callable with signature
                ``(arm: Arm, dt: float) -> None`` invoked on every loop
                iteration.
            rate_hz: Desired loop rate in Hz.  If ``None`` or non-positive
                the loop period from the arm configuration (``loop_dt_s``)
                is used.
        """
        if self.control_loop_active:
            self.stop_control_loop()
        self._ctrl_fn = callback
        self._ctrl_running = True
        dt = self._cfg.loop_dt_s if rate_hz is None or rate_hz <= 0 else 1.0 / rate_hz

        def _loop():
            while self._ctrl_running:
                t0 = time.perf_counter()
                if self._ctrl_fn is not None:
                    self._ctrl_fn(self, dt)
                dt_sleep = dt - (time.perf_counter() - t0)
                if dt_sleep > 0:
                    time.sleep(dt_sleep)

        self._ctrl_thread = threading.Thread(target=_loop, daemon=True)
        self._ctrl_thread.start()

    def stop_control_loop(self) -> None:
        """Stop the background control loop started by :meth:`start_control_loop`.

        Signals the loop thread to exit and waits up to one second for it
        to join.  Safe to call even when no loop is running.
        """
        self._ctrl_running = False
        if self._ctrl_thread is not None:
            if threading.current_thread() is not self._ctrl_thread:
                self._ctrl_thread.join(timeout=1.0)
            self._ctrl_thread = None

    def _run_joint_target(self, q_target: list[float], vlim: float, profile: str = "linear") -> None:
        vlim = self._safety.validate_velocity_limit(vlim)
        q_now = self.get_joint_positions()
        steps = estimate_steps(q_now, q_target)
        points = interpolate_joint_linear(q_now, q_target, steps, profile=profile)
        self._run_joint_points(points, vlim=vlim, motion_name="move_j", steps=steps)

    def _run_joint_points(self, points: list[list[float]], vlim: float, motion_name: str, steps: int | None = None) -> None:
        self._abort_event.clear()
        self._runtime.transition(ArmRunState.RUNNING)
        self._session.ensure_mode_all(ModeLike.POS_VEL)
        self._cache.update_run_state(ArmRunState.RUNNING)
        try:
            self._executor.run(points, self._session.set_pos_vel_all, vlim, abort_event=self._abort_event)
        finally:
            self._runtime.transition(ArmRunState.ENABLED)
            self._cache.update_run_state(ArmRunState.ENABLED)
        payload = {"vlim": vlim, "points": len(points)}
        if steps is not None:
            payload["steps"] = steps
        self._recorder.add(motion_name, payload)

    def execute_joint_trajectory(self, points: list[list[float]], vlim: float = 1.0, motion_name: str = "custom") -> None:
        """Execute a pre-computed joint-space trajectory on the hardware.

        This is the public interface for sending a list of joint-space
        waypoints to the motors.  It is used internally by :meth:`move_j`,
        :meth:`move_l`, and :meth:`move_c`, and can be called directly
        for custom trajectory playback.

        Args:
            points: List of joint-space waypoints (each a list of floats).
            vlim: Velocity limit as a fraction of maximum.  Defaults to ``1.0``.
            motion_name: Label for telemetry recording.  Defaults to ``"custom"``.
        """
        self._run_joint_points(points, vlim=vlim, motion_name=motion_name)

    def _gripper_move(self, pos: float, vlim: float) -> None:
        idx = self._find_gripper_index()
        if idx is None:
            raise ValueError("gripper_joint is not configured")
        q = self.get_joint_positions()
        q[idx] = pos
        self.move_j(q, vlim=vlim)

    def _find_gripper_index(self) -> int | None:
        if self._cfg.gripper_joint:
            for i, j in enumerate(self._cfg.joints):
                if j.name == self._cfg.gripper_joint:
                    return i
        for i, j in enumerate(self._cfg.joints):
            if "gripper" in j.name.lower():
                return i
        return None

    def _resolve_joint_index(self, joint: int | str) -> int:
        if isinstance(joint, int):
            if 0 <= joint < len(self._cfg.joints):
                return joint
            raise IndexError(f"joint index out of range: {joint}")
        for i, jc in enumerate(self._cfg.joints):
            if jc.name == joint:
                return i
        raise KeyError(f"unknown joint name: {joint}")

    def __enter__(self) -> "Arm":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Arm(name={self._cfg.name!r}, joints={len(self._cfg.joints)}, mode={self._mode!r})"
