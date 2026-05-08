from __future__ import annotations

import logging
import math
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
from .safety.outlier_filter import OutlierFilter
from .safety.supervisor import SafetySupervisor
from .session import ModeLike, MotorBridgeSession
from .telemetry.fps_monitor import FPSMonitor
from .telemetry.recorder import Recorder
from .telemetry.state_cache import StateCache
from .errors import ArmError, ArmErrorCode
from .types import ArmConfig, ArmRunState, ArmState, FaultState, JointState, PayloadConfig, Pose6D, ToolConfig
from .ipc.shared_state import SharedArmState
from .model.kinematics import Kinematics
from .vendors import MotorAdapterRegistry, create_default_adapter_registry

logger = logging.getLogger(__name__)

# Named constants for control gains and safety margins.
_DEFAULT_MIT_KP = 20.0
_DEFAULT_MIT_KD = 2.0
_SAFE_RAISE_RAD = 0.3
_ESTOP_KD_BOOST_LARGE = 3.0
_ESTOP_KD_BOOST_SMALL = 1.5


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
        self._outlier_filter = OutlierFilter(num_joints=len(config.joints))
        self._cache = StateCache(config)
        self._executor = JointMotionExecutor(config.loop_dt_s)
        self._zero = ZeroCalibrator(self._session)
        self._kin = Kinematics(config.urdf_path, config.ee_frame)
        self._registry = registry or create_default_registry()
        self._recorder = Recorder()
        self._fps = FPSMonitor()
        self._tool = ToolConfig()
        self._payload = PayloadConfig()
        self._mode = "pos_vel"
        self._damping_mode = False
        self._ctrl_thread: threading.Thread | None = None
        self._ctrl_fn = None
        self._abort_event = threading.Event()
        self._ctrl_stop_event = threading.Event()
        self._gravity_comp_enabled: bool = False
        self._drm = None  # DynamicsRobotModel, lazy-loaded on first enable

        # Shared-memory IPC (optional). / 共享内存 IPC（可选）。
        self._shm: SharedArmState | None = None
        if config.shared_memory_name is not None:
            self._shm = SharedArmState(
                name=config.shared_memory_name,
                num_joints=len(config.joints),
                create=True,
            )

    def connect(self) -> None:
        """Establish communication with the robotic arm hardware.

        Opens the underlying MotorBridge session, registers all configured
        joints, and transitions the runtime state to IDLE.

        Raises:
            RuntimeError: If the session cannot be opened or the runtime
                state does not allow a transition to IDLE.
        """
        prev_state = self._runtime.state
        self._runtime.transition(ArmRunState.IDLE)
        try:
            self._session.connect()
            for j in self._cfg.joints:
                self._session.add_joint(j)
        except Exception:
            self._runtime.force(prev_state)
            self._cache.update_run_state(prev_state)
            raise
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
    def damping_mode(self) -> bool:
        """Whether damping mode is currently active.

        In damping mode the arm is compliant (kp=0) but still damped (kd>0),
        allowing it to be moved by hand while resisting sudden shocks.
        """
        return self._damping_mode

    @property
    def control_loop_active(self) -> bool:
        t = self._ctrl_thread
        return t is not None and t.is_alive()

    @property
    def config(self) -> ArmConfig:
        """The arm configuration."""
        return self._cfg

    @property
    def kinematics(self) -> Kinematics:
        """The kinematics solver."""
        return self._kin

    def close(self) -> None:
        """Shut down the arm controller and release all resources.

        Stops any running control loop, forces the runtime state to
        DISCONNECTED, and closes the underlying MotorBridge session.
        """
        self._abort_event.set()
        self.stop_control_loop()
        self._runtime.force(ArmRunState.DISCONNECTED)
        self._session.close()
        self._cache.update_run_state(ArmRunState.DISCONNECTED)
        if self._shm is not None:
            self._shm.close()
            self._shm.unlink()
            self._shm = None

    def reconnect(self, init_delay: float = 1.0, post_setup_delay: float = 0.5) -> None:
        """Close the current connection and re-establish a fresh one.

        Convenience wrapper that calls :meth:`close` followed by
        :meth:`connect`.  Useful for recovering from transient communication
        errors.

        Args:
            init_delay: Seconds to wait after closing the previous connection
                before opening a new one.  Allows the CAN bus to settle.
            post_setup_delay: Seconds to wait after registering all motors
                before returning.  Allows firmware to initialize.
        """
        self.close()
        if init_delay > 0:
            time.sleep(init_delay)
        self.connect()
        if post_setup_delay > 0:
            time.sleep(post_setup_delay)
        if self._cfg.shared_memory_name is not None and self._shm is None:
            self._shm = SharedArmState(
                name=self._cfg.shared_memory_name,
                num_joints=len(self._cfg.joints),
                create=True,
            )

    def enable(self) -> None:
        """Enable all joint motors and transition to the ENABLED state.

        After calling this method the arm is powered and ready to accept
        motion commands.

        Raises:
            RuntimeError: If the runtime is not in a state that allows
                transition to ENABLED (e.g. currently DISCONNECTED).
        """
        self._require_connected()
        self._runtime.transition(ArmRunState.ENABLED)
        self._session.enable_all_with_poll()
        self._cache.update_run_state(ArmRunState.ENABLED)

    def disable(self) -> None:
        """Disable all joint motors and transition back to IDLE.

        If the arm is currently RUNNING it is first transitioned to ENABLED
        before being disabled.  All joint motors are de-energised.  Polls
        each motor to confirm it reached disabled state.
        """
        self._require_connected()
        if self._runtime.state == ArmRunState.DISCONNECTED:
            return
        if self._runtime.state == ArmRunState.RUNNING:
            self._runtime.transition(ArmRunState.ENABLED)
        self._runtime.transition(ArmRunState.IDLE)
        not_disabled = self._session.disable_all_with_poll()
        if not_disabled:
            logger.warning("disable: joints still enabled: %s", not_disabled)
        self._cache.update_run_state(ArmRunState.IDLE)

    def set_to_damping(self) -> None:
        """Switch the arm into damping (compliant) mode.

        Records the current joint positions as the target, sets kp=0 on all
        joints while keeping kd non-zero, and switches to MIT impedance mode.
        The arm becomes compliant (easy to push by hand) but is still damped,
        resisting sudden velocity changes.  This is useful for safe manual
        repositioning or as a transition state before returning to position
        control.

        The damping gain defaults to 2.0 per joint.  Override by calling
        :meth:`mit` directly after this method with custom kd values.
        """
        self._require_connected()
        q_now = self.get_joint_positions()
        n = len(self._cfg.joints)
        kp = [0.0] * n
        kd = [_DEFAULT_MIT_KD] * n
        vel = [0.0] * n
        tau = [0.0] * n
        self.mode_mit()
        self._session.set_mit_all(q_now, vel, kp, kd, tau)
        self._damping_mode = True
        self._cache.update_run_state(ArmRunState.ENABLED)
        logger.info("set_to_damping: arm is now in damping mode (kp=0, kd=%.1f)", _DEFAULT_MIT_KD)

    def reset_to_home(self, vlim: float = 0.5) -> None:
        """Safely return the arm to its home position with gain ramping.

        Instead of a simple move_j, this method performs a controlled return
        to home in two stages with gradual gain ramping:

        1. **Intermediate safe position**: Joint 3 (index 2) is raised slightly
           to avoid collisions, then moved to an intermediate pose.
        2. **Gain ramped move to home**: Position stiffness is ramped from a
           low value up to the default over a duration proportional to the
           maximum joint position error, using a minimum-jerk interpolation
           profile.

        If the arm is currently in damping mode, this method exits damping
        mode first by restoring MIT gains before moving.

        Args:
            vlim: Velocity limit as a fraction of maximum.  Defaults to ``0.5``
                for a conservative return speed.
        """
        self._require_connected()
        if self._runtime.state == ArmRunState.RUNNING:
            self._runtime.transition(ArmRunState.ENABLED)
            self._cache.update_run_state(ArmRunState.ENABLED)

        # Exit damping mode if active.
        if self._damping_mode:
            self._damping_mode = False

        n = len(self._cfg.joints)
        q_now = self.get_joint_positions()
        q_home = list(self._cfg.default_home) if self._cfg.default_home else [0.0] * n

        # --- Stage 1: Move to an intermediate safe position. ---
        q_intermediate = list(q_now)
        # Raise joint 3 (index 2) slightly to clear obstacles.
        if n > 2:
            q_intermediate[2] = q_now[2] + _SAFE_RAISE_RAD
            # Clamp to joint limits.
            jc = self._cfg.joints[2]
            q_intermediate[2] = max(jc.limit_pos_min, min(jc.limit_pos_max, q_intermediate[2]))

        self.move_j(q_intermediate, vlim=vlim, profile="min_jerk")

        # --- Stage 2: Gain-ramped move to home. ---
        # Compute max position error to determine ramp duration.
        q_after_intermediate = self.get_joint_positions()
        errors = [abs(q_after_intermediate[i] - q_home[i]) for i in range(n)]
        max_err = max(errors) if errors else 0.0

        # Ramp duration proportional to max error: at least 1s, at most 5s.
        ramp_duration = max(1.0, min(5.0, max_err * 2.0))
        ramp_steps = max(10, int(ramp_duration / self._cfg.loop_dt_s))

        # Interpolate from intermediate to home with min_jerk profile.
        points = interpolate_joint_linear(
            q_after_intermediate, q_home, ramp_steps, profile="min_jerk"
        )

        # Execute with gain ramping via MIT mode.
        # Start with low kp, ramp to full kp over the trajectory.
        default_kp = _DEFAULT_MIT_KP
        default_kd = _DEFAULT_MIT_KD
        self.mode_mit()
        self._runtime.transition(ArmRunState.RUNNING)
        self._cache.update_run_state(ArmRunState.RUNNING)
        try:
            for i, q_pt in enumerate(points):
                if self._abort_event.is_set():
                    break
                # Ramp fraction from 0 to 1.
                frac = i / max(ramp_steps - 1, 1)
                kp_val = default_kp * frac
                kp_list = [kp_val] * n
                kd_list = [default_kd] * n
                vel_list = [0.0] * n
                tau_list = [0.0] * n
                self._session.set_mit_all(q_pt, vel_list, kp_list, kd_list, tau_list)
                time.sleep(self._cfg.loop_dt_s)
        finally:
            if self._runtime.state == ArmRunState.RUNNING:
                self._runtime.transition(ArmRunState.ENABLED)
                self._cache.update_run_state(ArmRunState.ENABLED)

        self._recorder.add("reset_to_home", {"vlim": vlim, "ramp_duration": ramp_duration})

    def estop(self) -> None:
        """Perform an emergency stop with boosted damping before disable.

        Instead of immediately disabling motors, the e-stop first applies
        boosted damping (MIT mode with kp=0 and elevated kd) to smoothly
        arrest motion, then fully disables motors after a short settling
        delay (0.5 s).  This produces a gentler but still fast stop compared
        to instant disable.

        The runtime is forced into the FAULT state regardless.  Use
        :meth:`clear_faults` and :meth:`enable` to resume normal operation.
        """
        self._abort_event.set()
        self.stop_control_loop()

        # Apply boosted damping to arrest motion before disabling.
        n = len(self._cfg.joints)
        try:
            q_now = self.get_joint_positions()
            self.mode_mit()
            kp = [0.0] * n
            # Boosted kd: 3x normal on first 3 joints, 1.5x on the rest.
            kd = [_DEFAULT_MIT_KD * _ESTOP_KD_BOOST_LARGE if i < 3 else _DEFAULT_MIT_KD * _ESTOP_KD_BOOST_SMALL for i in range(n)]
            vel = [0.0] * n
            tau = [0.0] * n
            self._session.set_mit_all(q_now, vel, kp, kd, tau)
            time.sleep(0.5)
        except Exception:
            logger.warning("estop: boosted damping phase failed, falling back to immediate disable")

        self._session.disable_all()
        self._runtime.force(ArmRunState.FAULT)
        self._cache.update_run_state(ArmRunState.FAULT)
        self._damping_mode = False

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
        self._require_connected()
        self._session.request_feedback_all()
        for i, h in enumerate(self._session.joints):
            raw = h.motor.get_state()
            if raw is None:
                logger.warning("joint %d (%s) returned None state, skipping", i, h.config.name)
                continue
            if h.config.direction == 0:
                logger.error("joint %s has direction=0, skipping", h.config.name)
                continue
            q = (raw.pos - h.config.zero_offset) / h.config.direction
            # 过滤异常值 / Filter outlier values
            filt_q, filt_vel, filt_torq = self._outlier_filter.filter_joint(i, q, raw.vel, raw.torq)
            if filt_q is not q or filt_vel is not raw.vel or filt_torq is not raw.torq:
                logger.warning(
                    "outlier filtered on joint %d (%s): pos=%s->%s vel=%s->%s torq=%s->%s",
                    i,
                    h.config.name,
                    f"{q:.4f}", f"{filt_q:.4f}" if filt_q is not None else "FILTERED",
                    f"{raw.vel:.4f}", f"{filt_vel:.4f}" if filt_vel is not None else "FILTERED",
                    f"{raw.torq:.4f}", f"{filt_torq:.4f}" if filt_torq is not None else "FILTERED",
                )
            self._cache.update_joint(
                i,
                JointState(
                    name=h.config.name,
                    pos=filt_q,
                    vel=filt_vel,
                    torq=filt_torq,
                    status_code=raw.status_code,
                    t_mos=raw.t_mos,
                    t_rotor=raw.t_rotor,
                ),
            )
        self._fps.tick("feedback")
        snapshot = self._cache.snapshot()

        # Publish to shared memory if configured. / 如果已配置，发布到共享内存。
        if self._shm is not None and self._shm.active:
            self._shm.write(snapshot.positions(), snapshot.velocities(), snapshot.torques(), snapshot.status_codes())

        return snapshot

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
        return st.positions()

    def get_positions(self, request: bool = True) -> list[float]:
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
        st = self.refresh_state() if request else self.get_state()
        return st.positions()

    def get_velocities(self, request: bool = True) -> list[float]:
        """Return joint velocities, optionally refreshing from hardware.

        Args:
            request: If ``True`` (default), fresh hardware feedback is
                requested before returning velocities.

        Returns:
            A list of joint velocity values in rad/s.  Joints with no
            reading default to ``0.0``.
        """
        st = self.refresh_state() if request else self.get_state()
        return st.velocities()

    def get_torques(self, request: bool = True) -> list[float]:
        """Return joint torques, optionally refreshing from hardware.

        Args:
            request: If ``True`` (default), fresh hardware feedback is
                requested before returning torques.

        Returns:
            A list of joint torque values in Nm.  Joints with no reading
            default to ``0.0``.
        """
        st = self.refresh_state() if request else self.get_state()
        return st.torques()

    def get_state_vectors(self) -> tuple[list[float], list[float], list[float]]:
        """Return positions, velocities, and torques in a single call.

        Performs a single hardware refresh and extracts all three vectors
        from the same state snapshot, avoiding redundant bus round-trips.

        Returns:
            A 3-tuple ``(positions, velocities, torques)`` where each
            element is a list of floats with one entry per configured joint.
        """
        st = self.refresh_state()
        return st.positions(), st.velocities(), st.torques()

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
        return self._apply_tool_offset(base)

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
        self._require_connected()
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

        Writes per-joint velocity and position PI gains to motor registers
        (25-28) before switching mode, matching the hardware's expected
        register configuration.

        Returns:
            ``True`` on success.
        """
        for i, jc in enumerate(self._cfg.joints):
            try:
                self._session.write_pi_gains(i, jc.vel_kp, jc.vel_ki, jc.pos_kp, jc.pos_ki)
            except ArmError:
                logger.debug("PI gain write for joint %d skipped (vendor may not support)", i)
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

    def enable_gravity_comp(self) -> None:
        """Enable gravity compensation as a feedforward torque offset.

        When enabled, the ``mit()`` command automatically computes the
        generalized gravity vector for the current configuration and adds
        it to the feedforward torque, compensating for the robot's own
        weight.

        Requires a valid URDF path in the arm configuration.  If no URDF
        is configured or Pinocchio is unavailable the method logs a
        warning and returns without enabling.

        / 启用重力补偿作为前馈力矩偏移。启用后，MIT 指令会自动计算当前构型的
        广义重力向量并将其叠加到前馈力矩上，补偿机器人自身重量。
        """
        if self._drm is None:
            if self._cfg.urdf_path is None:
                logger.warning("gravity compensation requires urdf_path in ArmConfig; not enabled")
                return
            from .dynamics.robot_model import load_dynamics_robot_model
            self._drm = load_dynamics_robot_model(self._cfg.urdf_path)
        if not self._drm.has_pinocchio:
            logger.warning("gravity compensation requires Pinocchio with a valid URDF; not enabled")
            return
        self._gravity_comp_enabled = True
        logger.info("gravity compensation enabled")

    def disable_gravity_comp(self) -> None:
        """Disable gravity compensation feedforward offset.

        / 禁用重力补偿前馈偏移。
        """
        self._gravity_comp_enabled = False
        logger.info("gravity compensation disabled")

    def _gravity_torque(self) -> list[float]:
        """Compute the gravity compensation torque for the current configuration.

        Returns a list of floats (one per joint), or a zero list if
        gravity compensation is disabled or unavailable.

        / 计算当前构型的重力补偿力矩。如果重力补偿被禁用或不可用，
        返回零列表。
        """
        if not self._gravity_comp_enabled or self._drm is None:
            return [0.0] * len(self._cfg.joints)
        if not self._drm.has_pinocchio:
            return [0.0] * len(self._cfg.joints)
        try:
            from .dynamics.inverse_dynamics import compute_generalized_gravity
            q = self.get_joint_positions()
            g = compute_generalized_gravity(self._drm, q=q)
            tau_g = list(g.flatten() if hasattr(g, "flatten") else g)
            # Trim or pad to match number of joints.
            n = len(self._cfg.joints)
            if len(tau_g) >= n:
                return [float(v) for v in tau_g[:n]]
            return [float(v) for v in tau_g] + [0.0] * (n - len(tau_g))
        except Exception as exc:
            logger.warning("gravity torque computation failed: %s", exc)
            return [0.0] * len(self._cfg.joints)

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
        self._require_connected()
        n = len(self._cfg.joints)
        if len(pos) != n:
            raise ValueError(f"pos length {len(pos)} does not match num_joints {n}")
        vel = vel if vel is not None else [0.0] * n
        kp = kp if kp is not None else [0.0] * n
        kd = kd if kd is not None else [0.0] * n
        tau = tau if tau is not None else [0.0] * n
        # Add gravity compensation feedforward torque when enabled.
        if self._gravity_comp_enabled:
            tau_grav = self._gravity_torque()
            tau = [t + tg for t, tg in zip(tau, tau_grav)]
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
        self._require_connected()
        n = len(self._cfg.joints)
        if len(pos) != n:
            raise ValueError(f"pos length {len(pos)} does not match num_joints {n}")
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
        self._require_connected()
        n = len(self._cfg.joints)
        if len(vel) != n:
            raise ValueError(f"vel length {len(vel)} does not match num_joints {n}")
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

    def _ik_waypoints(self, poses: list[Pose6D], q_seed: list[float]) -> list[list[float]]:
        """Solve IK for a sequence of Cartesian poses, seed-chaining each result."""
        points: list[list[float]] = []
        q = list(q_seed)
        for pose in poses:
            q = self._kin.inverse(pose, q)
            points.append(self._safety.clamp_joint_targets(q))
        return points

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
        self._require_connected()
        q_now = self.get_joint_positions()
        start = self._apply_tool_offset(self._kin.forward(q_now))
        profile_name = profile or self._cfg.motion_profile
        dist = ((target.x - start.x) ** 2 + (target.y - start.y) ** 2 + (target.z - start.z) ** 2) ** 0.5
        steps = max(2, int(dist / max(step_m, 1e-4)) + 1)
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
            joint_points = self._ik_waypoints(poses, q_now)
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
        self._require_connected()
        q_now = self.get_joint_positions()
        start = self._apply_tool_offset(self._kin.forward(q_now))
        poses = interpolate_pose_circular(
            start,
            target,
            ArcSpec(center_x=center_x, center_y=center_y, normal_z=normal_z),
            steps,
            profile=profile or self._cfg.motion_profile,
        )
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
            joint_points = self._ik_waypoints(poses, q_now)
        self._run_joint_points(joint_points, vlim=vlim, motion_name="move_c")

    def read_param(self, joint_index: int, param_id: int, param_type: str | None = None, timeout_ms: int = 1000) -> int | float:
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
        if not (0 <= joint_index < len(self._cfg.joints)):
            raise IndexError(f"joint_index {joint_index} out of range [0, {len(self._cfg.joints)})")
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
        if not (0 <= joint_index < len(self._cfg.joints)):
            raise IndexError(f"joint_index {joint_index} out of range [0, {len(self._cfg.joints)})")
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

    def get_fps(self, channel: str = "feedback") -> float:
        """Return the current FPS for a named telemetry channel.

        返回指定遥测通道的当前帧率。

        The default channel ``"feedback"`` is ticked on every call to
        :meth:`refresh_state`.  The ``"control_loop"`` channel is ticked at
        the start of every control-loop iteration.

        默认通道 ``"feedback"`` 在每次调用 :meth:`refresh_state` 时记录一帧。
        ``"control_loop"`` 通道在每次控制循环迭代开始时记录一帧。

        Args:
            channel: Name of the telemetry channel.  Defaults to
                ``"feedback"``.
                遥测通道名称，默认为 ``"feedback"``。

        Returns:
            Frames-per-second for the channel, or ``0.0`` if insufficient
            data is available.
            该通道的每秒帧数；数据不足时返回 ``0.0``。
        """
        return self._fps.fps(channel)

    def get_all_fps(self) -> dict[str, float]:
        """Return FPS values for all tracked channels.

        返回所有已追踪通道的帧率。

        Returns:
            A dict mapping channel name to current FPS.
            键为通道名称、值为当前 FPS 的字典。
        """
        return self._fps.all_fps()

    def clear_faults(self) -> None:
        """Clear fault conditions on all joints.

        Sends a clear-fault command to every motor and resets the abort event
        so that motion commands can be issued again.  After clearing you
        typically need to call :meth:`enable` to resume operation.
        """
        self._abort_event.clear()
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
        self._ctrl_stop_event.clear()
        dt = self._cfg.loop_dt_s if rate_hz is None or rate_hz <= 0 else 1.0 / rate_hz

        def _loop():
            next_time = time.perf_counter()
            while not self._ctrl_stop_event.is_set():
                self._fps.tick("control_loop")
                if self._ctrl_fn is not None:
                    try:
                        self._ctrl_fn(self, dt)
                    except Exception:
                        logger.exception("control loop callback raised exception")
                next_time += dt
                dt_sleep = next_time - time.perf_counter()
                if dt_sleep > 0:
                    time.sleep(dt_sleep)
                else:
                    next_time = time.perf_counter()

        self._ctrl_thread = threading.Thread(target=_loop, daemon=True)
        self._ctrl_thread.start()

    def stop_control_loop(self) -> None:
        """Stop the background control loop started by :meth:`start_control_loop`.

        Signals the loop thread to exit and waits up to one second for it
        to join.  Safe to call even when no loop is running.
        """
        self._ctrl_stop_event.set()
        if self._ctrl_thread is not None:
            if threading.current_thread() is not self._ctrl_thread:
                self._ctrl_thread.join(timeout=1.0)
                if self._ctrl_thread.is_alive():
                    logger.warning("control loop thread did not join within timeout; possible zombie thread")
            self._ctrl_thread = None

    def _run_joint_target(self, q_target: list[float], vlim: float, profile: str = "linear") -> None:
        vlim = self._safety.validate_velocity_limit(vlim)
        q_now = self.get_joint_positions()
        steps = estimate_steps(q_now, q_target)
        points = interpolate_joint_linear(q_now, q_target, steps, profile=profile)
        self._run_joint_points(points, vlim=vlim, motion_name="move_j", steps=steps)

    def _run_joint_points(self, points: list[list[float]], vlim: float, motion_name: str, steps: int | None = None) -> None:
        if self._abort_event.is_set():
            raise ArmError(ArmErrorCode.ERR_STATE, "abort event is set, clear faults first")
        self._abort_event.clear()
        self._runtime.transition(ArmRunState.RUNNING)
        saved_mode = self._session._current_mode
        self._session.ensure_mode_all(ModeLike.POS_VEL)
        self._cache.update_run_state(ArmRunState.RUNNING)
        try:
            self._executor.run(points, self._session.set_pos_vel_all, vlim, abort_event=self._abort_event)
        finally:
            if self._runtime.state == ArmRunState.RUNNING:
                self._runtime.transition(ArmRunState.ENABLED)
                self._cache.update_run_state(ArmRunState.ENABLED)
            if saved_mode is not None and saved_mode != ModeLike.POS_VEL:
                try:
                    self._session.ensure_mode_all(saved_mode)
                except Exception as exc:
                    logger.warning("failed to restore mode after motion: %s", exc)
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
            raise ValueError(
                f"gripper_joint '{self._cfg.gripper_joint}' not found in configured joints: "
                f"{[j.name for j in self._cfg.joints]}"
            )
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

    def _require_connected(self) -> None:
        """Raise ArmError if the arm is in the DISCONNECTED state."""
        if self._runtime.state == ArmRunState.DISCONNECTED:
            raise ArmError(ArmErrorCode.ERR_STATE, "arm is not connected")

    @staticmethod
    def _rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> list[list[float]]:
        """Convert roll-pitch-yaw to a 3x3 rotation matrix (ZYX convention)."""
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]

    @staticmethod
    def _rotation_matrix_to_rpy(R: list[list[float]]) -> tuple[float, float, float]:
        """Extract roll-pitch-yaw from a 3x3 rotation matrix (ZYX convention)."""
        sy = -R[2][0]
        sy = max(-1.0, min(1.0, sy))
        pitch = math.asin(sy)
        if abs(math.cos(pitch)) > 1e-6:
            roll = math.atan2(R[2][1], R[2][2])
            yaw = math.atan2(R[1][0], R[0][0])
        else:
            # Gimbal lock: roll and yaw are coupled.
            # pitch = +pi/2: roll - yaw = atan2(R[0][1], R[1][1])
            # pitch = -pi/2: roll + yaw = atan2(-R[0][1], R[1][1])
            yaw = 0.0
            if sy > 0:
                roll = math.atan2(R[0][1], R[1][1])
            else:
                roll = math.atan2(-R[0][1], R[1][1])
        return roll, pitch, yaw

    def _apply_tool_offset(self, pose: Pose6D) -> Pose6D:
        """Apply tool offset as a proper SE(3) frame composition.

        Computes T_base_tool = T_base_flange * T_flange_tool using 4x4
        homogeneous transform multiplication instead of Euler angle addition.
        """
        t = self._tool
        if t.x == 0.0 and t.y == 0.0 and t.z == 0.0 and t.roll == 0.0 and t.pitch == 0.0 and t.yaw == 0.0:
            return pose

        R_flange = self._rpy_to_rotation_matrix(pose.roll, pose.pitch, pose.yaw)
        p_flange = [pose.x, pose.y, pose.z]

        R_tool = self._rpy_to_rotation_matrix(t.roll, t.pitch, t.yaw)
        p_tool = [t.x, t.y, t.z]

        R_result = [
            [
                R_flange[i][0] * R_tool[0][j] + R_flange[i][1] * R_tool[1][j] + R_flange[i][2] * R_tool[2][j]
                for j in range(3)
            ]
            for i in range(3)
        ]
        p_result = [
            R_flange[i][0] * p_tool[0] + R_flange[i][1] * p_tool[1] + R_flange[i][2] * p_tool[2] + p_flange[i]
            for i in range(3)
        ]

        roll, pitch, yaw = self._rotation_matrix_to_rpy(R_result)
        return Pose6D(x=p_result[0], y=p_result[1], z=p_result[2], roll=roll, pitch=pitch, yaw=yaw)

    def __enter__(self) -> "Arm":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Arm(name={self._cfg.name!r}, joints={len(self._cfg.joints)}, mode={self._mode!r})"
