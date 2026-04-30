"""Standalone single-motor gripper controller.

Provides a dedicated handle for a gripper motor with its own control loop,
MIT / POS_VEL / VEL modes, and feedback polling -- independent of the Arm
facade.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from .errors import ArmError, ArmErrorCode
from .session import ModeLike, MotorBridgeSession
from .types import JointConfig
from .vendors import create_default_adapter_registry

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GripperConfig:
    """Configuration for a single-motor gripper. / 单电机夹爪配置。

    Attributes:
        name: Gripper name. / 夹爪名称。
        vendor: Motor vendor string. / 电机供应商。
        model: Motor model string. / 电机型号。
        esc_id: ESC (CAN) ID for commanding the motor. / 控制 CAN ID。
        feedback_id: CAN ID for reading feedback. / 反馈 CAN ID。
        direction: Direction multiplier (1.0 or -1.0). / 方向系数。
        zero_offset: Position offset applied after direction. / 零位偏移。
        limit_pos_min: Minimum position in radians. / 最小位置（弧度）。
        limit_pos_max: Maximum position in radians. / 最大位置（弧度）。
        limit_vel: Maximum velocity in rad/s. / 最大速度。
        limit_tau: Maximum torque in Nm. / 最大力矩。
        mit_kp: Default MIT kp gain. / 默认 MIT kp 增益。
        mit_kd: Default MIT kd gain. / 默认 MIT kd 增益。
        pos_kp: POS_VEL position Kp. / POS_VEL 位置 Kp。
        pos_ki: POS_VEL position Ki. / POS_VEL 位置 Ki。
        vel_kp: POS_VEL velocity Kp. / POS_VEL 速度 Kp。
        vel_ki: POS_VEL velocity Ki. / POS_VEL 速度 Ki。
        vlim: Default velocity limit for POS_VEL commands. / POS_VEL 默认速度限制。
    """

    name: str = "gripper"
    vendor: str = "robstride"
    model: str = "rs04"
    esc_id: int = 10
    feedback_id: int = 10
    direction: float = 1.0
    zero_offset: float = 0.0
    limit_pos_min: float = -3.1415926
    limit_pos_max: float = 3.1415926
    limit_vel: float = 2.0
    limit_tau: float = 5.0
    mit_kp: float = 18.0
    mit_kd: float = 2.0
    pos_kp: float = 50.0
    pos_ki: float = 0.5
    vel_kp: float = 10.0
    vel_ki: float = 0.1
    vlim: float = 1.0
    force_threshold: float = 0.0  # Default computed as half of limit_tau in __post_init__.

    def __post_init__(self):
        if self.force_threshold <= 0.0:
            self.force_threshold = self.limit_tau / 2.0


class Gripper:
    """Single-motor gripper controller.

    Provides MIT impedance, POS_VEL, and velocity control modes, plus a
    background control loop for real-time operation.

    Usage::

        cfg = GripperConfig(esc_id=10, feedback_id=10)
        gripper = Gripper(cfg, channel="can0")
        gripper.connect()
        gripper.enable()
        gripper.mode_pos_vel()
        gripper.pos_vel(0.5)
        gripper.close()
    """

    def __init__(self, config: GripperConfig, channel: str = "can0") -> None:
        self._cfg = config
        joint_cfg = JointConfig(
            name=config.name,
            vendor=config.vendor,
            model=config.model,
            esc_id=config.esc_id,
            feedback_id=config.feedback_id,
            direction=config.direction,
            zero_offset=config.zero_offset,
            limit_pos_min=config.limit_pos_min,
            limit_pos_max=config.limit_pos_max,
            limit_vel=config.limit_vel,
            limit_tau=config.limit_tau,
        )
        self._session = MotorBridgeSession(
            channel=channel,
            adapter_registry=create_default_adapter_registry(),
        )
        self._joint_cfg = joint_cfg
        self._motor = None
        self._mode: str = "unknown"
        self._ctrl_fn = None
        self._ctrl_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._force_threshold: float = config.force_threshold
        self._calibrated_open: float | None = None
        self._calibrated_closed: float | None = None

    @property
    def mode(self) -> str:
        """Current control mode string. / 当前控制模式。"""
        return self._mode

    def connect(self) -> None:
        """Open CAN bus and register the gripper motor. / 打开 CAN 总线并注册夹爪电机。"""
        self._session.connect()
        self._session.add_joint(self._joint_cfg)
        self._motor = self._session.joints[0].motor

    def close(self) -> None:
        """Stop control loop, disable motor, and close session. / 停止控制循环、禁用电机、关闭会话。"""
        self.stop_control_loop()
        try:
            self._session.disable_all()
        except Exception as exc:
            logger.warning("disable during close failed: %s", exc)
        self._session.close()
        self._motor = None

    def enable(self, retries: int = 10, poll_interval: float = 0.1) -> bool:
        """Enable the gripper motor.

        Args:
            retries: Maximum number of poll attempts. / 轮询最大次数。
            poll_interval: Seconds between polls. / 轮询间隔（秒）。

        Returns:
            ``True`` if the motor reached enabled state.
        """
        if self._motor is None:
            raise ArmError(ArmErrorCode.ERR_STATE, "gripper not connected")
        self._session.enable_all()
        for _ in range(retries):
            time.sleep(poll_interval)
            try:
                self._motor.request_feedback()
                st = self._motor.get_state()
                if st is not None and getattr(st, "status_code", None) == 1:
                    return True
            except Exception:
                pass
        logger.warning("gripper enable: motor did not reach enabled state")
        return False

    def disable(self, retries: int = 10, poll_interval: float = 0.1) -> bool:
        """Disable the gripper motor.

        Returns:
            ``True`` if the motor reached disabled state.
        """
        self.stop_control_loop()
        if self._motor is None:
            return True
        try:
            self._session.disable_all()
        except Exception:
            pass
        for _ in range(retries):
            time.sleep(poll_interval)
            try:
                self._motor.request_feedback()
                st = self._motor.get_state()
                if st is not None and getattr(st, "status_code", None) == 0:
                    return True
            except Exception:
                pass
        return False

    def set_zero(self, poll_max: int = 200, poll_interval: float = 0.05) -> bool:
        """Set the current position as zero.

        Disables the motor first, waits for status 0, then sets zero.

        Returns:
            ``True`` on success.
        """
        if self._motor is None:
            return False
        try:
            self._session.set_zero_joint(0)
            return True
        except Exception as exc:
            logger.warning("gripper set_zero failed: %s", exc)
            return False

    def get_state(self, request: bool = True) -> tuple[float, float, float]:
        """Return (position, velocity, torque) as scalars.

        Args:
            request: If ``True``, request fresh feedback first.
        """
        if self._motor is None:
            raise ArmError(ArmErrorCode.ERR_STATE, "gripper not connected")
        if request:
            self._request_and_poll()
        st = self._motor.get_state()
        if st is None:
            return (0.0, 0.0, 0.0)
        d = self._cfg.direction
        off = self._cfg.zero_offset
        pos = (getattr(st, "pos", 0.0) - off) * d
        vel = getattr(st, "vel", 0.0) * d
        torq = getattr(st, "torq", 0.0) * d
        return (pos, vel, torq)

    def get_position(self, request: bool = True) -> float:
        return self.get_state(request)[0]

    def get_velocity(self, request: bool = True) -> float:
        return self.get_state(request)[1]

    def get_torque(self, request: bool = True) -> float:
        return self.get_state(request)[2]

    def mode_mit(self, kp: float | None = None, kd: float | None = None, stabilize_delay: float = 0.2) -> bool:
        """Switch the gripper motor to MIT impedance mode.

        Args:
            kp: Optional kp override. / 可选 kp 覆盖。
            kd: Optional kd override. / 可选 kd 覆盖。
            stabilize_delay: Seconds to wait after mode switch. / 模式切换后等待时间（秒）。
        """
        if kp is not None:
            self._cfg.mit_kp = kp
        if kd is not None:
            self._cfg.mit_kd = kd
        try:
            self._session.ensure_mode_joint(0, ModeLike.MIT)
            self._mode = "mit"
            time.sleep(stabilize_delay)
            return True
        except Exception as exc:
            logger.warning("gripper mode_mit failed: %s", exc)
            return False

    def mode_pos_vel(self, stabilize_delay: float = 0.2) -> bool:
        """Switch to position-velocity mode with pre-configured PI gains.

        Writes velocity Kp/Ki (registers 25-26) and position Kp/Ki
        (registers 27-28) before switching the mode.
        """
        try:
            self._session.set_param(0, 25, "f32", self._cfg.vel_kp)
            self._session.set_param(0, 26, "f32", self._cfg.vel_ki)
            self._session.set_param(0, 27, "f32", self._cfg.pos_kp)
            self._session.set_param(0, 28, "f32", self._cfg.pos_ki)
            self._session.ensure_mode_joint(0, ModeLike.POS_VEL)
            self._mode = "pos_vel"
            time.sleep(stabilize_delay)
            return True
        except Exception as exc:
            logger.warning("gripper mode_pos_vel failed: %s", exc)
            return False

    def mode_vel(self, stabilize_delay: float = 0.2) -> bool:
        """Switch to pure velocity mode. / 切换到纯速度模式。"""
        try:
            self._session.ensure_mode_joint(0, ModeLike.VEL)
            self._mode = "vel"
            time.sleep(stabilize_delay)
            return True
        except Exception as exc:
            logger.warning("gripper mode_vel failed: %s", exc)
            return False

    def mit(self, pos: float, vel: float = 0.0, kp: float | None = None, kd: float | None = None, tau: float = 0.0) -> None:
        """Send MIT impedance command.

        Args:
            pos: Target position in radians. / 目标位置（弧度）。
            vel: Target velocity in rad/s. / 目标速度（rad/s）。
            kp: Impedance kp (uses default if None). / 刚度 kp。
            kd: Impedance kd (uses default if None). / 阻尼 kd。
            tau: Feedforward torque in Nm. / 前馈力矩（Nm）。
        """
        if self._motor is None:
            raise ArmError(ArmErrorCode.ERR_STATE, "gripper not connected")
        kp_val = kp if kp is not None else self._cfg.mit_kp
        kd_val = kd if kd is not None else self._cfg.mit_kd
        self._session.set_mit_joint(0, pos, vel, kp_val, kd_val, tau)
        self._request_and_poll()

    def pos_vel(self, pos: float, vlim: float | None = None) -> None:
        """Send position-with-velocity-limit command.

        After sending, checks current torque.  If the measured torque exceeds
        the force threshold the gripper stops in the direction that is causing
        the force by re-sending the current position as the target.

        Args:
            pos: Target position in radians. / 目标位置（弧度）。
            vlim: Velocity limit (uses default if None). / 速度限制。
        """
        if self._motor is None:
            raise ArmError(ArmErrorCode.ERR_STATE, "gripper not connected")
        v = vlim if vlim is not None else self._cfg.vlim
        self._session.set_pos_vel_joint(0, pos, v)
        self._request_and_poll()
        # Check torque after sending.  If the current torque exceeds the
        # force threshold, stop moving in the direction causing force by
        # commanding the current position.
        if not self._check_force(pos):
            current_pos = self.get_position(request=False)
            logger.warning(
                "gripper force exceeded threshold (%.3f Nm), holding at current position %.4f",
                self._force_threshold, current_pos,
            )
            self._session.set_pos_vel_joint(0, current_pos, v)

    def set_vel(self, vel: float) -> None:
        """Send pure velocity command. / 发送纯速度指令。"""
        if self._motor is None:
            raise ArmError(ArmErrorCode.ERR_STATE, "gripper not connected")
        self._session.set_vel_joint(0, vel)
        self._request_and_poll()

    def set_force_threshold(self, threshold: float) -> None:
        """Set the force (torque) threshold for gripper force protection.

        / 设置夹爪力保护的力矩阈值。

        Args:
            threshold: Maximum allowed torque in Nm before the gripper
                stops closing/opening.  Must be positive.
                / 允许的最大力矩（Nm），超过后夹爪停止闭合/张开。必须为正数。
        """
        if threshold <= 0.0:
            raise ValueError("force threshold must be positive")
        self._force_threshold = threshold

    def _check_force(self, target_pos: float) -> bool:
        """Check whether current torque is within the force threshold.

        Returns ``True`` if force is within limits, ``False`` if the
        measured torque exceeds the threshold and the gripper is actively
        moving toward a position that would increase the force.

        / 检查当前力矩是否在力阈值内。在阈值内返回 True，超出返回 False。

        Args:
            target_pos: The position that was just commanded, used to
                determine the direction of motion.
        """
        _, _, torque = self.get_state(request=False)
        if abs(torque) <= self._force_threshold:
            return True
        current_pos = self.get_position(request=False)
        # If torque exceeds threshold and the gripper is moving in a
        # direction that increases force, reject.
        moving_closing = target_pos < current_pos
        force_closing = torque > 0
        if moving_closing == force_closing:
            return False
        return True

    def calibrate(self) -> tuple[float, float]:
        """Interactive gripper calibration.

        Opens the gripper fully, waits for user confirmation, sets the
        zero position, then closes fully and records the closed position.
        Returns ``(open_pos, closed_pos)`` after calibration.

        / 交互式夹爪标定。完全张开夹爪，等待用户确认，设置零位，
        然后完全闭合并记录闭合位置。标定后返回 (张开位置, 闭合位置)。

        Returns:
            A tuple of ``(open_position, closed_position)`` in radians.
        """
        if self._motor is None:
            raise ArmError(ArmErrorCode.ERR_STATE, "gripper not connected")

        # Step 1: Open fully with low stiffness to find the mechanical open limit.
        logger.info("Calibration: opening gripper fully with low kp...")
        self.mode_mit(kp=2.0, kd=1.0)
        open_pos = self._cfg.limit_pos_max
        self.mit(pos=open_pos, vel=0.0, kp=2.0, kd=1.0, tau=0.0)
        time.sleep(1.5)

        # Step 2: Wait for user to confirm the gripper is at the desired open position.
        input(
            "Gripper is now fully open (low kp).  Adjust if needed, then press ENTER to set as zero position. "
            "/ 夹爪已完全张开（低 kp）。如需调整请操作，然后按回车键设为零位。"
        )
        open_pos = self.get_position()
        logger.info("Calibration: open position recorded at %.4f rad", open_pos)

        # Step 3: Set zero position.
        self.disable()
        time.sleep(0.3)
        self.set_zero()
        time.sleep(0.3)
        self.enable()
        time.sleep(0.3)
        logger.info("Calibration: zero position set.")

        # Step 4: Close fully with low stiffness to find the mechanical close limit.
        logger.info("Calibration: closing gripper fully with low kp...")
        self.mode_mit(kp=2.0, kd=1.0)
        close_pos = self._cfg.limit_pos_min
        self.mit(pos=close_pos, vel=0.0, kp=2.0, kd=1.0, tau=0.0)
        time.sleep(1.5)

        input(
            "Gripper is now fully closed (low kp).  Press ENTER to record closed position. "
            "/ 夹爪已完全闭合（低 kp）。按回车键记录闭合位置。"
        )
        closed_pos = self.get_position()
        logger.info("Calibration: closed position recorded at %.4f rad", closed_pos)

        # Store the calibrated range.
        self._calibrated_open = open_pos
        self._calibrated_closed = closed_pos
        logger.info(
            "Calibration complete: open=%.4f, closed=%.4f",
            self._calibrated_open, self._calibrated_closed,
        )
        return (self._calibrated_open, self._calibrated_closed)

    def start_control_loop(self, controller, rate: float = 100.0) -> None:
        """Start a background control loop calling *controller(self, dt)*.

        Args:
            controller: Callable ``(Gripper, float) -> None``.
            rate: Loop rate in Hz.  Default 100 Hz.
        """
        if self._ctrl_thread is not None and self._ctrl_thread.is_alive():
            return
        self._ctrl_fn = controller
        self._stop_event.clear()
        dt = 1.0 / max(rate, 1.0)

        def _loop():
            while not self._stop_event.is_set():
                t0 = time.monotonic()
                try:
                    self._ctrl_fn(self, dt)
                except Exception as exc:
                    logger.warning("gripper control loop error: %s", exc)
                elapsed = time.monotonic() - t0
                wait = dt - elapsed
                if wait > 0:
                    self._stop_event.wait(timeout=wait)

        self._ctrl_thread = threading.Thread(target=_loop, daemon=True, name="gripper-ctrl")
        self._ctrl_thread.start()

    def stop_control_loop(self) -> None:
        """Stop the background control loop. / 停止后台控制循环。"""
        self._stop_event.set()
        if self._ctrl_thread is not None:
            self._ctrl_thread.join(timeout=5.0)
            self._ctrl_thread = None

    def _request_and_poll(self) -> None:
        if self._motor is None:
            return
        try:
            self._motor.request_feedback()
        except Exception:
            pass
        try:
            self._session.request_feedback_all()
        except Exception:
            pass

    def __enter__(self) -> Gripper:
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Gripper(name={self._cfg.name!r}, vendor={self._cfg.vendor!r}, mode={self._mode!r})"
