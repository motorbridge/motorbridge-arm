from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import IntEnum

from .errors import ArmError, ArmErrorCode
from .types import JointConfig
from .vendors import MotorAdapterRegistry, create_default_adapter_registry

logger = logging.getLogger(__name__)


class ModeLike(IntEnum):
    MIT = 0
    POS_VEL = 1
    VEL = 2
    FORCE_POS = 3


@dataclass(slots=True)
class JointHandle:
    config: JointConfig
    motor: object


class MotorBridgeSession:
    def __init__(self, channel: str, adapter_registry: MotorAdapterRegistry | None = None) -> None:
        self._channel = channel
        self._controller: object | None = None
        self._joints: list[JointHandle] = []
        self._adapter_registry = adapter_registry or create_default_adapter_registry()
        self._op_retry_count = 3
        self._op_retry_delay_s = 0.01
        self._lock = threading.Lock()

    def _check_index(self, index: int) -> None:
        if index < 0 or index >= len(self._joints):
            raise ArmError(ArmErrorCode.ERR_CONFIG, f"joint index {index} out of range [0, {len(self._joints)})")

    @property
    def joints(self) -> list[JointHandle]:
        return self._joints

    def connect(self) -> None:
        with self._lock:
            if self._controller is None:
                try:
                    from motorbridge import Controller
                except Exception as exc:
                    raise ArmError(
                        ArmErrorCode.ERR_BUS,
                        "motorbridge package not available; install dependency before hardware run",
                    ) from exc
                self._controller = Controller(self._channel)

    def close(self) -> None:
        with self._lock:
            if self._controller is not None:
                try:
                    self.disable_all()
                except Exception as exc:
                    logger.warning("disable_all() during close failed: %s", exc)
                try:
                    self._controller.shutdown()
                except Exception as exc:
                    logger.warning("controller.shutdown() during close failed: %s", exc)
                try:
                    self._controller.close()
                except Exception as exc:
                    logger.warning("controller.close() during close failed: %s", exc)
                self._controller = None
            self._joints.clear()

    def __enter__(self) -> "MotorBridgeSession":
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def add_joint(self, joint: JointConfig) -> None:
        if self._controller is None:
            raise ArmError(ArmErrorCode.ERR_STATE, "controller not connected")
        m = self._adapter_registry.create_motor(
            controller=self._controller,
            vendor=joint.vendor,
            esc_id=joint.esc_id,
            feedback_id=joint.feedback_id,
            model=joint.model,
        )
        self._joints.append(JointHandle(config=joint, motor=m))

    def enable_all(self) -> None:
        if self._controller is None:
            raise ArmError(ArmErrorCode.ERR_STATE, "controller not connected")
        self._controller.enable_all()

    def enable_all_with_poll(self, timeout_ms: int = 5000, poll_interval_ms: int = 100) -> list[str]:
        """Enable all joints and poll until each reports status_code == 1 (enabled).

        Returns a list of joint names that did NOT reach enabled state within
        the timeout.
        """
        if self._controller is None:
            raise ArmError(ArmErrorCode.ERR_STATE, "controller not connected")
        self._controller.enable_all()
        deadline = time.monotonic() + timeout_ms / 1000.0
        not_ready = {h.config.name for h in self._joints}
        while not_ready and time.monotonic() < deadline:
            time.sleep(poll_interval_ms / 1000.0)
            for h in self._joints:
                if h.config.name in not_ready:
                    try:
                        h.motor.request_feedback()
                        st = h.motor.get_state()
                        if st is not None and getattr(st, "status_code", None) == 1:
                            not_ready.discard(h.config.name)
                    except Exception:
                        pass
        if not_ready:
            logger.warning("enable_all_with_poll: joints not ready: %s", not_ready)
        return list(not_ready)

    def disable_all(self) -> None:
        if self._controller is None:
            return
        self._controller.disable_all()

    def disable_all_with_poll(self, timeout_ms: int = 5000, poll_interval_ms: int = 100) -> list[str]:
        """Disable all joints and poll until each reports status_code == 0 (disabled).

        Returns a list of joint names that did NOT reach disabled state within
        the timeout.
        """
        if self._controller is None:
            return []
        self._controller.disable_all()
        deadline = time.monotonic() + timeout_ms / 1000.0
        not_ready = {h.config.name for h in self._joints}
        while not_ready and time.monotonic() < deadline:
            time.sleep(poll_interval_ms / 1000.0)
            for h in self._joints:
                if h.config.name in not_ready:
                    try:
                        h.motor.request_feedback()
                        st = h.motor.get_state()
                        if st is not None and getattr(st, "status_code", None) == 0:
                            not_ready.discard(h.config.name)
                    except Exception:
                        pass
        if not_ready:
            logger.warning("disable_all_with_poll: joints still enabled: %s", not_ready)
        return list(not_ready)

    def ensure_mode_all(self, mode: int, timeout_ms: int = 1000) -> None:
        for h in self._joints:
            self._retry_call(
                lambda hh=h: hh.motor.ensure_mode(mode, timeout_ms),
                op_name=f"ensure_mode({h.config.name})",
                err_code=ArmErrorCode.ERR_MODE,
            )

    def ensure_mode_joint(self, index: int, mode: int, timeout_ms: int = 1000) -> None:
        self._check_index(index)
        h = self._joints[index]
        self._retry_call(
            lambda: h.motor.ensure_mode(mode, timeout_ms),
            op_name=f"ensure_mode({h.config.name})",
            err_code=ArmErrorCode.ERR_MODE,
        )

    def set_pos_vel_all(self, q: list[float], vlim: float) -> None:
        if len(q) != len(self._joints):
            raise ArmError(ArmErrorCode.ERR_CONFIG, "q length mismatch")
        for target, h in zip(q, self._joints):
            motor_target = target * h.config.direction + h.config.zero_offset
            h.motor.send_pos_vel(float(motor_target), float(vlim))

    def set_vel_all(self, vel: list[float]) -> None:
        if len(vel) != len(self._joints):
            raise ArmError(ArmErrorCode.ERR_CONFIG, "vel length mismatch")
        for target, h in zip(vel, self._joints):
            motor_vel = target * h.config.direction
            h.motor.send_vel(float(motor_vel))

    def set_mit_all(self, pos: list[float], vel: list[float], kp: list[float], kd: list[float], tau: list[float] | None = None) -> None:
        n = len(self._joints)
        if not (len(pos) == len(vel) == len(kp) == len(kd) == n):
            raise ArmError(ArmErrorCode.ERR_CONFIG, "mit vector length mismatch")
        if tau is None:
            tau = [0.0] * n
        if len(tau) != n:
            raise ArmError(ArmErrorCode.ERR_CONFIG, "tau length mismatch")
        for i, h in enumerate(self._joints):
            motor_pos = pos[i] * h.config.direction + h.config.zero_offset
            motor_vel = vel[i] * h.config.direction
            motor_tau = tau[i] * h.config.direction
            h.motor.send_mit(float(motor_pos), float(motor_vel), float(kp[i]), float(kd[i]), float(motor_tau))

    def set_pos_vel_joint(self, index: int, q_target: float, vlim: float) -> None:
        self._check_index(index)
        h = self._joints[index]
        motor_target = q_target * h.config.direction + h.config.zero_offset
        self._retry_call(
            lambda: h.motor.send_pos_vel(float(motor_target), float(vlim)),
            op_name=f"send_pos_vel({h.config.name})",
            err_code=ArmErrorCode.ERR_TIMEOUT,
        )

    def set_vel_joint(self, index: int, vel: float) -> None:
        self._check_index(index)
        h = self._joints[index]
        motor_vel = vel * h.config.direction
        self._retry_call(
            lambda: h.motor.send_vel(float(motor_vel)),
            op_name=f"send_vel({h.config.name})",
            err_code=ArmErrorCode.ERR_TIMEOUT,
        )

    def set_mit_joint(self, index: int, pos: float, vel: float, kp: float, kd: float, tau: float = 0.0) -> None:
        self._check_index(index)
        h = self._joints[index]
        motor_pos = pos * h.config.direction + h.config.zero_offset
        motor_vel = vel * h.config.direction
        motor_tau = tau * h.config.direction
        self._retry_call(
            lambda: h.motor.send_mit(float(motor_pos), float(motor_vel), float(kp), float(kd), float(motor_tau)),
            op_name=f"send_mit({h.config.name})",
            err_code=ArmErrorCode.ERR_TIMEOUT,
        )

    def request_feedback_all(self) -> None:
        for h in self._joints:
            self._retry_call(
                lambda hh=h: hh.motor.request_feedback(),
                op_name=f"request_feedback({h.config.name})",
                err_code=ArmErrorCode.ERR_TIMEOUT,
            )

    def set_zero_joint(self, index: int) -> None:
        self._check_index(index)
        h = self._joints[index]
        self._retry_call(
            lambda: h.motor.disable(),
            op_name=f"disable({h.config.name})",
            err_code=ArmErrorCode.ERR_MODE,
        )
        self._retry_call(
            lambda: h.motor.set_zero_position(),
            op_name=f"set_zero({h.config.name})",
            err_code=ArmErrorCode.ERR_TIMEOUT,
        )

    def set_zero_all(self) -> None:
        for i in range(len(self._joints)):
            self.set_zero_joint(i)

    def clear_fault_all(self) -> None:
        for h in self._joints:
            self._retry_call(
                lambda hh=h: hh.motor.clear_error(),
                op_name=f"clear_error({h.config.name})",
                err_code=ArmErrorCode.ERR_MODE,
            )

    def set_param(self, index: int, param_id: int, param_type: str, value: int | float) -> None:
        self._check_index(index)
        h = self._joints[index]
        vendor = h.config.vendor.lower()
        method_name = self._adapter_registry.get_write_method(vendor, param_type)
        fn = getattr(h.motor, method_name, None)
        if fn is None:
            raise ArmError(ArmErrorCode.ERR_UNSUPPORTED, f"motor missing method: {method_name}")
        if param_type == "f32":
            cast_val = float(value)
        else:
            cast_val = int(value)
        self._retry_call(lambda: fn(param_id, cast_val), method_name)

    def get_param(self, index: int, param_id: int, param_type: str, timeout_ms: int = 1000) -> int | float:
        self._check_index(index)
        h = self._joints[index]
        vendor = h.config.vendor.lower()
        method_name = self._adapter_registry.get_read_method(vendor, param_type)
        fn = getattr(h.motor, method_name, None)
        if fn is None:
            raise ArmError(ArmErrorCode.ERR_UNSUPPORTED, f"motor missing method: {method_name}")
        result = self._retry_call(lambda: fn(param_id, timeout_ms), method_name)
        return float(result) if param_type == "f32" else int(result)

    def _retry_call(self, fn, op_name: str, err_code: ArmErrorCode = ArmErrorCode.ERR_TIMEOUT):
        last_exc: Exception | None = None
        for attempt in range(self._op_retry_count):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if attempt < self._op_retry_count - 1:
                    time.sleep(self._op_retry_delay_s)
        msg = f"{op_name} failed after {self._op_retry_count} retries"
        if last_exc is not None:
            msg = f"{msg}: {last_exc}"
        raise ArmError(err_code, msg)
