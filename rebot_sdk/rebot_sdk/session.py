from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum

from .errors import ArmError, ArmErrorCode
from .types import JointConfig
from .vendors import MotorAdapterRegistry, create_default_adapter_registry


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

    @property
    def joints(self) -> list[JointHandle]:
        return self._joints

    def connect(self) -> None:
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
        if self._controller is not None:
            try:
                self.disable_all()
            except Exception:
                pass
            try:
                self._controller.shutdown()
            except Exception:
                pass
            self._controller.close()
            self._controller = None
        self._joints.clear()

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

    def disable_all(self) -> None:
        if self._controller is None:
            return
        self._controller.disable_all()

    def ensure_mode_all(self, mode: int, timeout_ms: int = 1000) -> None:
        for h in self._joints:
            self._retry_call(
                lambda hh=h: hh.motor.ensure_mode(mode, timeout_ms),
                op_name=f"ensure_mode({h.config.name})",
                err_code=ArmErrorCode.ERR_MODE,
            )

    def set_pos_vel_all(self, q: list[float], vlim: float) -> None:
        if len(q) != len(self._joints):
            raise ArmError(ArmErrorCode.ERR_CONFIG, "q length mismatch")
        for target, h in zip(q, self._joints):
            motor_target = target * h.config.direction + h.config.zero_offset
            h.motor.send_pos_vel(float(motor_target), float(vlim))

    def request_feedback_all(self) -> None:
        for h in self._joints:
            self._retry_call(
                lambda hh=h: hh.motor.request_feedback(),
                op_name=f"request_feedback({h.config.name})",
                err_code=ArmErrorCode.ERR_TIMEOUT,
            )

    def set_zero_joint(self, index: int) -> None:
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

    def set_param(self, index: int, param_id: int, param_type: str, value: int | float) -> None:
        h = self._joints[index]
        vendor = h.config.vendor.lower()
        if vendor == "robstride":
            if param_type == "i8":
                self._retry_call(lambda: h.motor.robstride_write_param_i8(param_id, int(value)), "robstride_write_param_i8")
            elif param_type == "u8":
                self._retry_call(lambda: h.motor.robstride_write_param_u8(param_id, int(value)), "robstride_write_param_u8")
            elif param_type == "u16":
                self._retry_call(lambda: h.motor.robstride_write_param_u16(param_id, int(value)), "robstride_write_param_u16")
            elif param_type == "u32":
                self._retry_call(lambda: h.motor.robstride_write_param_u32(param_id, int(value)), "robstride_write_param_u32")
            elif param_type == "f32":
                self._retry_call(lambda: h.motor.robstride_write_param_f32(param_id, float(value)), "robstride_write_param_f32")
            else:
                raise ArmError(ArmErrorCode.ERR_UNSUPPORTED, f"unsupported param type: {param_type}")
            return
        if vendor == "damiao":
            if param_type == "u32":
                self._retry_call(lambda: h.motor.damiao_write_param_u32(param_id, int(value)), "damiao_write_param_u32")
            elif param_type == "f32":
                self._retry_call(lambda: h.motor.damiao_write_param_f32(param_id, float(value)), "damiao_write_param_f32")
            else:
                raise ArmError(ArmErrorCode.ERR_UNSUPPORTED, f"damiao unsupported param type: {param_type}")
            return
        raise ArmError(ArmErrorCode.ERR_UNSUPPORTED, f"param rw not implemented for vendor={vendor}")

    def get_param(self, index: int, param_id: int, param_type: str, timeout_ms: int = 1000) -> int | float:
        h = self._joints[index]
        vendor = h.config.vendor.lower()
        if vendor == "robstride":
            if param_type == "i8":
                return int(self._retry_call(lambda: h.motor.robstride_get_param_i8(param_id, timeout_ms), "robstride_get_param_i8"))
            if param_type == "u8":
                return int(self._retry_call(lambda: h.motor.robstride_get_param_u8(param_id, timeout_ms), "robstride_get_param_u8"))
            if param_type == "u16":
                return int(self._retry_call(lambda: h.motor.robstride_get_param_u16(param_id, timeout_ms), "robstride_get_param_u16"))
            if param_type == "u32":
                return int(self._retry_call(lambda: h.motor.robstride_get_param_u32(param_id, timeout_ms), "robstride_get_param_u32"))
            if param_type == "f32":
                return float(self._retry_call(lambda: h.motor.robstride_get_param_f32(param_id, timeout_ms), "robstride_get_param_f32"))
        if vendor == "damiao":
            if param_type == "u32":
                return int(self._retry_call(lambda: h.motor.damiao_get_param_u32(param_id, timeout_ms), "damiao_get_param_u32"))
            if param_type == "f32":
                return float(self._retry_call(lambda: h.motor.damiao_get_param_f32(param_id, timeout_ms), "damiao_get_param_f32"))
        raise ArmError(ArmErrorCode.ERR_UNSUPPORTED, f"unsupported get_param type={param_type} vendor={vendor}")

    def _retry_call(self, fn, op_name: str, err_code: ArmErrorCode = ArmErrorCode.ERR_TIMEOUT):
        last_exc: Exception | None = None
        for _ in range(self._op_retry_count):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                time.sleep(self._op_retry_delay_s)
        msg = f"{op_name} failed after {self._op_retry_count} retries"
        if last_exc is not None:
            msg = f"{msg}: {last_exc}"
        raise ArmError(err_code, msg)
