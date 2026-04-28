from __future__ import annotations

from dataclasses import dataclass

from ..errors import ArmError, ArmErrorCode


@dataclass(frozen=True, slots=True)
class MotorAdapter:
    vendor: str
    add_motor_method: str


class MotorAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, MotorAdapter] = {}

    def register(self, adapter: MotorAdapter) -> None:
        self._adapters[adapter.vendor.lower()] = adapter

    def get(self, vendor: str) -> MotorAdapter:
        key = vendor.lower()
        adapter = self._adapters.get(key)
        if adapter is None:
            raise ArmError(ArmErrorCode.ERR_UNSUPPORTED, f"unsupported vendor: {vendor}")
        return adapter

    def create_motor(self, controller: object, vendor: str, esc_id: int, feedback_id: int, model: str) -> object:
        adapter = self.get(vendor)
        fn = getattr(controller, adapter.add_motor_method, None)
        if fn is None:
            raise ArmError(
                ArmErrorCode.ERR_UNSUPPORTED,
                f"controller missing method for vendor={vendor}: {adapter.add_motor_method}",
            )
        return fn(esc_id, feedback_id, model)


def create_default_adapter_registry() -> MotorAdapterRegistry:
    reg = MotorAdapterRegistry()
    reg.register(MotorAdapter(vendor="damiao", add_motor_method="add_damiao_motor"))
    reg.register(MotorAdapter(vendor="robstride", add_motor_method="add_robstride_motor"))
    reg.register(MotorAdapter(vendor="myactuator", add_motor_method="add_myactuator_motor"))
    reg.register(MotorAdapter(vendor="hightorque", add_motor_method="add_hightorque_motor"))
    reg.register(MotorAdapter(vendor="hexfellow", add_motor_method="add_hexfellow_motor"))
    return reg
