from __future__ import annotations

from dataclasses import dataclass

from ..errors import ArmError, ArmErrorCode


@dataclass(frozen=True, slots=True)
class MotorAdapter:
    vendor: str
    add_motor_method: str
    write_param_methods: tuple[str, ...] = ()
    read_param_methods: tuple[str, ...] = ()


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

    def get_write_method(self, vendor: str, param_type: str) -> str:
        """Look up the parameter write method name for a vendor and type."""
        adapter = self.get(vendor)
        type_prefix = f"{vendor}_write_param_{param_type}"
        for m in adapter.write_param_methods:
            if m == type_prefix:
                return m
        raise ArmError(ArmErrorCode.ERR_UNSUPPORTED, f"unsupported write param type={param_type} for vendor={vendor}")

    def get_read_method(self, vendor: str, param_type: str) -> str:
        """Look up the parameter read method name for a vendor and type."""
        adapter = self.get(vendor)
        type_prefix = f"{vendor}_get_param_{param_type}"
        for m in adapter.read_param_methods:
            if m == type_prefix:
                return m
        raise ArmError(ArmErrorCode.ERR_UNSUPPORTED, f"unsupported read param type={param_type} for vendor={vendor}")

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
    reg.register(MotorAdapter(vendor="damiao", add_motor_method="add_damiao_motor", write_param_methods=("damiao_write_param_u32", "damiao_write_param_f32"), read_param_methods=("damiao_get_param_u32", "damiao_get_param_f32")))
    reg.register(MotorAdapter(vendor="robstride", add_motor_method="add_robstride_motor", write_param_methods=("robstride_write_param_i8", "robstride_write_param_u8", "robstride_write_param_u16", "robstride_write_param_u32", "robstride_write_param_f32"), read_param_methods=("robstride_get_param_i8", "robstride_get_param_u8", "robstride_get_param_u16", "robstride_get_param_u32", "robstride_get_param_f32")))
    reg.register(MotorAdapter(vendor="myactuator", add_motor_method="add_myactuator_motor"))
    reg.register(MotorAdapter(vendor="hightorque", add_motor_method="add_hightorque_motor"))
    reg.register(MotorAdapter(vendor="hexfellow", add_motor_method="add_hexfellow_motor"))
    return reg
