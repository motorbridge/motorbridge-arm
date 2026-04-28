from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParamSpec:
    vendor: str
    param_id: int
    param_type: str
    access: str
    name: str
    unit: str = ""
    desc: str = ""


class ParamRegistry:
    def __init__(self) -> None:
        self._specs: dict[tuple[str, int], ParamSpec] = {}

    def register(self, spec: ParamSpec) -> None:
        self._specs[(spec.vendor.lower(), spec.param_id)] = spec

    def get(self, vendor: str, param_id: int) -> ParamSpec | None:
        return self._specs.get((vendor.lower(), param_id))

    def all_for_vendor(self, vendor: str) -> list[ParamSpec]:
        v = vendor.lower()
        return [s for (vv, _), s in self._specs.items() if vv == v]


def create_default_registry() -> ParamRegistry:
    reg = ParamRegistry()
    # Common robstride entries used early in bring-up.
    reg.register(ParamSpec("robstride", 0x200A, "u8", "rw", "CAN_ID"))
    reg.register(ParamSpec("robstride", 0x200B, "u8", "rw", "CAN_MASTER"))
    reg.register(ParamSpec("robstride", 0x200C, "u32", "rw", "CAN_TIMEOUT", unit="ms"))
    reg.register(ParamSpec("robstride", 0x2012, "f32", "rw", "cur_kp"))
    reg.register(ParamSpec("robstride", 0x2013, "f32", "rw", "cur_ki"))
    # Damiao examples.
    reg.register(ParamSpec("damiao", 0x7016, "u32", "rw", "can_timeout_ms", unit="ms"))
    reg.register(ParamSpec("damiao", 0x7017, "f32", "rw", "pos_offset_rad", unit="rad"))
    return reg
