from __future__ import annotations

from dataclasses import dataclass

_VALID_PARAM_TYPES = {"u8", "u16", "u32", "i8", "i16", "i32", "f32", "f64", "bool"}
_VALID_ACCESS = {"r", "w", "rw"}


@dataclass(frozen=True, slots=True)
class ParamSpec:
    vendor: str
    param_id: int
    param_type: str
    access: str
    name: str
    unit: str = ""
    desc: str = ""

    def __post_init__(self) -> None:
        if self.param_type not in _VALID_PARAM_TYPES:
            raise ValueError(
                f"invalid param_type {self.param_type!r}; "
                f"must be one of {sorted(_VALID_PARAM_TYPES)}"
            )
        if self.access not in _VALID_ACCESS:
            raise ValueError(
                f"invalid access {self.access!r}; "
                f"must be one of {sorted(_VALID_ACCESS)}"
            )


class ParamRegistry:
    def __init__(self) -> None:
        self._specs: dict[tuple[str, int], ParamSpec] = {}

    def register(self, spec: ParamSpec) -> None:
        key = (spec.vendor.lower(), spec.param_id)
        if key in self._specs:
            raise ValueError(
                f"duplicate registration for vendor={spec.vendor!r} "
                f"param_id=0x{spec.param_id:04X}"
            )
        self._specs[key] = spec

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
