from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ArmErrorCode(str, Enum):
    ERR_TIMEOUT = "ERR_TIMEOUT"
    ERR_MODE = "ERR_MODE"
    ERR_LIMIT = "ERR_LIMIT"
    ERR_SINGULAR = "ERR_SINGULAR"
    ERR_NO_IK = "ERR_NO_IK"
    ERR_BUS = "ERR_BUS"
    ERR_STATE = "ERR_STATE"
    ERR_CONFIG = "ERR_CONFIG"
    ERR_UNSUPPORTED = "ERR_UNSUPPORTED"


@dataclass(slots=True)
class ArmError(Exception):
    code: ArmErrorCode
    message: str

    def __post_init__(self):
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.code.value}: {self.message}"
