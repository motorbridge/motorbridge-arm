from __future__ import annotations

from dataclasses import dataclass

from ..session import MotorBridgeSession


@dataclass(slots=True)
class ZeroResult:
    ok: bool
    scope: str
    detail: str


class ZeroCalibrator:
    def __init__(self, session: MotorBridgeSession) -> None:
        self._session = session

    def zero_joint(self, index: int) -> ZeroResult:
        self._session.set_zero_joint(index)
        return ZeroResult(ok=True, scope=f"joint:{index}", detail="set_zero_position sent")

    def zero_all(self) -> ZeroResult:
        self._session.set_zero_all()
        return ZeroResult(ok=True, scope="all", detail="all joints zeroed")
