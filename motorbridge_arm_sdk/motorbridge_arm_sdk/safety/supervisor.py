from __future__ import annotations

import logging

from ..errors import ArmError, ArmErrorCode
from ..types import ArmConfig

logger = logging.getLogger(__name__)


class SafetySupervisor:
    def __init__(self, config: ArmConfig) -> None:
        self._cfg = config

    def clamp_joint_targets(self, q_target: list[float]) -> list[float]:
        if len(q_target) != len(self._cfg.joints):
            raise ArmError(ArmErrorCode.ERR_CONFIG, "joint target length mismatch")
        out: list[float] = []
        for idx, (target, jc) in enumerate(zip(q_target, self._cfg.joints)):
            clamped = max(jc.limit_pos_min, min(jc.limit_pos_max, target))
            if clamped != target:
                logger.warning(
                    "joint %d clamped: target=%.4f -> [%.4f, %.4f] -> %.4f",
                    idx, target, jc.limit_pos_min, jc.limit_pos_max, clamped,
                )
            out.append(clamped)
        return out

    def validate_velocity_limit(self, vlim: float) -> float:
        if vlim <= 0:
            raise ArmError(ArmErrorCode.ERR_LIMIT, "vlim must be > 0")
        max_allowed = max(j.limit_vel for j in self._cfg.joints)
        return min(vlim, max_allowed)
