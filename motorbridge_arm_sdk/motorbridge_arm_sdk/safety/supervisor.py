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
            if abs(clamped - target) > 1e-9:
                logger.warning(
                    "joint %d clamped: target=%.4f -> [%.4f, %.4f] -> %.4f",
                    idx, target, jc.limit_pos_min, jc.limit_pos_max, clamped,
                )
            out.append(clamped)
        return out

    def clamp_torque_targets(self, tau_target: list[float]) -> list[float]:
        """Clamp per-joint torque targets to their configured limits.

        将每个关节的力矩目标限制在其配置限制内。

        Args:
            tau_target: Desired torques in Nm, one per joint.
                        期望力矩（Nm），每个关节一个。

        Returns:
            List of clamped torques in the same order.
            限幅后的力矩列表，顺序相同。

        Raises:
            ArmError: If the length of *tau_target* does not match the number
                      of configured joints.
        """
        if len(tau_target) != len(self._cfg.joints):
            raise ArmError(ArmErrorCode.ERR_CONFIG, "torque target length mismatch")
        out: list[float] = []
        for idx, (target, jc) in enumerate(zip(tau_target, self._cfg.joints)):
            limit = jc.limit_tau
            clamped = max(-limit, min(limit, target))
            if abs(clamped - target) > 1e-9:
                logger.warning(
                    "joint %d torque clamped: target=%.4f -> [-%.4f, %.4f] -> %.4f",
                    idx, target, limit, limit, clamped,
                )
            out.append(clamped)
        return out

    def validate_velocity_limit(self, vlim: float) -> float:
        if vlim <= 0:
            raise ArmError(ArmErrorCode.ERR_LIMIT, "vlim must be > 0")
        max_allowed = max(j.limit_vel for j in self._cfg.joints)
        clamped = min(vlim, max_allowed)
        if abs(clamped - vlim) > 1e-9:
            logger.warning(
                "velocity limit clamped: requested=%.4f -> max_allowed=%.4f",
                vlim, max_allowed,
            )
        return clamped
