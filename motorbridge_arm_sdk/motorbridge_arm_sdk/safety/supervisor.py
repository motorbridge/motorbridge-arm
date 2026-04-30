from __future__ import annotations

import logging
import math
import time

from ..errors import ArmError, ArmErrorCode
from ..types import ArmConfig, Pose6D

logger = logging.getLogger(__name__)


class SafetySupervisor:
    def __init__(self, config: ArmConfig) -> None:
        self._cfg = config
        self._watchdog_timeout_s: float = 0.0  # 0 means disabled.
        self._watchdog_last_feed: float = time.monotonic()

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

    # -- Pose jump rejection / 位姿跳变拒绝 ------------------------------------

    @staticmethod
    def _pose_distance(p1: Pose6D, p2: Pose6D) -> tuple[float, float]:
        """Return (pos_distance, rot_distance) between two poses.

        Position distance is the Euclidean distance in metres.
        Rotation distance is the sum of absolute RPY differences in radians.

        / 返回两个位姿之间的 (位置距离, 旋转距离)。
        """
        pos_dist = math.sqrt(
            (p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2 + (p2.z - p1.z) ** 2
        )
        rot_dist = (
            abs(p2.roll - p1.roll)
            + abs(p2.pitch - p1.pitch)
            + abs(p2.yaw - p1.yaw)
        )
        return pos_dist, rot_dist

    def check_pose_jump(
        self,
        last_pose: Pose6D,
        new_pose: Pose6D,
        max_pos_jump: float = 0.1,
        max_rot_jump: float = 0.5,
    ) -> bool:
        """Check whether the pose change between two commands is too large.

        Returns ``True`` if the jump exceeds either threshold, indicating a
        potentially dangerous discontinuity.

        / 检查两次指令之间的位姿变化是否过大。如果跳变超过阈值则返回 True。

        Args:
            last_pose: Previous commanded pose.
            new_pose: Newly requested pose.
            max_pos_jump: Maximum allowed position jump in metres.
                Defaults to ``0.1``.
            max_rot_jump: Maximum allowed rotation jump in radians (sum of
                absolute RPY deltas).  Defaults to ``0.5``.

        Returns:
            ``True`` if the jump is too large, ``False`` if acceptable.
        """
        pos_dist, rot_dist = self._pose_distance(last_pose, new_pose)
        if pos_dist > max_pos_jump or rot_dist > max_rot_jump:
            logger.warning(
                "pose jump detected: pos_dist=%.4f m (max %.4f), rot_dist=%.4f rad (max %.4f)",
                pos_dist, max_pos_jump, rot_dist, max_rot_jump,
            )
            return True
        return False

    # -- Command watchdog / 指令看门狗 ----------------------------------------

    def set_watchdog_timeout(self, timeout_s: float) -> None:
        """Set the watchdog timeout in seconds.

        A value of ``0`` disables the watchdog.  When enabled, the caller
        must call :meth:`feed_watchdog` at least once within every timeout
        window to prevent expiration.

        / 设置看门狗超时（秒）。值为 0 表示禁用。启用后，调用者必须
        在每个超时窗口内至少调用一次 feed_watchdog 以防止过期。

        Args:
            timeout_s: Timeout in seconds.  Use ``0`` to disable.
        """
        if timeout_s < 0.0:
            raise ValueError("watchdog timeout must be >= 0")
        self._watchdog_timeout_s = timeout_s
        self._watchdog_last_feed = time.monotonic()

    def feed_watchdog(self) -> None:
        """Reset the watchdog timer, indicating that a valid command was sent.

        / 重置看门狗计时器，表示已发送有效指令。
        """
        self._watchdog_last_feed = time.monotonic()

    def is_watchdog_expired(self) -> bool:
        """Return ``True`` if the watchdog is enabled and no command has been
        received within the configured timeout.

        / 如果看门狗已启用且在配置的超时时间内未收到指令，则返回 True。
        """
        if self._watchdog_timeout_s <= 0.0:
            return False
        elapsed = time.monotonic() - self._watchdog_last_feed
        return elapsed > self._watchdog_timeout_s

    # -- EE target validation / 末端执行器目标验证 ----------------------------

    def validate_ee_target(self, last_pose: Pose6D, new_pose: Pose6D) -> Pose6D:
        """Validate an EE target pose and reject pose jumps.

        If the new pose is more than ``0.1 m`` away from the last pose the
        target is rejected and the last pose is returned instead, effectively
        clamping the motion.

        / 验证末端执行器目标位姿并拒绝跳变。如果新位姿与上一位姿的距离
        超过 0.1 m，则拒绝目标并返回上一位姿。

        Args:
            last_pose: Previously accepted pose.
            new_pose: Candidate new pose.

        Returns:
            ``new_pose`` if the jump is within limits, otherwise ``last_pose``.
        """
        if self.check_pose_jump(last_pose, new_pose, max_pos_jump=0.1, max_rot_jump=0.5):
            return last_pose
        return new_pose
