from __future__ import annotations

"""异常值过滤器 —— 保护控制环路免受不合理传感器读数的影响。

Outlier data filter — protects the control loop from unrealistic sensor readings.

本模块提供 :class:`OutlierFilter`，用于检测并替换机械臂关节反馈数据中
超出合理范围的异常值（例如位置超过 360°、速度过大、力矩异常等）。
所有操作均为线程安全的。

This module provides :class:`OutlierFilter`, which detects and replaces
outlier values in robotic arm joint feedback data that fall outside
plausible ranges (e.g. position exceeding 360°, excessive velocity,
abnormal torque).  All operations are thread-safe.
"""

import logging
import math
import threading
from typing import Any

logger = logging.getLogger(__name__)


class OutlierFilter:
    """关节反馈异常值检测与过滤器。

    Joint feedback outlier detection and filter.

    对每个关节的位置、速度、力矩进行范围检查。超过阈值的值会被
    替换为 ``None``，以避免异常数据进入控制环路。

    Performs range checking on each joint's position, velocity, and torque.
    Values exceeding the configured thresholds are replaced with ``None``
    to prevent anomalous data from entering the control loop.

    典型用法 / Typical usage::

        filt = OutlierFilter()
        pos, vel, torq = filt.filter_joint(0, 3.14, 0.5, 1.2)
        # pos=3.14, vel=0.5, torq=1.2  (全部正常 / all normal)

        pos, vel, torq = filt.filter_joint(0, 99.0, 0.5, 1.2)
        # pos=None, vel=0.5, torq=1.2  (位置异常 / position outlier)

    Args:
        max_joint_angle: 关节角度上限（弧度）。默认 6.28（≈360°）。
            Maximum joint angle in radians.  Default 6.28 (≈360°).
        max_end_effector_distance: 末端执行器到原点的最大允许距离（米）。
            默认 1.5 m。
            Maximum allowed end-effector distance from origin in metres.
            Default 1.5 m.
        max_velocity: 关节速度上限（rad/s）。默认 50.0。
            Maximum joint velocity in rad/s.  Default 50.0.
        max_torque: 关节力矩上限（Nm）。默认 100.0。
            Maximum joint torque in Nm.  Default 100.0.
        num_joints: 关节数量（用于统计追踪）。默认 7。
            Number of joints (for statistics tracking).  Default 7.
    """

    def __init__(
        self,
        max_joint_angle: float = 6.28,
        max_end_effector_distance: float = 1.5,
        max_velocity: float = 50.0,
        max_torque: float = 100.0,
        num_joints: int = 7,
    ) -> None:
        self._max_joint_angle = max_joint_angle
        self._max_end_effector_distance = max_end_effector_distance
        self._max_velocity = max_velocity
        self._max_torque = max_torque
        self._num_joints = num_joints

        # 线程锁，保证并发安全 / Thread lock for concurrency safety
        self._lock = threading.Lock()

        # 每个关节的过滤计数：{joint_index: {"pos": int, "vel": int, "torq": int}}
        # Filter counts per joint.
        self._filtered_counts: dict[int, dict[str, int]] = {
            i: {"pos": 0, "vel": 0, "torq": 0} for i in range(num_joints)
        }

    # ------------------------------------------------------------------
    # 公共接口 / Public API
    # ------------------------------------------------------------------

    def filter_joint(
        self,
        index: int,
        pos: float | None,
        vel: float | None,
        torq: float | None,
    ) -> tuple[float | None, float | None, float | None]:
        """过滤单个关节的反馈值，将异常值替换为 ``None``。

        Filter a single joint's feedback values, replacing outliers with ``None``.

        对位置、速度、力矩分别进行阈值检查。超出范围的值会被替换为
        ``None``，同时该关节对应通道的过滤计数器递增。

        Each of *pos*, *vel*, and *torq* is checked against the configured
        threshold.  Out-of-range values are replaced with ``None`` and the
        corresponding per-joint filter counter is incremented.

        Args:
            index: 关节索引（从 0 开始）。
                Zero-based joint index.
            pos: 关节位置（弧度），可为 ``None`` 表示无数据。
                Joint position in radians, or ``None`` if unavailable.
            vel: 关节速度（rad/s），可为 ``None`` 表示无数据。
                Joint velocity in rad/s, or ``None`` if unavailable.
            torq: 关节力矩（Nm），可为 ``None`` 表示无数据。
                Joint torque in Nm, or ``None`` if unavailable.

        Returns:
            一个三元组 ``(filtered_pos, filtered_vel, filtered_torq)``，
            异常值已被替换为 ``None``。

            A 3-tuple ``(filtered_pos, filtered_vel, filtered_torq)``
            where outliers have been replaced with ``None``.
        """
        if index < 0 or index >= self._num_joints:
            raise IndexError(
                f"joint index {index} out of range [0, {self._num_joints})"
            )

        filtered_pos: float | None = None
        filtered_vel: float | None = None
        filtered_torq: float | None = None

        with self._lock:
            # --- position check ---
            if self._is_pos_outlier(pos):
                self._increment(index, "pos")
                logger.warning(
                    "Outlier detected: joint %d position=%.4f exceeds limit %.4f",
                    index, pos, self._max_joint_angle,
                )
            else:
                filtered_pos = pos

            # --- velocity check ---
            if self._is_vel_outlier(vel):
                self._increment(index, "vel")
                logger.warning(
                    "Outlier detected: joint %d velocity=%.4f exceeds limit %.4f",
                    index, vel, self._max_velocity,
                )
            else:
                filtered_vel = vel

            # --- torque check ---
            if self._is_torq_outlier(torq):
                self._increment(index, "torq")
                logger.warning(
                    "Outlier detected: joint %d torque=%.4f exceeds limit %.4f",
                    index, torq, self._max_torque,
                )
            else:
                filtered_torq = torq

        return filtered_pos, filtered_vel, filtered_torq

    def is_outlier(
        self,
        pos: float | None,
        vel: float | None,
        torq: float | None,
    ) -> bool:
        """检查给定值中是否存在至少一个异常值。

        Check whether any of the given values is an outlier.

        不修改内部统计计数器。仅做只读判断。

        This method does **not** modify internal statistics — it performs a
        read-only check.

        Args:
            pos: 关节位置（弧度）。
                Joint position in radians.
            vel: 关节速度（rad/s）。
                Joint velocity in rad/s.
            torq: 关节力矩（Nm）。
                Joint torque in Nm.

        Returns:
            若任意值超出阈值返回 ``True``，否则 ``False``。
            ``True`` if any value exceeds its threshold, ``False`` otherwise.
        """
        return (
            self._is_pos_outlier(pos)
            or self._is_vel_outlier(vel)
            or self._is_torq_outlier(torq)
        )

    def get_stats(self) -> dict[str, Any]:
        """返回每个关节的异常值过滤统计信息。

        Return per-joint outlier filter statistics.

        返回一个字典，包含各关节的位置、速度、力矩被过滤的次数，
        以及所有关节的合计总数。

        Returns a dict mapping joint indices to their per-channel filter
        counts, plus a ``"total"`` summary.

        Returns:
            统计字典，格式如下 / Statistics dict with the following shape::

                {
                    0: {"pos": 3, "vel": 1, "torq": 0},
                    1: {"pos": 0, "vel": 2, "torq": 1},
                    ...
                    "total": {"pos": 3, "vel": 3, "torq": 1}
                }
        """
        with self._lock:
            result: dict[str, Any] = {}
            total_pos = 0
            total_vel = 0
            total_torq = 0
            for idx, counts in sorted(self._filtered_counts.items()):
                result[str(idx)] = dict(counts)
                total_pos += counts["pos"]
                total_vel += counts["vel"]
                total_torq += counts["torq"]
            result["total"] = {"pos": total_pos, "vel": total_vel, "torq": total_torq}
            return result

    # ------------------------------------------------------------------
    # 内部方法 / Internal helpers
    # ------------------------------------------------------------------

    def _is_pos_outlier(self, pos: float | None) -> bool:
        """判断位置是否为异常值。

        Determine whether the position is an outlier.

        位置为 ``None`` 不算异常（可能是传感器无数据）。
        有限值超过 :attr:`_max_joint_angle` 即视为异常。

        A ``None`` position is **not** considered an outlier (it may simply
        indicate missing sensor data).  A finite value exceeding
        :attr:`_max_joint_angle` is flagged.
        """
        if pos is None:
            return False
        if math.isinf(pos) or math.isnan(pos):
            return True
        return abs(pos) > self._max_joint_angle

    def _is_vel_outlier(self, vel: float | None) -> bool:
        """判断速度是否为异常值。

        Determine whether the velocity is an outlier.

        速度为 ``None`` 不算异常。有限值超过 :attr:`_max_velocity` 即视为异常。

        A ``None`` velocity is **not** an outlier.  A finite value exceeding
        :attr:`_max_velocity` is flagged.
        """
        if vel is None:
            return False
        if math.isinf(vel) or math.isnan(vel):
            return True
        return abs(vel) > self._max_velocity

    def _is_torq_outlier(self, torq: float | None) -> bool:
        """判断力矩是否为异常值。

        Determine whether the torque is an outlier.

        力矩为 ``None`` 不算异常。有限值超过 :attr:`_max_torque` 即视为异常。

        A ``None`` torque is **not** an outlier.  A finite value exceeding
        :attr:`_max_torque` is flagged.
        """
        if torq is None:
            return False
        if math.isinf(torq) or math.isnan(torq):
            return True
        return abs(torq) > self._max_torque

    def _increment(self, index: int, channel: str) -> None:
        """递增指定关节通道的过滤计数（调用方需持有锁）。

        Increment the filter count for a joint channel.

        Note:
            调用方必须已经持有 :attr:`_lock`。
            The caller **must** already hold :attr:`_lock`.
        """
        if index < 0 or index >= self._num_joints:
            logger.warning(
                "Ignoring increment for out-of-range joint index %d (valid: 0..%d)",
                index,
                self._num_joints - 1,
            )
            return
        self._filtered_counts[index][channel] += 1
