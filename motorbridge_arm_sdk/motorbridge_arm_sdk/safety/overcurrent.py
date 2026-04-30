"""过流保护模块 —— 检测关节力矩连续超限并触发保护。

Overcurrent protection module — detects joints whose torque exceeds a
configured limit for too many consecutive frames and reports them.

设计灵感来源于 arx5-sdk 的过流保护策略。每个关节独立维护一个连续
超限帧计数器，当计数器超过阈值时，该关节被标记为过流。

Inspired by the arx5-sdk overcurrent protection strategy.  Each joint
maintains an independent consecutive-overlimit frame counter; when the
counter exceeds a configurable threshold the joint is flagged.

典型用法 / Typical usage::

    from motorbridge_arm_sdk.safety.overcurrent import OvercurrentGuard

    guard = OvercurrentGuard(
        max_torque_per_joint=[5.0, 5.0, 4.0, 4.0, 3.0, 3.0, 2.0],
        max_consecutive_count=10,
    )

    # 每个控制周期调用 / called every control cycle
    faulted = guard.check(joint_states)
    if faulted:
        print("Overcurrent on joints:", faulted)

    # 故障清除后重置 / reset after fault is cleared
    guard.reset()
"""
from __future__ import annotations

import logging
import threading

from ..types import JointState

logger = logging.getLogger(__name__)


class OvercurrentGuard:
    """关节过流保护守护。

    Joint overcurrent protection guard.

    跟踪每个关节的力矩，当某个关节的力矩绝对值持续超过其配置阈值
    达到指定连续帧数时，将该关节报告为过流。所有操作通过
    ``threading.Lock`` 保证线程安全。

    Tracks per-joint torque.  When a joint's absolute torque stays above its
    configured limit for more than *max_consecutive_count* consecutive frames
    the joint is reported as overcurrent.  All operations are protected by a
    ``threading.Lock`` for thread safety.

    Args:
        max_torque_per_joint:
            每个关节允许的最大力矩绝对值（Nm）列表，长度等于关节数。
            List of maximum allowed absolute torque values (Nm) per joint.
        max_consecutive_count:
            触发过流保护所需的最小连续超限帧数。默认 10。
            Minimum number of consecutive over-limit frames required to
            trigger overcurrent protection.  Default 10.
    """

    def __init__(
        self,
        max_torque_per_joint: list[float],
        max_consecutive_count: int = 10,
    ) -> None:
        if not max_torque_per_joint:
            raise ValueError("max_torque_per_joint must not be empty")
        if max_consecutive_count < 1:
            raise ValueError("max_consecutive_count must be >= 1")

        self._max_torque: list[float] = list(max_torque_per_joint)
        self._max_count: int = max_consecutive_count
        self._num_joints: int = len(self._max_torque)

        # 每个关节的连续过流帧计数 / consecutive overcurrent frame counts
        self._counts: list[int] = [0] * self._num_joints

        # 线程锁 / thread lock for concurrency safety
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 公共接口 / Public API
    # ------------------------------------------------------------------

    def check(self, joint_states: list[JointState]) -> list[int]:
        """检查当前帧各关节力矩，返回过流关节索引列表。

        Check all joints for overcurrent this frame and return a list of
        joint indices that exceeded the threshold for too many consecutive
        frames.

        每次调用时，对每个关节判断其力矩绝对值是否超过阈值：
        - 若超过，递增该关节的连续计数器。
        - 若未超过，将计数器归零。

        若某个关节的连续计数器达到 *max_consecutive_count*，
        则该关节索引被加入返回列表。

        On each call the absolute torque of every joint is compared with
        its threshold:

        - If exceeded, the joint's consecutive counter is incremented.
        - If not exceeded, the counter is reset to zero.

        When a joint's counter reaches *max_consecutive_count* its index
        is included in the returned list.

        Args:
            joint_states:
                当前帧的关节状态列表，顺序与构造时 *max_torque_per_joint* 一致。
                Current frame joint states, in the same order as the
                *max_torque_per_joint* list passed to the constructor.

        Returns:
            过流关节的索引列表（可能为空）。关节一旦首次触发即被
            报告，后续帧只要计数器仍 >= max_consecutive_count 就
            会继续出现在列表中。

            List of joint indices that are in an overcurrent state.
            May be empty.  A joint is reported as soon as its counter
            reaches *max_consecutive_count* and continues to be reported
            on subsequent frames as long as the counter stays at or above
            that threshold.
        """
        if len(joint_states) != self._num_joints:
            logger.warning(
                "joint_states length (%d) != expected (%d); skipping check",
                len(joint_states),
                self._num_joints,
            )
            return []

        faulted: list[int] = []

        with self._lock:
            for idx, js in enumerate(joint_states):
                torq = js.torq

                # 无力矩数据则跳过 / skip if no torque reading
                if torq is None:
                    self._counts[idx] = 0
                    continue

                if abs(torq) > self._max_torque[idx]:
                    self._counts[idx] += 1
                    logger.warning(
                        "Joint %d overcurrent: torque=%.4f Nm, limit=%.4f Nm, "
                        "consecutive=%d/%d",
                        idx,
                        torq,
                        self._max_torque[idx],
                        self._counts[idx],
                        self._max_count,
                    )
                else:
                    self._counts[idx] = 0

                if self._counts[idx] >= self._max_count:
                    faulted.append(idx)

        if faulted:
            logger.warning(
                "Overcurrent triggered on joints: %s", faulted,
            )

        return faulted

    def reset(self) -> None:
        """清除所有关节的连续过流计数器。

        Reset all per-joint consecutive overcurrent counters to zero.

        通常在故障解除、操作员确认安全后调用。

        Typically called after a fault has been acknowledged and the
        operator has confirmed it is safe to resume.
        """
        with self._lock:
            self._counts = [0] * self._num_joints
        logger.info("Overcurrent counters reset")

    # ------------------------------------------------------------------
    # 辅助属性 / Utility properties
    # ------------------------------------------------------------------

    @property
    def counts(self) -> list[int]:
        """返回当前各关节连续过流帧计数的副本。

        Return a copy of the current per-joint consecutive overlimit
        frame counts.
        """
        with self._lock:
            return list(self._counts)

    @property
    def num_joints(self) -> int:
        """返回监控的关节数量。 / Number of monitored joints."""
        return self._num_joints
