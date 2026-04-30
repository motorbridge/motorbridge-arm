from __future__ import annotations

import logging

from dataclasses import dataclass

from ..session import MotorBridgeSession

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ZeroResult:
    ok: bool
    scope: str
    detail: str


class ZeroCalibrator:
    """Performs zero-position calibration for arm joints.

    执行机械臂关节的零位校准。

    Wraps the underlying session calls with bounds checking and error
    handling so that callers always receive a :class:`ZeroResult` rather
    than an unhandled exception.

    对底层会话调用进行边界检查和错误处理，使调用者始终收到
    :class:`ZeroResult` 而不是未处理的异常。
    """

    def __init__(self, session: MotorBridgeSession) -> None:
        self._session = session

    def zero_joint(self, index: int) -> ZeroResult:
        """Zero a single joint by index.

        按索引归零单个关节。

        Args:
            index: Zero-based joint index.  Must be >= 0.
                   从零开始的关节索引。必须 >= 0。

        Returns:
            A :class:`ZeroResult` indicating success or failure.
            表示成功或失败的 :class:`ZeroResult`。
        """
        if index < 0:
            return ZeroResult(
                ok=False,
                scope=f"joint:{index}",
                detail=f"invalid joint index {index}: must be >= 0",
            )
        try:
            self._session.set_zero_joint(index)
            return ZeroResult(ok=True, scope=f"joint:{index}", detail="set_zero_position sent")
        except Exception as exc:
            logger.exception("zero_joint(%d) failed", index)
            return ZeroResult(
                ok=False,
                scope=f"joint:{index}",
                detail=f"zeroing failed: {exc}",
            )

    def zero_all(self) -> ZeroResult:
        """Zero all joints simultaneously.

        同时归零所有关节。

        Returns:
            A :class:`ZeroResult` indicating success or failure.
            表示成功或失败的 :class:`ZeroResult`。
        """
        try:
            self._session.set_zero_all()
            return ZeroResult(ok=True, scope="all", detail="all joints zeroed")
        except Exception as exc:
            logger.exception("zero_all() failed")
            return ZeroResult(
                ok=False,
                scope="all",
                detail=f"zeroing failed: {exc}",
            )
