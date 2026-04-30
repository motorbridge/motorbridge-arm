from __future__ import annotations

from ..arm import Arm
from ..types import Pose6D


class ArmController:
    """High-level motion controller facade on top of Arm.

    基于 Arm 的高层运动控制器外观。

    Provides a simplified, validated interface for common motion primitives
    (joint-space, linear, and circular moves).  Each method validates its
    parameters before delegating to the underlying :class:`Arm`.

    为常见运动基元（关节空间、线性和圆弧运动）提供简化且经过验证的接口。
    每个方法在委托给底层 :class:`Arm` 之前都会验证其参数。

    Args:
        arm: The underlying arm instance to control.
             底层机械臂实例。
    """

    def __init__(self, arm: Arm) -> None:
        self._arm = arm

    def move_j(self, q_target: list[float], vlim: float = 1.0, profile: str | None = None) -> None:
        """Move to a joint-space target.

        移动到关节空间目标。

        Args:
            q_target: Target joint positions in radians.
                      目标关节位置（弧度）。
            vlim: Velocity limit in rad/s.  Must be > 0.
                  速度限制（rad/s）。必须 > 0。
            profile: Optional motion profile name (e.g. ``"min_jerk"``).
                     可选的运动轨迹名称（如 ``"min_jerk"``）。

        Raises:
            ValueError: If *vlim* is not positive.
        """
        if vlim <= 0:
            raise ValueError(f"vlim must be > 0, got {vlim}")
        self._arm.move_j(q_target, vlim=vlim, profile=profile)

    def move_l(self, target: Pose6D, vlim: float = 1.0, step_m: float = 0.01, profile: str | None = None) -> None:
        """Move in a straight-line (Cartesian) path to *target*.

        沿直线（笛卡尔）路径移动到 *target*。

        Args:
            target: Desired end-effector pose.
                    期望的末端执行器位姿。
            vlim: Velocity limit in rad/s.  Must be > 0.
                  速度限制（rad/s）。必须 > 0。
            step_m: Cartesian step size in metres.  Must be > 0.
                    笛卡尔步长（米）。必须 > 0。
            profile: Optional motion profile name.
                     可选的运动轨迹名称。

        Raises:
            ValueError: If *vlim* or *step_m* is not positive.
        """
        if vlim <= 0:
            raise ValueError(f"vlim must be > 0, got {vlim}")
        if step_m <= 0:
            raise ValueError(f"step_m must be > 0, got {step_m}")
        self._arm.move_l(target, vlim=vlim, step_m=step_m, profile=profile)

    def move_c(
        self,
        target: Pose6D,
        center_x: float,
        center_y: float,
        normal_z: float = 1.0,
        vlim: float = 1.0,
        steps: int = 80,
        profile: str | None = None,
    ) -> None:
        """Move along a circular arc to *target*.

        沿圆弧移动到 *target*。

        Args:
            target: Desired end-effector pose at the end of the arc.
                    圆弧终点的期望末端执行器位姿。
            center_x: X coordinate of the arc centre.
                      圆弧中心的 X 坐标。
            center_y: Y coordinate of the arc centre.
                      圆弧中心的 Y 坐标。
            normal_z: Z component of the arc plane normal.  Defaults to 1.0.
                      圆弧平面法线的 Z 分量。默认 1.0。
            vlim: Velocity limit in rad/s.  Must be > 0.
                  速度限制（rad/s）。必须 > 0。
            steps: Number of interpolation steps along the arc.  Must be >= 1.
                   沿圆弧的插值步数。必须 >= 1。
            profile: Optional motion profile name.
                     可选的运动轨迹名称。

        Raises:
            ValueError: If *vlim* is not positive or *steps* is less than 1.
        """
        if vlim <= 0:
            raise ValueError(f"vlim must be > 0, got {vlim}")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        self._arm.move_c(
            target=target,
            center_x=center_x,
            center_y=center_y,
            normal_z=normal_z,
            vlim=vlim,
            steps=steps,
            profile=profile,
        )
