from __future__ import annotations

from ..arm import Arm
from ..types import Pose6D


class ArmController:
    """High-level motion controller facade on top of Arm."""

    def __init__(self, arm: Arm) -> None:
        self._arm = arm

    def move_j(self, q_target: list[float], vlim: float = 1.0, profile: str | None = None) -> None:
        self._arm.move_j(q_target, vlim=vlim, profile=profile)

    def move_l(self, target: Pose6D, vlim: float = 1.0, step_m: float = 0.01, profile: str | None = None) -> None:
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
        self._arm.move_c(
            target=target,
            center_x=center_x,
            center_y=center_y,
            normal_z=normal_z,
            vlim=vlim,
            steps=steps,
            profile=profile,
        )
