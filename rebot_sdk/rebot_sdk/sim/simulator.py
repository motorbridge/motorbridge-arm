from __future__ import annotations

from ..model.kinematics import Kinematics
from ..types import ArmConfig, Pose6D


class SimArm:
    """Model-only arm simulator without motor hardware."""

    def __init__(self, config: ArmConfig) -> None:
        self._cfg = config
        self._q = list(config.default_home or [0.0 for _ in config.joints])
        self._kin = Kinematics(config.urdf_path, config.ee_frame)

    def set_joint_positions(self, q: list[float]) -> None:
        if len(q) != len(self._cfg.joints):
            raise ValueError("joint length mismatch")
        self._q = list(q)

    def get_joint_positions(self) -> list[float]:
        return list(self._q)

    def get_pose(self) -> Pose6D:
        return self._kin.forward(self._q)

    def move_j(self, q_target: list[float]) -> None:
        self.set_joint_positions(q_target)

    def solve_ik(self, pose: Pose6D) -> list[float]:
        return self._kin.inverse(pose, self._q)
