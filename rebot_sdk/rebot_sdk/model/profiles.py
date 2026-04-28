from __future__ import annotations

from pathlib import Path

from ..types import ArmConfig, JointConfig


# Example product profile for a 6-DOF robstride arm kit.
def rebot_arm_robstride(channel: str = "can0") -> ArmConfig:
    joints = [
        JointConfig(name=f"j{i}", vendor="robstride", model="rs-00", esc_id=i, feedback_id=0xFD)
        for i in range(1, 7)
    ]
    repo_root = Path(__file__).resolve().parents[3]
    urdf_path = str(
        repo_root
        / "models"
        / "urdf"
        / "reBot-DevArm_fixend_description"
        / "urdf"
        / "reBot-DevArm_fixend.urdf"
    )
    return ArmConfig(
        name="rebot-arm-robstride",
        model="rebot-arm-robstride",
        channel=channel,
        joints=joints,
        default_home=[0.0, -0.4, 1.2, 0.0, 1.0, 0.0],
        loop_dt_s=0.02,
        urdf_path=urdf_path,
        ee_frame="tool0",
    )
