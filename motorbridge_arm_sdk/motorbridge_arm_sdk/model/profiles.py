from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from ..types import ArmConfig, JointConfig


def _resolve_urdf_path() -> str:
    """Resolve URDF path from package assets, with source-tree fallback."""
    # Try package-installed assets first (works after pip install)
    try:
        pkg = files("motorbridge_arm_sdk")
        candidate = pkg / "assets" / "urdf" / "reBot-DevArm_fixend_description" / "urdf" / "reBot-DevArm_fixend.urdf"
        p = Path(str(candidate))
        if p.exists():
            return str(p)
    except Exception:
        pass
    # Fallback: source-tree relative path (development mode)
    repo_root = Path(__file__).resolve().parents[3]
    fallback = (
        repo_root
        / "models"
        / "urdf"
        / "reBot-DevArm_fixend_description"
        / "urdf"
        / "reBot-DevArm_fixend.urdf"
    )
    return str(fallback)


# Example product profile for a 6-DOF robstride arm kit.
def rebot_arm_robstride(channel: str = "can0") -> ArmConfig:
    joints = [
        JointConfig(name=f"j{i}", vendor="robstride", model="rs-00", esc_id=i, feedback_id=0xFD)
        for i in range(1, 7)
    ]
    urdf_path = _resolve_urdf_path()
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


def arm_robstride(channel: str = "can0") -> ArmConfig:
    """Preferred profile name under motorbridge-arm naming."""
    return rebot_arm_robstride(channel=channel)


def motorbridge_arm_robstride(channel: str = "can0") -> ArmConfig:
    """Alias kept for explicit product naming."""
    return rebot_arm_robstride(channel=channel)
