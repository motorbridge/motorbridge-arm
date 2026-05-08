from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from ..types import ArmConfig, JointConfig


def _find_repo_root() -> Path:
    """Walk up from this file to find the repository root.

    Looks for ``pyproject.toml`` or ``.git`` as a marker, avoiding the
    fragile ``parents[N]`` approach that breaks under different install layouts.
    """
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "pyproject.toml").exists() or (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    # Last resort: use the original parents[3] guess.
    return Path(__file__).resolve().parents[3]


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
    repo_root = _find_repo_root()
    fallback = (
        repo_root
        / "models"
        / "urdf"
        / "reBot-DevArm_fixend_description"
        / "urdf"
        / "reBot-DevArm_fixend.urdf"
    )
    return str(fallback)


def _resolve_arm02_urdf_path() -> str:
    """Resolve the arm02 URDF used by motorbridge-studio's /simu page.

    The studio repository owns the visual arm02 assets today.  Keep the
    resolver tolerant so the SDK still imports when only motorbridge-arm is
    installed.
    """
    repo_root = _find_repo_root()
    sibling = (
        repo_root.parent
        / "motorbridge-studio"
        / "public"
        / "resources"
        / "arm02"
        / "reBot_B601_DM_with_gripper"
        / "urdf"
        / "reBot_B601_DM_with_gripper.urdf"
    )
    if sibling.exists():
        return str(sibling)
    return _resolve_urdf_path()


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


def rebot_arm02_dm_with_gripper(channel: str = "can0") -> ArmConfig:
    """Profile matching motorbridge-studio's arm02 simulation asset.

    The first six joints are revolute arm joints.  The final two are the
    gripper prismatic joints from the arm02 URDF.
    """
    joints = [
        JointConfig(name=f"joint{i}", vendor="damiao", model="dm", esc_id=i, feedback_id=0xFD)
        for i in range(1, 7)
    ]
    joints.extend(
        [
            JointConfig(
                name="gripper_joint1",
                vendor="damiao",
                model="dm-gripper",
                esc_id=7,
                feedback_id=0xFD,
                limit_pos_min=0.0,
                limit_pos_max=0.0715,
            ),
            JointConfig(
                name="gripper_joint2",
                vendor="damiao",
                model="dm-gripper",
                esc_id=7,
                feedback_id=0xFD,
                limit_pos_min=0.0,
                limit_pos_max=0.0715,
            ),
        ]
    )
    return ArmConfig(
        name="rebot-arm02-dm-with-gripper",
        model="rebot-arm02-dm-with-gripper",
        channel=channel,
        joints=joints,
        default_home=[0.0] * len(joints),
        loop_dt_s=0.02,
        urdf_path=_resolve_arm02_urdf_path(),
        ee_frame="gripper_link",
        gripper_joint="gripper_joint1",
    )


def arm_robstride(channel: str = "can0") -> ArmConfig:
    """Preferred profile name under motorbridge-arm naming."""
    return rebot_arm_robstride(channel=channel)


def motorbridge_arm_robstride(channel: str = "can0") -> ArmConfig:
    """Alias kept for explicit product naming."""
    return rebot_arm_robstride(channel=channel)
