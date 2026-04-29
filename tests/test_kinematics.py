from __future__ import annotations

from unittest.mock import patch

from motorbridge_arm_sdk.model.kinematics import Kinematics
from motorbridge_arm_sdk.types import Pose6D


def test_forward_fallback_without_pinocchio():
    kin = Kinematics()  # no URDF, no pinocchio
    pose = kin.forward([0.0, 0.0, 0.0])
    assert isinstance(pose, Pose6D)


def test_inverse_returns_correct_length():
    kin = Kinematics()
    q_seed = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    target = Pose6D(x=0.3, y=0.0, z=0.2, roll=0.0, pitch=0.0, yaw=0.0)
    result = kin.inverse(target, q_seed)
    assert isinstance(result, list)
    assert len(result) == len(q_seed)


def test_public_properties():
    kin = Kinematics()
    assert kin.has_pinocchio is False
    assert kin.pinocchio_model is None
    assert kin.end_frame_id is None
