"""Tests for the standalone Gripper controller."""
from unittest.mock import MagicMock, patch

from motorbridge_arm_sdk.errors import ArmError, ArmErrorCode
from motorbridge_arm_sdk.gripper import Gripper, GripperConfig
from motorbridge_arm_sdk.session import ModeLike


def _make_gripper() -> tuple[Gripper, MagicMock]:
    cfg = GripperConfig(esc_id=10, feedback_id=10, vendor="robstride", model="rs04")
    g = Gripper(cfg, channel="can0")
    mock_session = MagicMock()
    g._session = mock_session
    mock_motor = MagicMock()
    g._motor = mock_motor
    mock_session.joints = [MagicMock()]
    mock_session.joints[0].motor = mock_motor
    return g, mock_session


def test_mode_mit_success():
    g, session = _make_gripper()
    assert g.mode_mit() is True
    session.ensure_mode_joint.assert_called_once_with(0, ModeLike.MIT)
    assert g.mode == "mit"


def test_mode_mit_propagates_arm_error():
    g, session = _make_gripper()
    session.ensure_mode_joint.side_effect = ArmError(ArmErrorCode.ERR_MODE, "mode switch failed")
    try:
        g.mode_mit()
        assert False, "expected ArmError"
    except ArmError as exc:
        assert "mode switch failed" in exc.message


def test_mode_pos_vel_success():
    g, session = _make_gripper()
    assert g.mode_pos_vel() is True
    session.write_pi_gains.assert_called_once_with(
        0, g._cfg.vel_kp, g._cfg.vel_ki, g._cfg.pos_kp, g._cfg.pos_ki,
    )
    session.ensure_mode_joint.assert_called_once_with(0, ModeLike.POS_VEL)
    assert g.mode == "pos_vel"


def test_mode_pos_vel_propagates_arm_error():
    g, session = _make_gripper()
    session.write_pi_gains.side_effect = ArmError(ArmErrorCode.ERR_TIMEOUT, "write failed")
    try:
        g.mode_pos_vel()
        assert False, "expected ArmError"
    except ArmError as exc:
        assert "write failed" in exc.message


def test_mode_vel_success():
    g, session = _make_gripper()
    assert g.mode_vel() is True
    session.ensure_mode_joint.assert_called_once_with(0, ModeLike.VEL)
    assert g.mode == "vel"


def test_mode_vel_propagates_arm_error():
    g, session = _make_gripper()
    session.ensure_mode_joint.side_effect = ArmError(ArmErrorCode.ERR_MODE, "vel failed")
    try:
        g.mode_vel()
        assert False, "expected ArmError"
    except ArmError as exc:
        assert "vel failed" in exc.message


def test_mit_command_sends_correct_params():
    g, session = _make_gripper()
    g.mode_mit(stabilize_delay=0)
    g.mit(pos=0.5, vel=0.1, kp=15.0, kd=1.5, tau=0.2)
    session.set_mit_joint.assert_called_once_with(0, 0.5, 0.1, 15.0, 1.5, 0.2)


def test_mit_command_not_connected_raises():
    g, _ = _make_gripper()
    g._motor = None
    try:
        g.mit(pos=0.5)
        assert False, "expected ArmError"
    except ArmError as exc:
        assert exc.code == ArmErrorCode.ERR_STATE


def test_pos_vel_command():
    g, session = _make_gripper()
    g.mode_pos_vel(stabilize_delay=0)
    g._motor.get_state.return_value = MagicMock(pos=0.3, vel=0.0, torq=0.1, status_code=1)
    g.pos_vel(0.5, vlim=1.0)
    session.set_pos_vel_joint.assert_called()


def test_set_vel_command():
    g, session = _make_gripper()
    g.set_vel(1.5)
    session.set_vel_joint.assert_called_once_with(0, 1.5)


def test_get_state_with_direction():
    cfg = GripperConfig(esc_id=10, feedback_id=10, direction=-1.0, zero_offset=0.1)
    g = Gripper(cfg, channel="can0")
    mock_motor = MagicMock()
    g._motor = mock_motor
    g._session = MagicMock()
    raw_state = MagicMock(pos=0.5, vel=1.0, torq=2.0)
    mock_motor.get_state.return_value = raw_state
    pos, vel, torq = g.get_state(request=False)
    assert pos == pytest.approx((0.5 - 0.1) * -1.0)
    assert vel == pytest.approx(1.0 * -1.0)
    assert torq == pytest.approx(2.0 * -1.0)


def test_get_state_none_returns_zeros():
    g, _ = _make_gripper()
    g._motor.get_state.return_value = None
    pos, vel, torq = g.get_state(request=False)
    assert (pos, vel, torq) == (0.0, 0.0, 0.0)


def test_set_force_threshold():
    g, _ = _make_gripper()
    g.set_force_threshold(3.0)
    assert g._force_threshold == 3.0


def test_set_force_threshold_negative_raises():
    g, _ = _make_gripper()
    try:
        g.set_force_threshold(-1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_context_manager():
    g, session = _make_gripper()
    with g:
        pass
    session.close.assert_called()


def test_repr():
    g, _ = _make_gripper()
    r = repr(g)
    assert "Gripper" in r
    assert "robstride" in r


import pytest
