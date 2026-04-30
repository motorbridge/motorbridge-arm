"""Tests for the zero-position calibration module."""
from unittest.mock import MagicMock

from motorbridge_arm_sdk.calibration.zeroing import ZeroCalibrator, ZeroResult
from motorbridge_arm_sdk.errors import ArmError, ArmErrorCode


def _make_calibrator() -> tuple[ZeroCalibrator, MagicMock]:
    session = MagicMock()
    cal = ZeroCalibrator(session)
    return cal, session


def test_zero_joint_success():
    cal, session = _make_calibrator()
    result = cal.zero_joint(2)
    assert result.ok is True
    assert result.scope == "joint:2"
    assert "set_zero_position" in result.detail
    session.set_zero_joint.assert_called_once_with(2)


def test_zero_joint_negative_index():
    cal, _ = _make_calibrator()
    result = cal.zero_joint(-1)
    assert result.ok is False
    assert "invalid" in result.detail
    assert result.scope == "joint:-1"


def test_zero_joint_session_error():
    cal, session = _make_calibrator()
    session.set_zero_joint.side_effect = ArmError(ArmErrorCode.ERR_TIMEOUT, "comm failed")
    result = cal.zero_joint(0)
    assert result.ok is False
    assert "zeroing failed" in result.detail
    assert "comm failed" in result.detail


def test_zero_all_success():
    cal, session = _make_calibrator()
    result = cal.zero_all()
    assert result.ok is True
    assert result.scope == "all"
    assert "zeroed" in result.detail
    session.set_zero_all.assert_called_once()


def test_zero_all_session_error():
    cal, session = _make_calibrator()
    session.set_zero_all.side_effect = RuntimeError("hardware fault")
    result = cal.zero_all()
    assert result.ok is False
    assert "zeroing failed" in result.detail
    assert "hardware fault" in result.detail


def test_zero_result_dataclass():
    r = ZeroResult(ok=True, scope="joint:0", detail="done")
    assert r.ok is True
    assert r.scope == "joint:0"
    assert r.detail == "done"
