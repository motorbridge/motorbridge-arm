from rebot_sdk.errors import ArmError
from rebot_sdk.session import MotorBridgeSession


class _FlakyMotor:
    def __init__(self):
        self.calls = 0

    def ensure_mode(self, mode, timeout_ms):
        self.calls += 1
        if self.calls < 2:
            raise RuntimeError("temporary")
        return None


class _BadMotor:
    def ensure_mode(self, mode, timeout_ms):
        raise RuntimeError("always bad")


def test_retry_call_eventual_success():
    s = MotorBridgeSession("can0")
    m = _FlakyMotor()
    s._retry_call(lambda: m.ensure_mode(1, 100), "ensure_mode")
    assert m.calls == 2


def test_retry_call_failure_raises_arm_error():
    s = MotorBridgeSession("can0")
    m = _BadMotor()
    try:
        s._retry_call(lambda: m.ensure_mode(1, 100), "ensure_mode")
        assert False, "expected ArmError"
    except ArmError as exc:
        assert "ensure_mode failed" in exc.message
