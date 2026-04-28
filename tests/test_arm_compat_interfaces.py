from motorbridge_arm_sdk.arm import Arm
from motorbridge_arm_sdk.types import ArmConfig, JointConfig


def _mk_cfg() -> ArmConfig:
    return ArmConfig(
        name="t",
        model="t",
        channel="can0",
        joints=[
            JointConfig(name="j1", vendor="robstride", model="rs-00", esc_id=1, feedback_id=0xFD),
            JointConfig(name="j2", vendor="robstride", model="rs-00", esc_id=2, feedback_id=0xFD),
        ],
        default_home=[0.0, 0.0],
    )


class _FakeSession:
    def __init__(self):
        self.calls = []

    def ensure_mode_all(self, mode):
        self.calls.append(("mode_all", int(mode)))

    def set_pos_vel_all(self, pos, vlim):
        self.calls.append(("pos_vel", list(pos), float(vlim)))

    def set_vel_all(self, vel):
        self.calls.append(("vel", list(vel)))

    def set_mit_all(self, pos, vel, kp, kd, tau):
        self.calls.append(("mit", list(pos), list(vel), list(kp), list(kd), list(tau)))


def test_mode_and_vector_commands():
    arm = Arm(_mk_cfg())
    arm._session = _FakeSession()
    arm.mode_pos_vel()
    arm.pos_vel([0.1, 0.2], vlim=0.8)
    arm.mode_vel()
    arm.set_vel([0.3, 0.4])
    arm.mode_mit()
    arm.mit([0.1, 0.2], vel=[0.0, 0.0], kp=[1.0, 1.0], kd=[0.5, 0.5], tau=[0.0, 0.0])
    assert arm.mode == "mit"
    assert len(arm._session.calls) >= 6


def test_control_loop_start_stop():
    arm = Arm(_mk_cfg())
    ticks = {"n": 0}

    def cb(_arm, _dt):
        ticks["n"] += 1
        if ticks["n"] > 2:
            _arm.stop_control_loop()

    arm.start_control_loop(cb, rate_hz=200)
    import time
    time.sleep(0.05)
    assert ticks["n"] >= 1
