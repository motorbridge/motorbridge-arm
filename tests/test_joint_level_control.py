from rebot_sdk.arm import Arm
from rebot_sdk.types import ArmConfig, JointConfig, JointState


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


def test_resolve_joint_index():
    arm = Arm(_mk_cfg())
    assert arm._resolve_joint_index(1) == 1
    assert arm._resolve_joint_index("j1") == 0


def test_get_joint_state_by_name():
    arm = Arm(_mk_cfg())
    arm._cache.update_joint(0, JointState(name="j1", pos=0.2, vel=0.0, torq=0.0, status_code=1))
    arm.refresh_state = lambda: arm.get_state()  # avoid hardware
    js = arm.get_joint_state("j1")
    assert js.name == "j1"
    assert js.pos == 0.2


def test_move_joint_updates_target_vector():
    arm = Arm(_mk_cfg())
    arm.get_joint_positions = lambda: [0.0, 0.0]
    captured = {}

    def _fake_move_j(q, vlim=1.0, profile=None):
        captured["q"] = q
        captured["vlim"] = vlim

    arm.move_j = _fake_move_j
    arm.move_joint("j2", 1.5, vlim=0.7)
    assert captured["q"] == [0.0, 1.5]
    assert captured["vlim"] == 0.7


def test_joint_vel_and_mit_call_session():
    arm = Arm(_mk_cfg())
    calls = []

    class _S:
        def ensure_mode_joint(self, i, mode):
            calls.append(("mode", i, int(mode)))

        def set_vel_joint(self, i, vel):
            calls.append(("vel", i, vel))

        def set_mit_joint(self, i, pos, vel, kp, kd, tau):
            calls.append(("mit", i, pos, vel, kp, kd, tau))

    arm._session = _S()
    arm.joint_vel("j1", 0.6)
    arm.joint_mit("j2", pos=0.1, vel=0.2, kp=10.0, kd=1.0, tau=0.0)
    assert calls[0][0] == "mode"
    assert calls[1] == ("vel", 0, 0.6)
    assert calls[2][0] == "mode"
    assert calls[3][0] == "mit"
