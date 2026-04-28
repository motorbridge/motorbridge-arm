from rebot_sdk.safety.supervisor import SafetySupervisor
from rebot_sdk.types import ArmConfig, JointConfig


def test_clamp_joint_targets():
    cfg = ArmConfig(
        name="t",
        model="t",
        channel="can0",
        joints=[JointConfig(name="j1", vendor="robstride", model="rs-00", esc_id=1, feedback_id=0xFD, limit_pos_min=-1.0, limit_pos_max=1.0)],
    )
    s = SafetySupervisor(cfg)
    assert s.clamp_joint_targets([3.0]) == [1.0]
