from motorbridge_arm_sdk.arm import Arm
from motorbridge_arm_sdk.types import ArmConfig, JointConfig, JointState, PayloadConfig, ToolConfig


def _mk_cfg() -> ArmConfig:
    return ArmConfig(
        name="t",
        model="t",
        channel="can0",
        joints=[
            JointConfig(name="j1", vendor="robstride", model="rs-00", esc_id=1, feedback_id=0xFD),
            JointConfig(name="gripper", vendor="robstride", model="rs-00", esc_id=2, feedback_id=0xFD),
        ],
        default_home=[0.0, 0.0],
        gripper_joint="gripper",
    )


def test_tool_and_payload_setters():
    arm = Arm(_mk_cfg())
    arm.set_tool(ToolConfig(x=0.01, y=0.02, z=0.03))
    arm.set_payload(PayloadConfig(mass=1.5, cx=0.0, cy=0.0, cz=0.1))
    assert arm._tool.z == 0.03
    assert arm._payload.mass == 1.5


def test_find_gripper_index():
    arm = Arm(_mk_cfg())
    assert arm._find_gripper_index() == 1


def test_get_faults_from_state_cache():
    arm = Arm(_mk_cfg())
    arm._cache.update_joint(0, JointState(name="j1", pos=0.0, vel=0.0, torq=0.0, status_code=1))
    arm._cache.update_joint(1, JointState(name="gripper", pos=0.0, vel=0.0, torq=0.0, status_code=0))
    arm.refresh_state = lambda: arm.get_state()  # avoid hardware call
    f = arm.get_faults()
    assert f.has_fault
    assert "gripper" in f.faulted_joints
