from motorbridge_arm_sdk.errors import ArmError
from motorbridge_arm_sdk.runtime import RuntimeStateMachine
from motorbridge_arm_sdk.types import ArmRunState


def test_valid_state_transition_flow():
    sm = RuntimeStateMachine(ArmRunState.DISCONNECTED)
    sm.transition(ArmRunState.IDLE)
    sm.transition(ArmRunState.ENABLED)
    sm.transition(ArmRunState.RUNNING)
    sm.transition(ArmRunState.ENABLED)
    sm.transition(ArmRunState.IDLE)
    sm.transition(ArmRunState.DISCONNECTED)
    assert sm.state == ArmRunState.DISCONNECTED


def test_invalid_state_transition_raises():
    sm = RuntimeStateMachine(ArmRunState.DISCONNECTED)
    try:
        sm.transition(ArmRunState.RUNNING)
        assert False, "expected ArmError"
    except ArmError as exc:
        assert exc.code.value == "ERR_STATE"
