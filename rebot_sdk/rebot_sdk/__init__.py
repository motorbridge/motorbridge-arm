from .errors import ArmError, ArmErrorCode
from .types import ArmConfig, ArmState, FaultState, JointConfig, JointState, PayloadConfig, Pose6D, ToolConfig

__all__ = [
    "ArmError",
    "ArmErrorCode",
    "ArmConfig",
    "ArmState",
    "FaultState",
    "JointConfig",
    "JointState",
    "PayloadConfig",
    "Pose6D",
    "ToolConfig",
    "Arm",
    "SimArm",
    "ArmController",
    "ArmEndPos",
]


def __getattr__(name: str):
    if name == "Arm":
        from .arm import Arm

        return Arm
    if name == "SimArm":
        from .sim import SimArm

        return SimArm
    if name == "ArmController":
        from .controllers import ArmController

        return ArmController
    if name == "ArmEndPos":
        from .controllers import ArmEndPos

        return ArmEndPos
    raise AttributeError(name)
