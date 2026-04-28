from .errors import ArmError, ArmErrorCode
from .types import ArmConfig, ArmState, JointConfig, JointState, Pose6D

__all__ = [
    "ArmError",
    "ArmErrorCode",
    "ArmConfig",
    "ArmState",
    "JointConfig",
    "JointState",
    "Pose6D",
    "Arm",
    "SimArm",
    "ArmController",
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
    raise AttributeError(name)
