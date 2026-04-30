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
    "Gripper",
    "RealtimeController",
    "CartesianController",
    "EEFCommand",
    "RobotProfileFactory",
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
    if name == "Gripper":
        from .gripper import Gripper

        return Gripper
    if name == "RealtimeController":
        from .controllers import RealtimeController

        return RealtimeController
    if name == "CartesianController":
        from .controllers import CartesianController

        return CartesianController
    if name == "EEFCommand":
        from .controllers import EEFCommand

        return EEFCommand
    if name == "RobotProfileFactory":
        from .model.robot_profiles import RobotProfileFactory

        return RobotProfileFactory
    raise AttributeError(name)
