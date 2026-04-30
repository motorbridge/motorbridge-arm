"""Pre-configured robot model profiles (factory pattern).

Provides :class:`RobotProfileFactory` with complete kinematic and dynamic
configurations for known robot models.  Each profile includes joint limits,
motor assignments, URDF paths, and default controller gains.

Inspired by the arx5-sdk ``RobotConfigFactory``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..types import ArmConfig, JointConfig, Pose6D


@dataclass(frozen=True, slots=True)
class MotorTypeInfo:
    """Motor type and vendor info for a single joint.

    Attributes:
        vendor: Motor vendor name.
        model: Motor model name.
    """

    vendor: str
    model: str


@dataclass(frozen=True, slots=True)
class RobotProfile:
    """Complete configuration for a specific robot model.

    Attributes:
        name: Robot model name (e.g. ``"X5"``, ``"L5"``, ``"X7_left"``).
        dof: Number of degrees of freedom.
        joint_pos_min: Minimum joint positions in radians.
        joint_pos_max: Maximum joint positions in radians.
        joint_vel_max: Maximum joint velocities in rad/s.
        joint_torque_max: Maximum joint torques in Nm.
        motor_ids: CAN IDs for each joint motor.
        motor_types: Motor type info for each joint.
        gripper_motor_id: CAN ID for the gripper motor (0 if none).
        gripper_motor_type: Motor type info for the gripper.
        gripper_width: Fully-opened gripper width in metres.
        urdf_filename: URDF file name (relative to a models directory).
        ee_frame: End-effector frame name in the URDF.
        default_home: Default home position (radians), or ``None`` for all-zeros.
        default_kp: Default joint PD position gains.
        default_kd: Default joint PD velocity gains.
    """

    name: str
    dof: int
    joint_pos_min: list[float]
    joint_pos_max: list[float]
    joint_vel_max: list[float]
    joint_torque_max: list[float]
    motor_ids: list[int]
    motor_types: list[MotorTypeInfo]
    gripper_motor_id: int = 0
    gripper_motor_type: MotorTypeInfo | None = None
    gripper_width: float = 0.088
    urdf_filename: str = ""
    ee_frame: str = "eef_link"
    default_home: list[float] | None = None
    default_kp: list[float] | None = None
    default_kd: list[float] | None = None


def _dm_j4310() -> MotorTypeInfo:
    return MotorTypeInfo(vendor="damiao", model="J4310")


def _dm_j4340() -> MotorTypeInfo:
    return MotorTypeInfo(vendor="damiao", model="J4340")


def _ec_a4310() -> MotorTypeInfo:
    return MotorTypeInfo(vendor="robstride", model="04")


class RobotProfileFactory:
    """Factory for pre-configured robot profiles.

    Usage::

        factory = RobotProfileFactory()
        profile = factory.get("X5")
        arm_config = profile.to_arm_config(channel="can0")

    Currently available models:
        ``"X5"``, ``"X5_umi"``, ``"L5"``, ``"L5_umi"``,
        ``"X7_left"``, ``"X7_right"``
    """

    def __init__(self) -> None:
        self._profiles: dict[str, RobotProfile] = {}
        self._register_defaults()

    def get(self, name: str) -> RobotProfile:
        """Look up a robot profile by name.

        Args:
            name: Robot model name (case-sensitive).

        Returns:
            The matching :class:`RobotProfile`.

        Raises:
            KeyError: If the model name is not registered.
        """
        if name not in self._profiles:
            available = ", ".join(sorted(self._profiles.keys()))
            raise KeyError(f"Unknown robot model '{name}'. Available: {available}")
        return self._profiles[name]

    def register(self, profile: RobotProfile) -> None:
        """Register a custom robot profile.

        Args:
            profile: The profile to register.
        """
        self._profiles[profile.name] = profile

    @property
    def available_models(self) -> list[str]:
        return sorted(self._profiles.keys())

    def _register_defaults(self) -> None:
        # ---- X5 (6-DOF) ----
        self._profiles["X5"] = RobotProfile(
            name="X5",
            dof=6,
            joint_pos_min=[-3.14, -0.05, -0.1, -1.6, -1.57, -2.0],
            joint_pos_max=[2.618, 3.50, 3.20, 1.55, 1.57, 2.0],
            joint_vel_max=[5.0, 5.0, 5.5, 5.5, 5.0, 5.0],
            joint_torque_max=[30.0, 40.0, 30.0, 15.0, 10.0, 10.0],
            motor_ids=[1, 2, 4, 5, 6, 7],
            motor_types=[_ec_a4310(), _ec_a4310(), _ec_a4310(), _dm_j4310(), _dm_j4310(), _dm_j4310()],
            gripper_motor_id=8,
            gripper_motor_type=_dm_j4310(),
            gripper_width=0.088,
            urdf_filename="X5.urdf",
            ee_frame="eef_link",
            default_kp=[80.0, 70.0, 70.0, 30.0, 30.0, 20.0],
            default_kd=[2.0, 2.0, 2.0, 1.0, 1.0, 0.7],
        )

        # ---- X5_umi ----
        self._profiles["X5_umi"] = RobotProfile(
            name="X5_umi",
            dof=6,
            joint_pos_min=[-3.14, -0.05, -0.1, -1.6, -1.57, -2.0],
            joint_pos_max=[2.618, 3.50, 3.20, 1.55, 1.57, 2.0],
            joint_vel_max=[5.0, 5.0, 5.5, 5.5, 5.0, 5.0],
            joint_torque_max=[30.0, 40.0, 30.0, 15.0, 10.0, 10.0],
            motor_ids=[1, 2, 4, 5, 6, 7],
            motor_types=[_ec_a4310(), _ec_a4310(), _ec_a4310(), _dm_j4310(), _dm_j4310(), _dm_j4310()],
            gripper_motor_id=8,
            gripper_motor_type=_dm_j4310(),
            gripper_width=0.086,
            urdf_filename="X5_umi.urdf",
            ee_frame="eef_link",
            default_kp=[80.0, 70.0, 70.0, 30.0, 30.0, 20.0],
            default_kd=[2.0, 2.0, 2.0, 1.0, 1.0, 0.7],
        )

        # ---- L5 (6-DOF, Damiao motors) ----
        self._profiles["L5"] = RobotProfile(
            name="L5",
            dof=6,
            joint_pos_min=[-3.14, -0.05, -0.1, -1.6, -1.57, -2.0],
            joint_pos_max=[2.618, 3.50, 3.20, 1.55, 1.57, 2.0],
            joint_vel_max=[5.0, 5.0, 5.5, 5.5, 5.0, 5.0],
            joint_torque_max=[30.0, 40.0, 30.0, 15.0, 10.0, 10.0],
            motor_ids=[1, 2, 4, 5, 6, 7],
            motor_types=[_dm_j4340(), _dm_j4340(), _dm_j4340(), _dm_j4310(), _dm_j4310(), _dm_j4310()],
            gripper_motor_id=8,
            gripper_motor_type=_dm_j4310(),
            gripper_width=0.088,
            urdf_filename="L5.urdf",
            ee_frame="eef_link",
            default_kp=[80.0, 70.0, 70.0, 30.0, 30.0, 20.0],
            default_kd=[2.0, 2.0, 2.0, 1.0, 1.0, 0.7],
        )

        # ---- L5_umi ----
        self._profiles["L5_umi"] = RobotProfile(
            name="L5_umi",
            dof=6,
            joint_pos_min=[-3.14, -0.05, -0.1, -1.6, -1.57, -2.0],
            joint_pos_max=[2.618, 3.50, 3.20, 1.55, 1.57, 2.0],
            joint_vel_max=[5.0, 5.0, 5.5, 5.5, 5.0, 5.0],
            joint_torque_max=[30.0, 40.0, 30.0, 15.0, 10.0, 10.0],
            motor_ids=[1, 2, 4, 5, 6, 7],
            motor_types=[_dm_j4340(), _dm_j4340(), _dm_j4340(), _dm_j4310(), _dm_j4310(), _dm_j4310()],
            gripper_motor_id=8,
            gripper_motor_type=_dm_j4310(),
            gripper_width=0.086,
            urdf_filename="L5_umi.urdf",
            ee_frame="eef_link",
            default_kp=[80.0, 70.0, 70.0, 30.0, 30.0, 20.0],
            default_kd=[2.0, 2.0, 2.0, 1.0, 1.0, 0.7],
        )

        # ---- X7_left (7-DOF) ----
        self._profiles["X7_left"] = RobotProfile(
            name="X7_left",
            dof=7,
            joint_pos_min=[-2.094, -1.5, -1.5, -1.5, -1.2, -0.3, -0.785],
            joint_pos_max=[2.094, 0.3, 1.5, 0.3, 1.2, 0.785, 0.785],
            joint_vel_max=[3.0, 5.0, 5.0, 5.5, 5.5, 5.0, 5.0],
            joint_torque_max=[30.0, 30.0, 40.0, 30.0, 15.0, 10.0, 10.0],
            motor_ids=[1, 2, 3, 4, 5, 6, 7],
            motor_types=[_dm_j4340(), _dm_j4340(), _dm_j4340(), _dm_j4340(), _dm_j4310(), _dm_j4310(), _dm_j4310()],
            gripper_motor_id=8,
            gripper_motor_type=_dm_j4310(),
            gripper_width=0.088,
            urdf_filename="X7_left.urdf",
            ee_frame="eef_link",
            default_kp=[80.0, 70.0, 70.0, 70.0, 30.0, 30.0, 20.0],
            default_kd=[2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.7],
        )

        # ---- X7_right (7-DOF) ----
        self._profiles["X7_right"] = RobotProfile(
            name="X7_right",
            dof=7,
            joint_pos_min=[-2.094, -0.3, -1.5, -0.3, -1.2, -0.785, -0.785],
            joint_pos_max=[2.094, 1.5, 1.5, 1.5, 1.2, 0.3, 0.785],
            joint_vel_max=[3.0, 5.0, 5.0, 5.5, 5.5, 5.0, 5.0],
            joint_torque_max=[30.0, 30.0, 40.0, 30.0, 15.0, 10.0, 10.0],
            motor_ids=[1, 2, 3, 4, 5, 6, 7],
            motor_types=[_dm_j4340(), _dm_j4340(), _dm_j4340(), _dm_j4340(), _dm_j4310(), _dm_j4310(), _dm_j4310()],
            gripper_motor_id=8,
            gripper_motor_type=_dm_j4310(),
            gripper_width=0.088,
            urdf_filename="X7_right.urdf",
            ee_frame="eef_link",
            default_kp=[80.0, 70.0, 70.0, 70.0, 30.0, 30.0, 20.0],
            default_kd=[2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.7],
        )


def profile_to_arm_config(
    profile: RobotProfile,
    channel: str = "can0",
    urdf_dir: str | None = None,
    loop_dt_s: float = 0.002,
) -> ArmConfig:
    """Convert a :class:`RobotProfile` to an :class:`ArmConfig`.

    Args:
        profile: The robot profile to convert.
        channel: CAN bus interface name.  Default ``"can0"``.
        urdf_dir: Directory containing URDF files.  If ``None``, the URDF
            path is set to just the filename.
        loop_dt_s: Control loop period in seconds.  Default 0.002 (500 Hz).

    Returns:
        An :class:`ArmConfig` ready for use with :class:`Arm`.
    """
    joints: list[JointConfig] = []
    for i in range(profile.dof):
        mt = profile.motor_types[i]
        joints.append(JointConfig(
            name=f"joint_{i}",
            vendor=mt.vendor,
            model=mt.model,
            esc_id=profile.motor_ids[i],
            feedback_id=profile.motor_ids[i],
            limit_pos_min=profile.joint_pos_min[i],
            limit_pos_max=profile.joint_pos_max[i],
            limit_vel=profile.joint_vel_max[i],
            limit_tau=profile.joint_torque_max[i],
        ))

    urdf_path: str | None = None
    if profile.urdf_filename:
        if urdf_dir:
            urdf_path = os.path.join(urdf_dir, profile.urdf_filename)
        else:
            urdf_path = profile.urdf_filename

    return ArmConfig(
        name=profile.name,
        model=profile.name,
        joints=joints,
        channel=channel,
        urdf_path=urdf_path,
        ee_frame=profile.ee_frame,
        loop_dt_s=loop_dt_s,
        default_home=profile.default_home or [],
    )
