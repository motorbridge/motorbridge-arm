from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ArmRunState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    IDLE = "IDLE"
    ENABLED = "ENABLED"
    RUNNING = "RUNNING"
    FAULT = "FAULT"


@dataclass(slots=True)
class JointConfig:
    name: str
    vendor: str
    model: str
    esc_id: int
    feedback_id: int
    direction: float = 1.0
    zero_offset: float = 0.0
    limit_pos_min: float = -3.1415926
    limit_pos_max: float = 3.1415926
    limit_vel: float = 2.0
    limit_tau: float = 5.0

    def __post_init__(self):
        if self.direction == 0:
            raise ValueError(f"joint '{self.name}': direction cannot be 0")
        if self.limit_pos_min >= self.limit_pos_max:
            raise ValueError(f"joint '{self.name}': limit_pos_min ({self.limit_pos_min}) must be < limit_pos_max ({self.limit_pos_max})")
        if self.limit_vel < 0:
            raise ValueError(f"joint '{self.name}': limit_vel must be >= 0")
        if self.limit_tau < 0:
            raise ValueError(f"joint '{self.name}': limit_tau must be >= 0")


@dataclass(slots=True)
class ArmConfig:
    name: str
    model: str
    channel: str
    joints: list[JointConfig]
    default_home: list[float] = field(default_factory=list)
    loop_dt_s: float = 0.02
    motion_profile: str = "linear"
    urdf_path: str | None = None
    ee_frame: str = "tool0"
    gripper_joint: str | None = None
    shared_memory_name: str | None = None


@dataclass(slots=True)
class JointState:
    name: str
    pos: float | None
    vel: float | None
    torq: float | None
    status_code: int | None = None
    t_mos: float | None = None
    t_rotor: float | None = None


@dataclass(slots=True)
class Pose6D:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


@dataclass(slots=True)
class ToolConfig:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass(slots=True)
class PayloadConfig:
    mass: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    cz: float = 0.0


@dataclass(slots=True)
class FaultState:
    has_fault: bool
    faulted_joints: list[str]


@dataclass(slots=True)
class ArmState:
    run_state: ArmRunState
    joints: list[JointState]
    updated_at_s: float
