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


@dataclass(slots=True)
class ArmConfig:
    name: str
    model: str
    channel: str
    joints: list[JointConfig]
    default_home: list[float] = field(default_factory=list)
    loop_dt_s: float = 0.02
    urdf_path: str | None = None
    ee_frame: str = "tool0"


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
class ArmState:
    run_state: ArmRunState
    joints: list[JointState]
    updated_at_s: float
