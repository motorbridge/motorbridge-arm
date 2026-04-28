from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DynamicsRobotModel:
    pin: object | None
    model: object | None
    data: object | None
    urdf_path: str


def load_dynamics_robot_model(urdf_path: str) -> DynamicsRobotModel:
    p = Path(urdf_path)
    if not p.exists():
        return DynamicsRobotModel(pin=None, model=None, data=None, urdf_path=urdf_path)
    try:
        import pinocchio as pin
    except Exception:
        return DynamicsRobotModel(pin=None, model=None, data=None, urdf_path=urdf_path)
    model = pin.buildModelFromUrdf(str(p))
    data = model.createData()
    return DynamicsRobotModel(pin=pin, model=model, data=data, urdf_path=urdf_path)
