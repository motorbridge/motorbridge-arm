from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from ..model.profiles import rebot_arm_robstride


EARTH_GRAVITY = (0.0, 0.0, -9.81)
ZERO_GRAVITY = (0.0, 0.0, 0.0)


@dataclass(slots=True)
class DynamicsRobotModel:
    pin: object | None
    model: object | None
    data: object | None
    urdf_path: str

    @property
    def has_pinocchio(self) -> bool:
        return self.pin is not None and self.model is not None and self.data is not None

    @property
    def nq(self) -> int:
        return int(self.model.nq) if self.model is not None else 0

    @property
    def nv(self) -> int:
        return int(self.model.nv) if self.model is not None else 0


def _default_urdf_path() -> str:
    return rebot_arm_robstride().urdf_path


def load_dynamics_robot_model(urdf_path: str | None = None) -> DynamicsRobotModel:
    path = Path(urdf_path or _default_urdf_path())
    if not path.exists():
        return DynamicsRobotModel(pin=None, model=None, data=None, urdf_path=str(path))

    try:
        import pinocchio as pin
    except Exception:
        return DynamicsRobotModel(pin=None, model=None, data=None, urdf_path=str(path))

    model = pin.buildModelFromUrdf(str(path))
    data = model.createData()
    return DynamicsRobotModel(pin=pin, model=model, data=data, urdf_path=str(path))


def create_data(drm: DynamicsRobotModel):
    if not drm.has_pinocchio:
        return None
    return drm.model.createData()


def neutral_configuration(drm: DynamicsRobotModel):
    if not drm.has_pinocchio:
        return []
    return drm.pin.neutral(drm.model)


def random_configuration(drm: DynamicsRobotModel):
    if not drm.has_pinocchio:
        return []
    return drm.pin.randomConfiguration(drm.model)


def set_gravity(drm: DynamicsRobotModel, gravity: Sequence[float]) -> None:
    if not drm.has_pinocchio:
        return
    if len(gravity) != 3:
        raise ValueError(f"gravity must have 3 elements, got {len(gravity)}")
    vec = gravity
    if np is not None:
        vec = np.asarray(gravity, dtype=float)
    drm.model.gravity = drm.pin.Motion(vec)


def get_gravity(drm: DynamicsRobotModel) -> list[float]:
    if not drm.has_pinocchio:
        return list(EARTH_GRAVITY)
    g = drm.model.gravity.linear
    return [float(g[0]), float(g[1]), float(g[2])]
