"""Dynamics robot model wrapping Pinocchio for rigid-body dynamics.
/ 基于 Pinocchio 的刚体动力学机器人模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import numpy as np
except ImportError:
    np = None

from ..model.profiles import rebot_arm_robstride


EARTH_GRAVITY = (0.0, 0.0, -9.81)
ZERO_GRAVITY = (0.0, 0.0, 0.0)


@dataclass(slots=True)
class DynamicsRobotModel:
    """Container for a Pinocchio rigid-body dynamics model and its data.
    / Pinocchio 刚体动力学模型及其数据的容器。

    Wraps the Pinocchio ``model`` and ``data`` objects together with
    a reference to the ``pinocchio`` module itself so that downstream
    dynamics routines can call Pinocchio functions without importing
    the library again.

    When Pinocchio is not available (import failure or missing URDF),
    all fields default to ``None`` and ``has_pinocchio`` returns ``False``,
    causing the dynamics functions to return zero-valued fallback results.

    Attributes:
        pin: The ``pinocchio`` module reference, or ``None``.
            / ``pinocchio`` 模块引用，或 ``None``。
        model: Pinocchio ``Model`` object built from a URDF, or ``None``.
            / 从 URDF 构建的 Pinocchio ``Model`` 对象，或 ``None``。
        data: Pinocchio ``Data`` object associated with the model, or ``None``.
            / 与模型关联的 Pinocchio ``Data`` 对象，或 ``None``。
        urdf_path: Filesystem path to the URDF used to build the model.
            / 用于构建模型的 URDF 文件路径。
    """

    pin: object | None
    model: object | None
    data: object | None
    urdf_path: str

    @property
    def has_pinocchio(self) -> bool:
        """Whether a valid Pinocchio model is loaded. / 是否已加载有效的 Pinocchio 模型。"""
        return self.pin is not None and self.model is not None and self.data is not None

    @property
    def nq(self) -> int:
        """Configuration-space dimension. / 配置空间维度。"""
        return int(self.model.nq) if self.model is not None else 0

    @property
    def nv(self) -> int:
        """Tangent-space (velocity) dimension. / 切空间（速度）维度。"""
        return int(self.model.nv) if self.model is not None else 0


def _default_urdf_path() -> str:
    return rebot_arm_robstride().urdf_path


def load_dynamics_robot_model(urdf_path: str | None = None) -> DynamicsRobotModel:
    """Load a dynamics robot model from a URDF file. / 从 URDF 文件加载动力学机器人模型。

    Builds a Pinocchio ``Model`` from the given URDF and creates
    the associated ``Data`` object.  Falls back gracefully when
    Pinocchio is not installed or the URDF does not exist.

    Args:
        urdf_path: Absolute or relative path to the URDF file.
            When ``None``, uses the default robot URDF from the
            built-in profile.
            / URDF 文件的绝对或相对路径。为 ``None`` 时使用内置配置的默认机器人 URDF。

    Returns:
        A ``DynamicsRobotModel`` instance.  If loading fails,
        ``has_pinocchio`` will be ``False``.
        / ``DynamicsRobotModel`` 实例。加载失败时 ``has_pinocchio`` 为 ``False``。
    """
    path = Path(urdf_path or _default_urdf_path())
    if not path.exists():
        return DynamicsRobotModel(pin=None, model=None, data=None, urdf_path=str(path))

    try:
        import pinocchio as pin
    except ImportError:
        return DynamicsRobotModel(pin=None, model=None, data=None, urdf_path=str(path))

    model = pin.buildModelFromUrdf(str(path))
    data = model.createData()
    return DynamicsRobotModel(pin=pin, model=model, data=data, urdf_path=str(path))


def create_data(drm: DynamicsRobotModel):
    """Create a fresh Pinocchio ``Data`` object for the model. / 为模型创建新的 Pinocchio ``Data`` 对象。

    Useful when a separate data buffer is needed for concurrent or
    nested dynamics computations that should not share intermediate
    results with the model's default data.

    Args:
        drm: Dynamics robot model instance.
            / 动力学机器人模型实例。

    Returns:
        A new Pinocchio ``Data`` object, or ``None`` if Pinocchio
        is unavailable.
        / 新的 Pinocchio ``Data`` 对象，若 Pinocchio 不可用则返回 ``None``。
    """
    if not drm.has_pinocchio:
        return None
    return drm.model.createData()


def _fresh_data(drm: DynamicsRobotModel):
    """Return a fresh per-call Data object for thread-safe dynamics.

    Returns ``drm.data`` only if Pinocchio is unavailable (the None
    value is fine for fallback paths).  Otherwise allocates a new
    ``Data`` so that concurrent calls never corrupt each other.
    """
    if not drm.has_pinocchio:
        return drm.data
    return drm.model.createData()


def neutral_configuration(drm: DynamicsRobotModel):
    """Return the neutral (zero) configuration of the robot. / 返回机器人的零位配置。

    Args:
        drm: Dynamics robot model instance.
            / 动力学机器人模型实例。

    Returns:
        Neutral configuration vector of length ``nq``, or an empty
        list when Pinocchio is unavailable.
        / 长度为 ``nq`` 的零位配置向量，若 Pinocchio 不可用则返回空列表。
    """
    if not drm.has_pinocchio:
        return []
    return drm.pin.neutral(drm.model)


def random_configuration(drm: DynamicsRobotModel):
    """Return a random configuration within joint limits. / 返回关节限位内的随机配置。

    Samples uniformly from the configuration space bounded by
    the model's lower and upper position limits.

    Args:
        drm: Dynamics robot model instance.
            / 动力学机器人模型实例。

    Returns:
        Random configuration vector of length ``nq``, or an empty
        list when Pinocchio is unavailable.
        / 长度为 ``nq`` 的随机配置向量，若 Pinocchio 不可用则返回空列表。
    """
    if not drm.has_pinocchio:
        return []
    return drm.pin.randomConfiguration(drm.model)


def set_gravity(drm: DynamicsRobotModel, gravity: Sequence[float]) -> None:
    """Override the gravity vector used by the model. / 覆盖模型使用的重力向量。

    Args:
        drm: Dynamics robot model instance.
            / 动力学机器人模型实例。
        gravity: 3-element sequence ``(gx, gy, gz)`` specifying the
            gravitational acceleration in the world frame.
            Use ``(0, 0, -9.81)`` for standard Earth gravity or
            ``(0, 0, 0)`` for zero gravity.
            / 3 元素序列 ``(gx, gy, gz)``，表示世界坐标系下的重力加速度。
            标准地球重力使用 ``(0, 0, -9.81)``，零重力使用 ``(0, 0, 0)``。

    Raises:
        ValueError: If ``gravity`` does not have exactly 3 elements.
            / 若 ``gravity`` 不是 3 个元素则抛出。
    """
    if not drm.has_pinocchio:
        return
    if len(gravity) != 3:
        raise ValueError(f"gravity must have 3 elements, got {len(gravity)}")
    vec = gravity
    if np is not None:
        vec = np.asarray(gravity, dtype=float)
    drm.model.gravity = drm.pin.Motion(vec)


def get_gravity(drm: DynamicsRobotModel) -> list[float]:
    """Return the current gravity vector of the model. / 返回模型当前的重力向量。

    Args:
        drm: Dynamics robot model instance.
            / 动力学机器人模型实例。

    Returns:
        3-element list ``[gx, gy, gz]``.  Falls back to standard
        Earth gravity ``(0, 0, -9.81)`` when Pinocchio is unavailable.
        / 3 元素列表 ``[gx, gy, gz]``。Pinocchio 不可用时回退到标准地球重力 ``(0, 0, -9.81)``。
    """
    if not drm.has_pinocchio:
        return list(EARTH_GRAVITY)
    g = drm.model.gravity.linear
    return [float(g[0]), float(g[1]), float(g[2])]
