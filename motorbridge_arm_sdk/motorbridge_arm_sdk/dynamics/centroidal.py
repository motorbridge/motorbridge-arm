"""Centroidal dynamics: CoM, momentum, centroidal matrix.
/ 质心动力学：质心位置、动量、质心矩阵。
"""

from __future__ import annotations

try:
    import numpy as np
except ImportError:
    np = None

from .inertia import _as_q, _as_v, _check_q_shape, _check_v_shape
from .robot_model import DynamicsRobotModel


def compute_center_of_mass(drm: DynamicsRobotModel, q=None, center_zero: bool = False) -> np.ndarray:
    """Compute the center of mass (CoM) position. / 计算质心 (CoM) 位置。

    Returns the 3-D position of the robot's center of mass expressed
    in the world frame for the given configuration.

    Args:
        drm: Dynamics robot model (holds Pinocchio model + data).
            / 动力学机器人模型（包含 Pinocchio 模型与数据）。
        q: Joint configuration vector of shape ``(nq,)``.
            Defaults to the neutral configuration when ``None``.
            / 关节配置向量，形状为 ``(nq,)``。为 ``None`` 时默认使用零位。
        center_zero: If ``True``, also compute CoM for subtree rooted
            at joint 0 (internal Pinocchio flag).  Default ``False``.
            / 若为 ``True``，同时计算关节 0 子树的质心（Pinocchio 内部标志）。默认 ``False``。

    Returns:
        CoM position vector of shape ``(3,)`` in world coordinates.
        / 世界坐标系下的质心位置向量，形状 ``(3,)``。
    """
    qv = _as_q(drm, q)
    if not drm.has_pinocchio:
        from ._fallback import _zeros_1d
        return _zeros_1d(3)
    _check_q_shape(drm, qv, "compute_center_of_mass")
    if center_zero:
        drm.pin.centerOfMass(drm.model, drm.data, qv, False)
    else:
        drm.pin.centerOfMass(drm.model, drm.data, qv)
    return np.asarray(drm.data.com[0], dtype=float).copy()


def compute_com_velocity(drm: DynamicsRobotModel, q=None, v=None) -> np.ndarray:
    """Compute the center of mass linear velocity. / 计算质心线速度。

    Returns the linear velocity of the robot's center of mass expressed
    in the world frame.

    Args:
        drm: Dynamics robot model (holds Pinocchio model + data).
            / 动力学机器人模型（包含 Pinocchio 模型与数据）。
        q: Joint configuration vector of shape ``(nq,)``.
            Defaults to the neutral configuration when ``None``.
            / 关节配置向量，形状为 ``(nq,)``。为 ``None`` 时默认使用零位。
        v: Joint velocity vector of shape ``(nv,)``.
            Defaults to zero when ``None``.
            / 关节速度向量，形状为 ``(nv,)``。为 ``None`` 时默认为零。

    Returns:
        CoM linear velocity vector of shape ``(3,)`` in world coordinates.
        / 世界坐标系下的质心线速度向量，形状 ``(3,)``。
    """
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    if not drm.has_pinocchio:
        from ._fallback import _zeros_1d
        return _zeros_1d(3)
    _check_q_shape(drm, qv, "compute_com_velocity")
    _check_v_shape(drm, vv, "compute_com_velocity")
    drm.pin.computeCentroidalVelocities(drm.model, drm.data, qv, vv)
    return np.asarray(drm.data.vcom[0], dtype=float).copy()


def compute_centroidal_momentum(drm: DynamicsRobotModel, q=None, v=None) -> np.ndarray:
    """Compute the centroidal momentum vector h_g. / 计算质心动量向量 h_g。

    Returns the 6-D centroidal momentum (linear + angular) stacked
    into a single vector::

        h_g = A_g(q) * q_dot

    where ``A_g(q)`` is the centroidal momentum matrix.

    Args:
        drm: Dynamics robot model (holds Pinocchio model + data).
            / 动力学机器人模型（包含 Pinocchio 模型与数据）。
        q: Joint configuration vector of shape ``(nq,)``.
            Defaults to the neutral configuration when ``None``.
            / 关节配置向量，形状为 ``(nq,)``。为 ``None`` 时默认使用零位。
        v: Joint velocity vector of shape ``(nv,)``.
            Defaults to zero when ``None``.
            / 关节速度向量，形状为 ``(nv,)``。为 ``None`` 时默认为零。

    Returns:
        Centroidal momentum vector of shape ``(6,)`` — linear part
        followed by angular part.
        / 质心动量向量，形状 ``(6,)``——线动量在前，角动量在后。
    """
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    if not drm.has_pinocchio:
        from ._fallback import _zeros_1d
        return _zeros_1d(6)
    _check_q_shape(drm, qv, "compute_centroidal_momentum")
    _check_v_shape(drm, vv, "compute_centroidal_momentum")
    drm.pin.ccrba(drm.model, drm.data, qv, vv)
    return np.asarray(drm.data.hg.vector, dtype=float).copy()


def compute_centroidal_matrix(drm: DynamicsRobotModel, q=None, v=None) -> np.ndarray:
    """Compute the centroidal momentum matrix A_g(q). / 计算质心动量矩阵 A_g(q)。

    Returns the 6 x nv matrix that maps joint velocities to centroidal
    momentum::

        h_g = A_g(q) * q_dot

    Computed via the Composite Centroidal Rigid-Body Algorithm (CCRBA).

    Args:
        drm: Dynamics robot model (holds Pinocchio model + data).
            / 动力学机器人模型（包含 Pinocchio 模型与数据）。
        q: Joint configuration vector of shape ``(nq,)``.
            Defaults to the neutral configuration when ``None``.
            / 关节配置向量，形状为 ``(nq,)``。为 ``None`` 时默认使用零位。
        v: Joint velocity vector of shape ``(nv,)``.
            Required internally by CCRBA but the matrix depends only on
            ``q``.  Defaults to zero when ``None``.
            / CCRBA 内部需要但矩阵仅依赖于 ``q``。为 ``None`` 时默认为零。

    Returns:
        Centroidal momentum matrix ``A_g`` of shape ``(6, nv)``.
        / 质心动量矩阵 ``A_g``，形状 ``(6, nv)``。
    """
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    if not drm.has_pinocchio:
        ncols = int(len(qv) or len(vv))
        if np is None:
            return [[0.0 for _ in range(ncols)] for _ in range(6)]
        return np.zeros((6, ncols), dtype=float)
    _check_q_shape(drm, qv, "compute_centroidal_matrix")
    _check_v_shape(drm, vv, "compute_centroidal_matrix")
    drm.pin.ccrba(drm.model, drm.data, qv, vv)
    return np.asarray(drm.data.Ag, dtype=float).copy()


# Backward-compatible alias

def centroidal_momentum(drm, q, dq):
    out = compute_centroidal_momentum(drm, q=q, v=dq)
    return out if isinstance(out, list) else out.tolist()
