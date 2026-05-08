"""Joint-space inertia, Coriolis, gravity, and non-linear effects computation.
/ 关节空间惯性矩阵、科氏力、重力及非线性效应计算。
"""

from __future__ import annotations

from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

from .robot_model import DynamicsRobotModel, neutral_configuration, _fresh_data


def _as_q(drm: DynamicsRobotModel, q) -> np.ndarray:
    if np is None:
        if q is None:
            return list(neutral_configuration(drm)) if drm.has_pinocchio else []
        return [float(x) for x in q]
    if q is None:
        if drm.has_pinocchio:
            return np.asarray(neutral_configuration(drm), dtype=float)
        return np.zeros(0, dtype=float)
    return np.asarray(q, dtype=float)


def _as_v(drm: DynamicsRobotModel, v) -> np.ndarray:
    if np is None:
        if v is None:
            return [0.0] * (drm.nv if drm.has_pinocchio else 0)
        return [float(x) for x in v]
    if v is None:
        return np.zeros(drm.nv if drm.has_pinocchio else 0, dtype=float)
    return np.asarray(v, dtype=float)


def _check_q_shape(drm: DynamicsRobotModel, q: np.ndarray, func_name: str) -> None:
    if drm.has_pinocchio and q.shape != (drm.nq,):
        raise ValueError(f"{func_name}: q must be ({drm.nq},), got {q.shape}")


def _check_v_shape(drm: DynamicsRobotModel, v: np.ndarray, func_name: str) -> None:
    if drm.has_pinocchio and v.shape != (drm.nv,):
        raise ValueError(f"{func_name}: v must be ({drm.nv},), got {v.shape}")


def compute_mass_matrix(drm: DynamicsRobotModel, q=None) -> np.ndarray:
    """Compute the joint-space mass (inertia) matrix M(q). / 计算关节空间质量（惯性）矩阵 M(q)。

    Uses the Composite Rigid-Body Algorithm (CRBA) to compute the
    configuration-dependent inertia matrix so that the manipulator
    equation of motion can be written as::

        M(q) * q_ddot + C(q, q_dot) * q_dot + g(q) = tau

    Args:
        drm: Dynamics robot model (holds Pinocchio model + data).
            / 动力学机器人模型（包含 Pinocchio 模型与数据）。
        q: Joint configuration vector of shape ``(nq,)``.
            Defaults to the neutral configuration when ``None``.
            / 关节配置向量，形状为 ``(nq,)``。为 ``None`` 时默认使用零位。

    Returns:
        Mass matrix ``M(q)`` of shape ``(nv, nv)``.  Symmetric and
        positive-definite for a valid configuration.
        / 质量矩阵 ``M(q)``，形状 ``(nv, nv)``。在有效构型下对称正定。
    """
    qv = _as_q(drm, q)
    if not drm.has_pinocchio:
        n = int(len(qv))
        from ._fallback import _eye_2d
        return _eye_2d(n)
    _check_q_shape(drm, qv, "compute_mass_matrix")
    data = _fresh_data(drm)
    drm.pin.crba(drm.model, data, qv)
    return np.asarray(data.M, dtype=float).copy()


def compute_coriolis_matrix(drm: DynamicsRobotModel, q=None, v=None) -> np.ndarray:
    """Compute the Coriolis / centrifugal matrix C(q, v). / 计算科氏力/离心力矩阵 C(q, v)。

    The Coriolis matrix captures velocity-dependent non-linear effects
    so that ``C(q, v) @ v`` equals the combined Coriolis and centrifugal
    wrench projected into joint space::

        C(q, q_dot) * q_dot

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
        Coriolis matrix ``C(q, v)`` of shape ``(nv, nv)``.
        / 科氏力矩阵 ``C(q, v)``，形状 ``(nv, nv)``。
    """
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    if not drm.has_pinocchio:
        n = int(len(qv) or len(vv))
        from ._fallback import _zeros_2d
        return _zeros_2d(n)
    _check_q_shape(drm, qv, "compute_coriolis_matrix")
    _check_v_shape(drm, vv, "compute_coriolis_matrix")
    data = _fresh_data(drm)
    drm.pin.computeCoriolisMatrix(drm.model, data, qv, vv)
    return np.asarray(data.C, dtype=float).copy()


def compute_gravity_vector(drm: DynamicsRobotModel, q=None) -> np.ndarray:
    """Compute the generalized gravity vector g(q). / 计算广义重力向量 g(q)。

    Evaluates the gravity-induced joint torques for the given
    configuration under the robot model's current gravity vector
    (default: standard Earth gravity ``[0, 0, -9.81]``)::

        g(q) = -sum_i ( m_i * J_i(q)^T * g_world )

    Args:
        drm: Dynamics robot model (holds Pinocchio model + data).
            / 动力学机器人模型（包含 Pinocchio 模型与数据）。
        q: Joint configuration vector of shape ``(nq,)``.
            Defaults to the neutral configuration when ``None``.
            / 关节配置向量，形状为 ``(nq,)``。为 ``None`` 时默认使用零位。

    Returns:
        Gravity vector ``g(q)`` of shape ``(nv,)``.
        / 重力向量 ``g(q)``，形状 ``(nv,)``。
    """
    qv = _as_q(drm, q)
    if not drm.has_pinocchio:
        from ._fallback import _zeros_1d
        return _zeros_1d(int(len(qv)))
    _check_q_shape(drm, qv, "compute_gravity_vector")
    data = _fresh_data(drm)
    drm.pin.computeGeneralizedGravity(drm.model, data, qv)
    return np.asarray(data.g, dtype=float).copy()


def compute_nle(drm: DynamicsRobotModel, q=None, v=None) -> np.ndarray:
    """Compute non-linear effects (NLE) vector: nle = C(q,v)*v + g(q). / 计算非线性效应向量：nle = C(q,v)*v + g(q)。

    Combines Coriolis/centrifugal and gravity terms into a single
    vector that appears in the standard manipulator equation::

        M(q) * q_ddot + nle(q, q_dot) = tau

    where ``nle(q, q_dot) = C(q, q_dot) * q_dot + g(q)``.

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
        Non-linear effects vector of shape ``(nv,)``.
        / 非线性效应向量，形状 ``(nv,)``。
    """
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    if not drm.has_pinocchio:
        from ._fallback import _zeros_1d
        return _zeros_1d(int(len(qv) or len(vv)))
    _check_q_shape(drm, qv, "compute_nle")
    _check_v_shape(drm, vv, "compute_nle")
    data = _fresh_data(drm)
    drm.pin.nonLinearEffects(drm.model, data, qv, vv)
    return np.asarray(data.nle, dtype=float).copy()


def compute_all_terms(drm: DynamicsRobotModel, q=None, v=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute M(q), C(q,v), and g(q) in one call. / 一次调用同时计算 M(q)、C(q,v) 和 g(q)。

    Convenience wrapper around Pinocchio's ``computeAllTerms`` which
    evaluates the mass matrix, Coriolis matrix, and gravity vector
    efficiently in a single pass through the kinematic tree.

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
        A tuple ``(M, C, g)`` where:

        - ``M`` – mass matrix of shape ``(nv, nv)``
        - ``C`` – Coriolis matrix of shape ``(nv, nv)``
        - ``g`` – gravity vector of shape ``(nv,)``

        / 元组 ``(M, C, g)``，其中 ``M`` 为质量矩阵 ``(nv, nv)``，
        ``C`` 为科氏力矩阵 ``(nv, nv)``，``g`` 为重力向量 ``(nv,)``。
    """
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    if not drm.has_pinocchio:
        n = int(len(qv) or len(vv))
        from ._fallback import _eye_2d, _zeros_2d, _zeros_1d
        return _eye_2d(n), _zeros_2d(n), _zeros_1d(n)
    _check_q_shape(drm, qv, "compute_all_terms")
    _check_v_shape(drm, vv, "compute_all_terms")
    data = _fresh_data(drm)
    drm.pin.computeAllTerms(drm.model, data, qv, vv)
    return (
        np.asarray(data.M, dtype=float).copy(),
        np.asarray(data.C, dtype=float).copy(),
        np.asarray(data.g, dtype=float).copy(),
    )


# Backward-compatible alias

def mass_matrix(drm: DynamicsRobotModel, q) -> list[list[float]]:
    m = compute_mass_matrix(drm, q)
    return m if isinstance(m, list) else m.tolist()
