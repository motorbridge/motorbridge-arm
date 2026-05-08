"""Inverse dynamics: RNEA, generalized gravity, static torque.
/ 逆动力学：RNEA、广义重力、静力矩。
"""

from __future__ import annotations

try:
    import numpy as np
except ImportError:
    np = None

from .inertia import _as_q, _as_v, _check_q_shape, _check_v_shape
from .robot_model import DynamicsRobotModel, _fresh_data


def _as_a(drm: DynamicsRobotModel, a) -> np.ndarray:
    if np is None:
        if a is None:
            return [0.0] * (drm.nv if drm.has_pinocchio else 0)
        return [float(x) for x in a]
    if a is None:
        return np.zeros(drm.nv if drm.has_pinocchio else 0, dtype=float)
    return np.asarray(a, dtype=float)


def _check_a_shape(drm: DynamicsRobotModel, a: np.ndarray, func_name: str) -> None:
    if drm.has_pinocchio and a.shape != (drm.nv,):
        raise ValueError(f"{func_name}: a must be ({drm.nv},), got {a.shape}")


def compute_inverse_dynamics(drm: DynamicsRobotModel, q=None, v=None, a=None, fext=None) -> np.ndarray:
    """Compute inverse dynamics via Recursive Newton-Euler Algorithm (RNEA). / 基于递归牛顿-欧拉算法 (RNEA) 计算逆动力学。

    Solves for the joint torques required to produce a desired
    acceleration given the current state::

        tau = M(q) * a + C(q, v) * v + g(q) - J^T * f_ext

    Args:
        drm: Dynamics robot model (holds Pinocchio model + data).
            / 动力学机器人模型（包含 Pinocchio 模型与数据）。
        q: Joint configuration vector of shape ``(nq,)``.
            Defaults to the neutral configuration when ``None``.
            / 关节配置向量，形状为 ``(nq,)``。为 ``None`` 时默认使用零位。
        v: Joint velocity vector of shape ``(nv,)``.
            Defaults to zero when ``None``.
            / 关节速度向量，形状为 ``(nv,)``。为 ``None`` 时默认为零。
        a: Joint acceleration vector of shape ``(nv,)``.
            Defaults to zero when ``None``.
            / 关节加速度向量，形状为 ``(nv,)``。为 ``None`` 时默认为零。
        fext: Optional list of external wrenches, one per joint.
            Each element should be a Pinocchio ``Force`` object.
            Defaults to zero wrenches when ``None``.
            / 可选的外部力旋量列表，每个关节一个。每个元素应为 Pinocchio ``Force`` 对象。
            为 ``None`` 时默认零力旋量。

    Returns:
        Joint torque vector ``tau`` of shape ``(nv,)``.
        / 关节力矩向量 ``tau``，形状 ``(nv,)``。
    """
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    aa = _as_a(drm, a)
    if not drm.has_pinocchio:
        from ._fallback import _zeros_1d
        return _zeros_1d(int(len(qv) or len(vv) or len(aa)))

    _check_q_shape(drm, qv, "compute_inverse_dynamics")
    _check_v_shape(drm, vv, "compute_inverse_dynamics")
    _check_a_shape(drm, aa, "compute_inverse_dynamics")

    if fext is None:
        fext = [drm.pin.Force.Zero() for _ in range(drm.model.njoints)]
    data = _fresh_data(drm)
    tau = drm.pin.rnea(drm.model, data, qv, vv, aa, fext)
    return np.asarray(tau, dtype=float)


def compute_generalized_gravity(drm: DynamicsRobotModel, q=None) -> np.ndarray:
    """Compute the generalized gravity vector g(q). / 计算广义重力向量 g(q)。

    Evaluates the joint torques needed to hold the robot stationary
    against gravity at the given configuration::

        tau_gravity = g(q)

    Equivalent to RNEA with zero velocity and zero acceleration.

    Args:
        drm: Dynamics robot model (holds Pinocchio model + data).
            / 动力学机器人模型（包含 Pinocchio 模型与数据）。
        q: Joint configuration vector of shape ``(nq,)``.
            Defaults to the neutral configuration when ``None``.
            / 关节配置向量，形状为 ``(nq,)``。为 ``None`` 时默认使用零位。

    Returns:
        Generalized gravity vector of shape ``(nv,)``.
        / 广义重力向量，形状 ``(nv,)``。
    """
    qv = _as_q(drm, q)
    if not drm.has_pinocchio:
        from ._fallback import _zeros_1d
        return _zeros_1d(int(len(qv)))
    _check_q_shape(drm, qv, "compute_generalized_gravity")
    data = _fresh_data(drm)
    drm.pin.computeGeneralizedGravity(drm.model, data, qv)
    return data.g.copy()


def compute_static_torque(drm: DynamicsRobotModel, q=None, fext=None) -> np.ndarray:
    """Compute static (gravitational + external) torque. / 计算静力矩（重力 + 外力）。

    Returns the joint torques required to maintain static equilibrium
    under gravity and any external wrenches::

        tau_static = g(q) - sum( J_i^T * f_ext_i )

    Args:
        drm: Dynamics robot model (holds Pinocchio model + data).
            / 动力学机器人模型（包含 Pinocchio 模型与数据）。
        q: Joint configuration vector of shape ``(nq,)``.
            Defaults to the neutral configuration when ``None``.
            / 关节配置向量，形状为 ``(nq,)``。为 ``None`` 时默认使用零位。
        fext: Optional list of external wrenches, one per joint.
            Each element should be a Pinocchio ``Force`` object.
            Defaults to zero wrenches when ``None``.
            / 可选的外部力旋量列表，每个关节一个。每个元素应为 Pinocchio ``Force`` 对象。
            为 ``None`` 时默认零力旋量。

    Returns:
        Static torque vector of shape ``(nv,)``.
        / 静力矩向量，形状 ``(nv,)``。
    """
    qv = _as_q(drm, q)
    if not drm.has_pinocchio:
        from ._fallback import _zeros_1d
        return _zeros_1d(int(len(qv)))
    _check_q_shape(drm, qv, "compute_static_torque")
    if fext is None:
        fext = [drm.pin.Force.Zero() for _ in range(drm.model.njoints)]
    data = _fresh_data(drm)
    drm.pin.computeStaticTorque(drm.model, data, qv, fext)
    return data.tau.copy()


# Backward-compatible alias

def rnea_torque(drm, q, dq, ddq):
    out = compute_inverse_dynamics(drm, q=q, v=dq, a=ddq)
    return out if isinstance(out, list) else out.tolist()
