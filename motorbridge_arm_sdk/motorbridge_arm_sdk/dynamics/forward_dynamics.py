"""Forward dynamics: ABA and explicit M^{-1} methods.
/ 正向动力学：ABA 与显式 M^{-1} 方法。
"""

from __future__ import annotations

try:
    import numpy as np
except ImportError:
    np = None

from .inertia import _as_q, _as_v, _check_q_shape, _check_v_shape
from .robot_model import DynamicsRobotModel


def _as_tau(drm: DynamicsRobotModel, tau) -> np.ndarray:
    if np is None:
        if tau is None:
            return [0.0] * (drm.nv if drm.has_pinocchio else 0)
        return [float(x) for x in tau]
    if tau is None:
        return np.zeros(drm.nv if drm.has_pinocchio else 0, dtype=float)
    return np.asarray(tau, dtype=float)


def _check_tau_shape(drm: DynamicsRobotModel, tau: np.ndarray, func_name: str) -> None:
    if drm.has_pinocchio and tau.shape != (drm.nv,):
        raise ValueError(f"{func_name}: tau must be ({drm.nv},), got {tau.shape}")


def compute_forward_dynamics(drm: DynamicsRobotModel, q=None, v=None, tau=None) -> np.ndarray:
    """Compute forward dynamics via Articulated-Body Algorithm (ABA). / 基于铰接体算法 (ABA) 计算正向动力学。

    Solves for joint accelerations produced by the given state and
    applied torques using the O(n) ABA algorithm::

        q_ddot = ABA(q, q_dot, tau)

    which satisfies::

        M(q) * q_ddot + C(q, q_dot) * q_dot + g(q) = tau

    Args:
        drm: Dynamics robot model (holds Pinocchio model + data).
            / 动力学机器人模型（包含 Pinocchio 模型与数据）。
        q: Joint configuration vector of shape ``(nq,)``.
            Defaults to the neutral configuration when ``None``.
            / 关节配置向量，形状为 ``(nq,)``。为 ``None`` 时默认使用零位。
        v: Joint velocity vector of shape ``(nv,)``.
            Defaults to zero when ``None``.
            / 关节速度向量，形状为 ``(nv,)``。为 ``None`` 时默认为零。
        tau: Applied joint torque vector of shape ``(nv,)``.
            Defaults to zero when ``None``.
            / 施加的关节力矩向量，形状为 ``(nv,)``。为 ``None`` 时默认为零。

    Returns:
        Joint acceleration vector ``q_ddot`` of shape ``(nv,)``.
        / 关节加速度向量 ``q_ddot``，形状 ``(nv,)``。
    """
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    tv = _as_tau(drm, tau)
    if not drm.has_pinocchio:
        from ._fallback import _zeros_1d
        return _zeros_1d(int(len(qv) or len(vv) or len(tv)))

    _check_q_shape(drm, qv, "compute_forward_dynamics")
    _check_v_shape(drm, vv, "compute_forward_dynamics")
    _check_tau_shape(drm, tv, "compute_forward_dynamics")

    ddq = drm.pin.aba(drm.model, drm.data, qv, vv, tv)
    return np.asarray(ddq, dtype=float)


def forward_dynamics_from_nle(drm: DynamicsRobotModel, q=None, v=None, tau=None) -> np.ndarray:
    """Compute forward dynamics via explicit M^{-1} solve. / 通过显式 M^{-1} 求解计算正向动力学。

    Uses the non-linear effects (NLE) formulation::

        q_ddot = M(q)^{-1} * (tau - nle(q, v))

    where ``nle(q, v) = C(q, v) * v + g(q)``.  This is equivalent to
    ABA but solves the mass matrix explicitly, which can be useful for
    inspecting ``M`` or when ``M^{-1}`` is needed separately.

    Args:
        drm: Dynamics robot model (holds Pinocchio model + data).
            / 动力学机器人模型（包含 Pinocchio 模型与数据）。
        q: Joint configuration vector of shape ``(nq,)``.
            Defaults to the neutral configuration when ``None``.
            / 关节配置向量，形状为 ``(nq,)``。为 ``None`` 时默认使用零位。
        v: Joint velocity vector of shape ``(nv,)``.
            Defaults to zero when ``None``.
            / 关节速度向量，形状为 ``(nv,)``。为 ``None`` 时默认为零。
        tau: Applied joint torque vector of shape ``(nv,)``.
            Defaults to zero when ``None``.
            / 施加的关节力矩向量，形状为 ``(nv,)``。为 ``None`` 时默认为零。

    Returns:
        Joint acceleration vector ``q_ddot`` of shape ``(nv,)``.
        / 关节加速度向量 ``q_ddot``，形状 ``(nv,)``。
    """
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    tv = _as_tau(drm, tau)
    if not drm.has_pinocchio:
        from ._fallback import _zeros_1d
        return _zeros_1d(int(len(qv) or len(vv) or len(tv)))

    _check_q_shape(drm, qv, "forward_dynamics_from_nle")
    _check_v_shape(drm, vv, "forward_dynamics_from_nle")
    _check_tau_shape(drm, tv, "forward_dynamics_from_nle")

    drm.pin.computeAllTerms(drm.model, drm.data, qv, vv)
    return np.linalg.solve(np.asarray(drm.data.M), tv - np.asarray(drm.data.nle))


# Backward-compatible alias

def aba_acceleration(drm, q, dq, tau):
    out = compute_forward_dynamics(drm, q=q, v=dq, tau=tau)
    return out if isinstance(out, list) else out.tolist()
