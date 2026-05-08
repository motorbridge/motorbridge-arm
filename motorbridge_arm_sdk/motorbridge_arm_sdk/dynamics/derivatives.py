"""Derivatives of dynamics quantities for sensitivity analysis.
/ 动力学量导数，用于灵敏度分析。
"""

from __future__ import annotations

try:
    import numpy as np
except ImportError:
    np = None

from .inertia import _as_q, _as_v, _check_q_shape, _check_v_shape
from .inverse_dynamics import _as_a, _check_a_shape
from .robot_model import DynamicsRobotModel, _fresh_data


def compute_rnea_derivatives(drm: DynamicsRobotModel, q=None, v=None, a=None):
    """Compute partial derivatives of RNEA (inverse dynamics) w.r.t. q, v, a.
    / 计算 RNEA（逆动力学）对 q、v、a 的偏导数。

    Returns the three Jacobian matrices of the inverse dynamics torques::

        dtau/dq,  dtau/dv,  dtau/da = dtau/dq_ddot

    where ``tau = RNEA(q, v, a)``.  The third matrix ``dtau/da``
    is equal to the mass matrix ``M(q)``.

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

    Returns:
        Tuple ``(dtau_dq, dtau_dv, dtau_da)`` each of shape ``(nv, nv)``.
        / 元组 ``(dtau_dq, dtau_dv, dtau_da)``，每个形状为 ``(nv, nv)``。
    """
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    aa = _as_a(drm, a)
    if not drm.has_pinocchio:
        n = int(len(qv) or len(vv) or len(aa))
        from ._fallback import _zeros_2d
        z = _zeros_2d(n)
        if np is not None:
            z = z.copy()
        return z, z.copy() if np is not None else [row[:] for row in z], z.copy() if np is not None else [row[:] for row in z]

    _check_q_shape(drm, qv, "compute_rnea_derivatives")
    _check_v_shape(drm, vv, "compute_rnea_derivatives")
    _check_a_shape(drm, aa, "compute_rnea_derivatives")

    data = _fresh_data(drm)
    drm.pin.computeRNEADerivatives(drm.model, data, qv, vv, aa)
    return (
        np.asarray(data.dtau_dq, dtype=float).copy(),
        np.asarray(data.dtau_dv, dtype=float).copy(),
        np.asarray(data.M, dtype=float).copy(),
    )


def compute_coriolis_derivatives(drm: DynamicsRobotModel, q=None, v=None):
    """Compute partial derivatives of the Coriolis/centrifugal term.
    / 计算科氏力/离心力项的偏导数。

    Returns the derivatives of the velocity-dependent non-linear effects
    (Coriolis + centrifugal) with respect to ``q`` and ``v``::

        d(C(q,v)*v)/dq,  d(C(q,v)*v)/dv

    Computed by evaluating RNEA derivatives at zero acceleration.

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
        Tuple ``(dc_dq, dc_dv)`` each of shape ``(nv, nv)``.
        / 元组 ``(dc_dq, dc_dv)``，每个形状为 ``(nv, nv)``。
    """
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    if not drm.has_pinocchio:
        n = int(len(qv) or len(vv))
        from ._fallback import _zeros_2d
        z = _zeros_2d(n)
        if np is not None:
            z = z.copy()
        return z, z.copy() if np is not None else [row[:] for row in z]

    _check_q_shape(drm, qv, "compute_coriolis_derivatives")
    _check_v_shape(drm, vv, "compute_coriolis_derivatives")
    data = _fresh_data(drm)
    drm.pin.computeRNEADerivatives(drm.model, data, qv, vv, np.zeros(drm.nv))
    return np.asarray(data.dtau_dq, dtype=float).copy(), np.asarray(data.dtau_dv, dtype=float).copy()


def compute_generalized_gravity_derivatives(drm: DynamicsRobotModel, q=None):
    """Compute the Jacobian of the generalized gravity vector w.r.t. q.
    / 计算广义重力向量对 q 的雅可比矩阵。

    Returns::

        dg(q)/dq

    which is an ``(nv, nv)`` matrix describing how the gravity-induced
    joint torques change with small perturbations in configuration.

    Args:
        drm: Dynamics robot model (holds Pinocchio model + data).
            / 动力学机器人模型（包含 Pinocchio 模型与数据）。
        q: Joint configuration vector of shape ``(nq,)``.
            Defaults to the neutral configuration when ``None``.
            / 关节配置向量，形状为 ``(nq,)``。为 ``None`` 时默认使用零位。

    Returns:
        Gravity Jacobian matrix of shape ``(nv, nv)``.
        / 重力雅可比矩阵，形状 ``(nv, nv)``。
    """
    qv = _as_q(drm, q)
    if not drm.has_pinocchio:
        n = int(len(qv))
        from ._fallback import _zeros_2d
        return _zeros_2d(n)
    _check_q_shape(drm, qv, "compute_generalized_gravity_derivatives")
    data = _fresh_data(drm)
    drm.pin.computeRNEADerivatives(drm.model, data, qv, np.zeros(drm.nv), np.zeros(drm.nv))
    return np.asarray(data.dtau_dq, dtype=float).copy()


def compute_mass_matrix_derivatives(drm: DynamicsRobotModel, q=None):
    """Compute the derivative of the mass matrix w.r.t. q. / 计算质量矩阵对 q 的导数。

    Returns a 3-D tensor::

        dM/dq[j] = dM(q)/dq_j    for j = 0 .. nq-1

    Each slice ``(j, :, :)`` is an ``(nv, nv)`` matrix giving the
    partial derivative of ``M(q)`` with respect to the j-th configuration
    variable.  Uses Pinocchio's analytical ``computeMassMatrixDerivatives``
    when available, falling back to central finite differences of CRBA for
    older Pinocchio versions.

    Args:
        drm: Dynamics robot model (holds Pinocchio model + data).
            / 动力学机器人模型（包含 Pinocchio 模型与数据）。
        q: Joint configuration vector of shape ``(nq,)``.
            Defaults to the neutral configuration when ``None``.
            / 关节配置向量，形状为 ``(nq,)``。为 ``None`` 时默认使用零位。

    Returns:
        Mass matrix derivative tensor of shape ``(nq, nv, nv)``.
        / 质量矩阵导数张量，形状 ``(nq, nv, nv)``。
    """
    qv = _as_q(drm, q)
    if not drm.has_pinocchio:
        n = int(len(qv))
        if np is None:
            return [[[0.0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
        return np.zeros((n, n, n), dtype=float)
    _check_q_shape(drm, qv, "compute_mass_matrix_derivatives")

    # Try analytical method first (more accurate).
    try:
        data = _fresh_data(drm)
        drm.pin.computeAllTerms(drm.model, data, qv, np.zeros(drm.nv))
        drm.pin.computeMassMatrixDerivatives(drm.model, data, qv)
        dMdq = np.zeros((drm.nq, drm.nv, drm.nv), dtype=float)
        for j in range(drm.nq):
            dMdq[j] = np.asarray(data.dMdq[j], dtype=float).copy()
        return dMdq
    except (AttributeError, TypeError):
        pass

    # Finite-difference fallback for older Pinocchio versions.
    dMdq = np.zeros((drm.nq, drm.nv, drm.nv), dtype=float)
    eps = 1e-6
    for j in range(drm.nq):
        qp = qv.copy()
        qm = qv.copy()
        qp[j] += eps
        qm[j] -= eps
        fd1 = _fresh_data(drm)
        drm.pin.crba(drm.model, fd1, qp)
        Mp = np.asarray(fd1.M, dtype=float).copy()
        fd2 = _fresh_data(drm)
        drm.pin.crba(drm.model, fd2, qm)
        Mm = np.asarray(fd2.M, dtype=float).copy()
        dMdq[j] = (Mp - Mm) / (2.0 * eps)
    return dMdq


# Backward-compatible alias

def inverse_dynamics_derivatives(drm, q, dq, ddq):
    dtau_dq, dtau_dv, dtau_da = compute_rnea_derivatives(drm, q=q, v=dq, a=ddq)
    return {
        "dtau_dq": dtau_dq if isinstance(dtau_dq, list) else dtau_dq.tolist(),
        "dtau_ddq": dtau_dv if isinstance(dtau_dv, list) else dtau_dv.tolist(),
        "dtau_dddq": dtau_da if isinstance(dtau_da, list) else dtau_da.tolist(),
    }
