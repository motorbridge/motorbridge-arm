"""Kinetic, potential, and total mechanical energy.
/ 动能、势能和总机械能。
"""

from __future__ import annotations

try:
    import numpy as np
except ImportError:
    np = None

from .inertia import _as_q, _as_v, _check_q_shape, _check_v_shape
from .robot_model import DynamicsRobotModel, _fresh_data


def compute_kinetic_energy(drm: DynamicsRobotModel, q=None, v=None) -> float:
    """Compute the kinetic energy of the robot. / 计算机器人动能。

    Evaluates::

        K(q, v) = 0.5 * v^T * M(q) * v

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
        Kinetic energy in joules (``float``).  Returns ``0.0`` when
        Pinocchio is unavailable or velocities are zero.
        / 动能（焦耳，``float``）。Pinocchio 不可用或速度为零时返回 ``0.0``。
    """
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    if not drm.has_pinocchio:
        return 0.0
    _check_q_shape(drm, qv, "compute_kinetic_energy")
    _check_v_shape(drm, vv, "compute_kinetic_energy")
    data = _fresh_data(drm)
    val = drm.pin.computeKineticEnergy(drm.model, data, qv, vv)
    return float(val)


def compute_potential_energy(drm: DynamicsRobotModel, q=None) -> float:
    """Compute the gravitational potential energy of the robot. / 计算机器人重力势能。

    Evaluates::

        U(q) = -sum_i ( m_i * g^T * com_i(q) ) + const

    The reference height is chosen so that the potential energy is
    consistent with the model's gravity vector.

    Args:
        drm: Dynamics robot model (holds Pinocchio model + data).
            / 动力学机器人模型（包含 Pinocchio 模型与数据）。
        q: Joint configuration vector of shape ``(nq,)``.
            Defaults to the neutral configuration when ``None``.
            / 关节配置向量，形状为 ``(nq,)``。为 ``None`` 时默认使用零位。

    Returns:
        Potential energy in joules (``float``).  Returns ``0.0`` when
        Pinocchio is unavailable.
        / 势能（焦耳，``float``）。Pinocchio 不可用时返回 ``0.0``。
    """
    qv = _as_q(drm, q)
    if not drm.has_pinocchio:
        return 0.0
    _check_q_shape(drm, qv, "compute_potential_energy")
    data = _fresh_data(drm)
    val = drm.pin.computePotentialEnergy(drm.model, data, qv)
    return float(val)


def compute_total_energy(drm: DynamicsRobotModel, q=None, v=None) -> float:
    """Compute the total mechanical energy (kinetic + potential). / 计算总机械能（动能 + 势能）。

    Evaluates::

        E(q, v) = K(q, v) + U(q)

    For a conservative system (no dissipation), ``E`` remains constant.

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
        Total mechanical energy in joules (``float``).
        / 总机械能（焦耳，``float``）。
    """
    return compute_kinetic_energy(drm, q=q, v=v) + compute_potential_energy(drm, q=q)


# Backward-compatible aliases

def kinetic_energy(drm, q, dq) -> float:
    return compute_kinetic_energy(drm, q=q, v=dq)


def potential_energy(drm, q) -> float:
    return compute_potential_energy(drm, q=q)


def total_energy(drm, q, dq) -> float:
    return compute_total_energy(drm, q=q, v=dq)
