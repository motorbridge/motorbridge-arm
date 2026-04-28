from __future__ import annotations

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from .inertia import _as_q, _as_v, _check_q_shape, _check_v_shape
from .robot_model import DynamicsRobotModel


def compute_kinetic_energy(drm: DynamicsRobotModel, q=None, v=None) -> float:
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    if not drm.has_pinocchio:
        return 0.0
    _check_q_shape(drm, qv, "compute_kinetic_energy")
    _check_v_shape(drm, vv, "compute_kinetic_energy")
    val = drm.pin.computeKineticEnergy(drm.model, drm.data, qv, vv)
    return float(val)


def compute_potential_energy(drm: DynamicsRobotModel, q=None) -> float:
    qv = _as_q(drm, q)
    if not drm.has_pinocchio:
        return 0.0
    _check_q_shape(drm, qv, "compute_potential_energy")
    val = drm.pin.computePotentialEnergy(drm.model, drm.data, qv)
    return float(val)


def compute_total_energy(drm: DynamicsRobotModel, q=None, v=None) -> float:
    return compute_kinetic_energy(drm, q=q, v=v) + compute_potential_energy(drm, q=q)


# Backward-compatible aliases

def kinetic_energy(drm, q, dq) -> float:
    return compute_kinetic_energy(drm, q=q, v=dq)


def potential_energy(drm, q) -> float:
    return compute_potential_energy(drm, q=q)


def total_energy(drm, q, dq) -> float:
    return compute_total_energy(drm, q=q, v=dq)
