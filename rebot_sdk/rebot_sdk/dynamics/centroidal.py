from __future__ import annotations

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from .inertia import _as_q, _as_v, _check_q_shape, _check_v_shape
from .robot_model import DynamicsRobotModel


def compute_center_of_mass(drm: DynamicsRobotModel, q=None, center_zero: bool = False) -> np.ndarray:
    qv = _as_q(drm, q)
    if not drm.has_pinocchio:
        return np.zeros(3, dtype=float)
    _check_q_shape(drm, qv, "compute_center_of_mass")
    if center_zero:
        drm.pin.centerOfMass(drm.model, drm.data, qv, False)
    else:
        drm.pin.centerOfMass(drm.model, drm.data, qv)
    return np.asarray(drm.data.com[0], dtype=float).copy()


def compute_com_velocity(drm: DynamicsRobotModel, q=None, v=None) -> np.ndarray:
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    if not drm.has_pinocchio:
        return np.zeros(3, dtype=float)
    _check_q_shape(drm, qv, "compute_com_velocity")
    _check_v_shape(drm, vv, "compute_com_velocity")
    drm.pin.computeCentroidalVelocities(drm.model, drm.data, qv, vv)
    return np.asarray(drm.data.vcom[0], dtype=float).copy()


def compute_centroidal_momentum(drm: DynamicsRobotModel, q=None, v=None) -> np.ndarray:
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    if not drm.has_pinocchio:
        return np.zeros(6, dtype=float)
    _check_q_shape(drm, qv, "compute_centroidal_momentum")
    _check_v_shape(drm, vv, "compute_centroidal_momentum")
    drm.pin.ccrba(drm.model, drm.data, qv, vv)
    return np.asarray(drm.data.hg.vector, dtype=float).copy()


def compute_centroidal_matrix(drm: DynamicsRobotModel, q=None, v=None) -> np.ndarray:
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    if not drm.has_pinocchio:
        return np.zeros((6, int(len(qv) or len(vv))), dtype=float)
    _check_q_shape(drm, qv, "compute_centroidal_matrix")
    _check_v_shape(drm, vv, "compute_centroidal_matrix")
    drm.pin.ccrba(drm.model, drm.data, qv, vv)
    return np.asarray(drm.data.Ag, dtype=float).copy()


# Backward-compatible alias

def centroidal_momentum(drm, q, dq):
    return compute_centroidal_momentum(drm, q=q, v=dq).tolist()
