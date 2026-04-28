from __future__ import annotations

from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from .robot_model import DynamicsRobotModel, neutral_configuration


def _as_q(drm: DynamicsRobotModel, q) -> np.ndarray:
    if q is None:
        if drm.has_pinocchio:
            return np.asarray(neutral_configuration(drm), dtype=float)
        return np.zeros(0, dtype=float)
    return np.asarray(q, dtype=float)


def _as_v(drm: DynamicsRobotModel, v) -> np.ndarray:
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
    qv = _as_q(drm, q)
    if not drm.has_pinocchio:
        n = int(len(qv))
        return np.eye(n, dtype=float)
    _check_q_shape(drm, qv, "compute_mass_matrix")
    drm.pin.crba(drm.model, drm.data, qv)
    return drm.data.M.copy()


def compute_coriolis_matrix(drm: DynamicsRobotModel, q=None, v=None) -> np.ndarray:
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    if not drm.has_pinocchio:
        n = int(len(qv) or len(vv))
        return np.zeros((n, n), dtype=float)
    _check_q_shape(drm, qv, "compute_coriolis_matrix")
    _check_v_shape(drm, vv, "compute_coriolis_matrix")
    drm.pin.computeCoriolisMatrix(drm.model, drm.data, qv, vv)
    return drm.data.C.copy()


def compute_gravity_vector(drm: DynamicsRobotModel, q=None) -> np.ndarray:
    qv = _as_q(drm, q)
    if not drm.has_pinocchio:
        return np.zeros(int(len(qv)), dtype=float)
    _check_q_shape(drm, qv, "compute_gravity_vector")
    drm.pin.computeGeneralizedGravity(drm.model, drm.data, qv)
    return drm.data.g.copy()


def compute_nle(drm: DynamicsRobotModel, q=None, v=None) -> np.ndarray:
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    if not drm.has_pinocchio:
        return np.zeros(int(len(qv) or len(vv)), dtype=float)
    _check_q_shape(drm, qv, "compute_nle")
    _check_v_shape(drm, vv, "compute_nle")
    drm.pin.nonLinearEffects(drm.model, drm.data, qv, vv)
    return drm.data.nle.copy()


def compute_all_terms(drm: DynamicsRobotModel, q=None, v=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    if not drm.has_pinocchio:
        n = int(len(qv) or len(vv))
        return np.eye(n, dtype=float), np.zeros((n, n), dtype=float), np.zeros(n, dtype=float)
    _check_q_shape(drm, qv, "compute_all_terms")
    _check_v_shape(drm, vv, "compute_all_terms")
    drm.pin.computeAllTerms(drm.model, drm.data, qv, vv)
    return drm.data.M.copy(), drm.data.C.copy(), drm.data.g.copy()


# Backward-compatible alias

def mass_matrix(drm: DynamicsRobotModel, q) -> list[list[float]]:
    return compute_mass_matrix(drm, q).tolist()
