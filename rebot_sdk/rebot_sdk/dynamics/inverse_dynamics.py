from __future__ import annotations

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from .inertia import _as_q, _as_v, _check_q_shape, _check_v_shape
from .robot_model import DynamicsRobotModel


def _as_a(drm: DynamicsRobotModel, a) -> np.ndarray:
    if a is None:
        return np.zeros(drm.nv if drm.has_pinocchio else 0, dtype=float)
    return np.asarray(a, dtype=float)


def _check_a_shape(drm: DynamicsRobotModel, a: np.ndarray, func_name: str) -> None:
    if drm.has_pinocchio and a.shape != (drm.nv,):
        raise ValueError(f"{func_name}: a must be ({drm.nv},), got {a.shape}")


def compute_inverse_dynamics(drm: DynamicsRobotModel, q=None, v=None, a=None, fext=None) -> np.ndarray:
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    aa = _as_a(drm, a)
    if not drm.has_pinocchio:
        n = int(len(qv) or len(vv) or len(aa))
        return np.zeros(n, dtype=float)

    _check_q_shape(drm, qv, "compute_inverse_dynamics")
    _check_v_shape(drm, vv, "compute_inverse_dynamics")
    _check_a_shape(drm, aa, "compute_inverse_dynamics")

    if fext is None:
        fext = [drm.pin.Force.Zero() for _ in range(drm.model.njoints)]
    tau = drm.pin.rnea(drm.model, drm.data, qv, vv, aa, fext)
    return np.asarray(tau, dtype=float)


def compute_generalized_gravity(drm: DynamicsRobotModel, q=None) -> np.ndarray:
    qv = _as_q(drm, q)
    if not drm.has_pinocchio:
        return np.zeros(int(len(qv)), dtype=float)
    _check_q_shape(drm, qv, "compute_generalized_gravity")
    drm.pin.computeGeneralizedGravity(drm.model, drm.data, qv)
    return drm.data.g.copy()


def compute_static_torque(drm: DynamicsRobotModel, q=None, fext=None) -> np.ndarray:
    qv = _as_q(drm, q)
    if not drm.has_pinocchio:
        return np.zeros(int(len(qv)), dtype=float)
    _check_q_shape(drm, qv, "compute_static_torque")
    if fext is None:
        fext = [drm.pin.Force.Zero() for _ in range(drm.model.njoints)]
    drm.pin.computeStaticTorque(drm.model, drm.data, qv, fext)
    return drm.data.tau.copy()


# Backward-compatible alias

def rnea_torque(drm, q, dq, ddq):
    return compute_inverse_dynamics(drm, q=q, v=dq, a=ddq).tolist()
