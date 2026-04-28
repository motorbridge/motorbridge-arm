from __future__ import annotations

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from .inertia import _as_q, _as_v, _check_q_shape, _check_v_shape
from .robot_model import DynamicsRobotModel


def _as_tau(drm: DynamicsRobotModel, tau) -> np.ndarray:
    if tau is None:
        return np.zeros(drm.nv if drm.has_pinocchio else 0, dtype=float)
    return np.asarray(tau, dtype=float)


def _check_tau_shape(drm: DynamicsRobotModel, tau: np.ndarray, func_name: str) -> None:
    if drm.has_pinocchio and tau.shape != (drm.nv,):
        raise ValueError(f"{func_name}: tau must be ({drm.nv},), got {tau.shape}")


def compute_forward_dynamics(drm: DynamicsRobotModel, q=None, v=None, tau=None) -> np.ndarray:
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    tv = _as_tau(drm, tau)
    if not drm.has_pinocchio:
        return np.zeros(int(len(qv) or len(vv) or len(tv)), dtype=float)

    _check_q_shape(drm, qv, "compute_forward_dynamics")
    _check_v_shape(drm, vv, "compute_forward_dynamics")
    _check_tau_shape(drm, tv, "compute_forward_dynamics")

    ddq = drm.pin.aba(drm.model, drm.data, qv, vv, tv)
    return np.asarray(ddq, dtype=float)


def forward_dynamics_from_nle(drm: DynamicsRobotModel, q=None, v=None, tau=None) -> np.ndarray:
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    tv = _as_tau(drm, tau)
    if not drm.has_pinocchio:
        return np.zeros(int(len(qv) or len(vv) or len(tv)), dtype=float)

    _check_q_shape(drm, qv, "forward_dynamics_from_nle")
    _check_v_shape(drm, vv, "forward_dynamics_from_nle")
    _check_tau_shape(drm, tv, "forward_dynamics_from_nle")

    drm.pin.computeAllTerms(drm.model, drm.data, qv, vv)
    return np.linalg.solve(np.asarray(drm.data.M), tv - np.asarray(drm.data.nle))


# Backward-compatible alias

def aba_acceleration(drm, q, dq, tau):
    return compute_forward_dynamics(drm, q=q, v=dq, tau=tau).tolist()
