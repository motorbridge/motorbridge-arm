from __future__ import annotations

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from .inertia import _as_q, _as_v, _check_q_shape, _check_v_shape
from .inverse_dynamics import _as_a, _check_a_shape
from .robot_model import DynamicsRobotModel


def compute_rnea_derivatives(drm: DynamicsRobotModel, q=None, v=None, a=None):
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    aa = _as_a(drm, a)
    if not drm.has_pinocchio:
        n = int(len(qv) or len(vv) or len(aa))
        if np is None:
            z = [[0.0 for _ in range(n)] for _ in range(n)]
            return z, [row[:] for row in z], [row[:] for row in z]
        z = np.zeros((n, n), dtype=float)
        return z.copy(), z.copy(), z.copy()

    _check_q_shape(drm, qv, "compute_rnea_derivatives")
    _check_v_shape(drm, vv, "compute_rnea_derivatives")
    _check_a_shape(drm, aa, "compute_rnea_derivatives")

    drm.pin.computeRNEADerivatives(drm.model, drm.data, qv, vv, aa)
    return (
        np.asarray(drm.data.dtau_dq, dtype=float).copy(),
        np.asarray(drm.data.dtau_dv, dtype=float).copy(),
        np.asarray(drm.data.M, dtype=float).copy(),
    )


def compute_coriolis_derivatives(drm: DynamicsRobotModel, q=None, v=None):
    qv = _as_q(drm, q)
    vv = _as_v(drm, v)
    if not drm.has_pinocchio:
        n = int(len(qv) or len(vv))
        if np is None:
            z = [[0.0 for _ in range(n)] for _ in range(n)]
            return z, [row[:] for row in z]
        z = np.zeros((n, n), dtype=float)
        return z.copy(), z.copy()

    _check_q_shape(drm, qv, "compute_coriolis_derivatives")
    _check_v_shape(drm, vv, "compute_coriolis_derivatives")
    drm.pin.computeRNEADerivatives(drm.model, drm.data, qv, vv, np.zeros(drm.nv))
    return np.asarray(drm.data.dtau_dq, dtype=float).copy(), np.asarray(drm.data.dtau_dv, dtype=float).copy()


def compute_generalized_gravity_derivatives(drm: DynamicsRobotModel, q=None):
    qv = _as_q(drm, q)
    if not drm.has_pinocchio:
        n = int(len(qv))
        if np is None:
            return [[0.0 for _ in range(n)] for _ in range(n)]
        return np.zeros((n, n), dtype=float)
    _check_q_shape(drm, qv, "compute_generalized_gravity_derivatives")
    drm.pin.computeRNEADerivatives(drm.model, drm.data, qv, np.zeros(drm.nv), np.zeros(drm.nv))
    return np.asarray(drm.data.dtau_dq, dtype=float).copy()


def compute_mass_matrix_derivatives(drm: DynamicsRobotModel, q=None):
    qv = _as_q(drm, q)
    if not drm.has_pinocchio:
        n = int(len(qv))
        if np is None:
            return [[[0.0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
        return np.zeros((n, n, n), dtype=float)
    _check_q_shape(drm, qv, "compute_mass_matrix_derivatives")

    dMdq = np.zeros((drm.nq, drm.nv, drm.nv), dtype=float)
    # Portable fallback: finite difference around CRBA for broad pinocchio version compatibility.
    eps = 1e-6
    for j in range(drm.nq):
        qp = qv.copy()
        qm = qv.copy()
        qp[j] += eps
        qm[j] -= eps
        drm.pin.crba(drm.model, drm.data, qp)
        Mp = np.asarray(drm.data.M, dtype=float).copy()
        drm.pin.crba(drm.model, drm.data, qm)
        Mm = np.asarray(drm.data.M, dtype=float).copy()
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
