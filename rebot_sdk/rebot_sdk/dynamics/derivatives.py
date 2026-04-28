from __future__ import annotations


def inverse_dynamics_derivatives(drm, q, dq, ddq):
    if drm.pin is None:
        n = len(q)
        z = [[0.0 for _ in range(n)] for _ in range(n)]
        return {"dtau_dq": z, "dtau_ddq": z, "dtau_dddq": z}
    import numpy as np

    qv = np.array(q, dtype=float)
    dqv = np.array(dq, dtype=float)
    ddqv = np.array(ddq, dtype=float)
    drm.pin.computeRNEADerivatives(drm.model, drm.data, qv, dqv, ddqv)
    return {
        "dtau_dq": drm.data.dtau_dq.tolist(),
        "dtau_ddq": drm.data.dtau_dv.tolist(),
        "dtau_dddq": drm.data.M.tolist(),
    }
