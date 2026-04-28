from __future__ import annotations


def rnea_torque(drm, q, dq, ddq):
    if drm.pin is None:
        return [0.0 for _ in q]
    import numpy as np

    tau = drm.pin.rnea(
        drm.model,
        drm.data,
        np.array(q, dtype=float),
        np.array(dq, dtype=float),
        np.array(ddq, dtype=float),
    )
    return tau.tolist()
