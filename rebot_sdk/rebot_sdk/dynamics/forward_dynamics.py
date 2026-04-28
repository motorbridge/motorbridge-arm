from __future__ import annotations


def aba_acceleration(drm, q, dq, tau):
    if drm.pin is None:
        return [0.0 for _ in q]
    import numpy as np

    ddq = drm.pin.aba(
        drm.model,
        drm.data,
        np.array(q, dtype=float),
        np.array(dq, dtype=float),
        np.array(tau, dtype=float),
    )
    return ddq.tolist()
