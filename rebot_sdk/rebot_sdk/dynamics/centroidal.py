from __future__ import annotations


def centroidal_momentum(drm, q, dq):
    if drm.pin is None:
        return [0.0] * 6
    import numpy as np

    hg = drm.pin.computeCentroidalMomentum(
        drm.model,
        drm.data,
        np.array(q, dtype=float),
        np.array(dq, dtype=float),
    )
    return hg.vector.tolist()
