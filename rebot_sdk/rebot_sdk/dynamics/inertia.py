from __future__ import annotations


def mass_matrix(drm, q):
    if drm.pin is None:
        n = len(q)
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    import numpy as np

    qv = np.array(q, dtype=float)
    M = drm.pin.crba(drm.model, drm.data, qv)
    return M.tolist()
