from __future__ import annotations


def kinetic_energy(drm, q, dq) -> float:
    if drm.pin is None:
        return 0.0
    import numpy as np

    return float(drm.pin.computeKineticEnergy(drm.model, drm.data, np.array(q, dtype=float), np.array(dq, dtype=float)))


def potential_energy(drm, q) -> float:
    if drm.pin is None:
        return 0.0
    import numpy as np

    return float(drm.pin.computePotentialEnergy(drm.model, drm.data, np.array(q, dtype=float)))


def total_energy(drm, q, dq) -> float:
    return kinetic_energy(drm, q, dq) + potential_energy(drm, q)
