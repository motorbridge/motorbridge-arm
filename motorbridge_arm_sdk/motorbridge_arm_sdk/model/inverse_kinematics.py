from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(slots=True)
class IKParams:
    max_iter: int = 1000
    tolerance: float = 1e-4
    step_size: float = 0.6
    damping: float = 1e-6
    line_search_steps: int = 4
    retry_count: int = 8
    null_gain: float = 0.0


@dataclass(slots=True)
class IKResult:
    q: list[float]
    success: bool
    error: float
    iterations: int


def clamp_config_safe(model, q):
    """Clamp configuration to model limits with NaN protection.

    - NaN limits are replaced by 0.0
    - NaN q values are replaced by 0.0
    """
    out = [float(v) if math.isfinite(float(v)) else 0.0 for v in q]
    lo = [float(x) if math.isfinite(float(x)) else 0.0 for x in model.lowerPositionLimit]
    hi = [float(x) if math.isfinite(float(x)) else 0.0 for x in model.upperPositionLimit]
    n = min(len(out), len(lo), len(hi))
    for i in range(n):
        if lo[i] <= hi[i]:
            if out[i] < lo[i]:
                out[i] = lo[i]
            elif out[i] > hi[i]:
                out[i] = hi[i]
    return out


def solve_ik_advanced(pin, model, data, frame_id: int, target_se3, q_seed: list[float], params: IKParams | None = None) -> IKResult:
    if params is None:
        params = IKParams()
    import numpy as np

    q0 = np.zeros(model.nq)
    n = min(model.nq, len(q_seed))
    q0[:n] = np.array(q_seed[:n], dtype=float)

    def _clamp(q):
        return np.array(clamp_config_safe(model, q), dtype=float)

    def _solve_once(q_init):
        q = q_init.copy()
        best_err = math.inf
        best_q = q.copy()
        for it in range(params.max_iter):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            T_cur = data.oMf[frame_id]
            err = pin.log6(T_cur.inverse() * target_se3).vector
            err_norm = float(np.linalg.norm(err))
            if err_norm < best_err:
                best_err = err_norm
                best_q = q.copy()
            if err_norm < params.tolerance:
                return IKResult(q=[float(v) for v in q], success=True, error=err_norm, iterations=it)

            pin.computeJointJacobians(model, data, q)
            J = pin.getFrameJacobian(model, data, frame_id, pin.ReferenceFrame.LOCAL)
            lam = params.damping * max(1.0, err_norm * 10.0)
            JJT = J @ J.T
            JJT[np.diag_indices_from(JJT)] += lam
            dq = params.step_size * J.T @ np.linalg.solve(JJT, err)

            alpha = 1.0
            accepted = False
            for _ in range(params.line_search_steps):
                q_new = _clamp(pin.integrate(model, q, alpha * dq))
                pin.forwardKinematics(model, data, q_new)
                pin.updateFramePlacements(model, data)
                e_new = pin.log6(data.oMf[frame_id].inverse() * target_se3).vector
                new_norm = float(np.linalg.norm(e_new))
                if new_norm < err_norm:
                    q = q_new
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                q = _clamp(pin.integrate(model, q, 0.1 * dq))
        return IKResult(q=[float(v) for v in best_q], success=False, error=best_err, iterations=params.max_iter)

    best = _solve_once(q0)
    if best.success:
        return best

    lo = [float(x) if math.isfinite(float(x)) else -math.pi for x in model.lowerPositionLimit]
    hi = [float(x) if math.isfinite(float(x)) else math.pi for x in model.upperPositionLimit]
    for _ in range(params.retry_count):
        q_rand = [random.uniform(lo[i], hi[i]) for i in range(model.nq)]
        r = _solve_once(np.array(q_rand, dtype=float))
        if r.error < best.error:
            best = r
        if best.success:
            break
    return best
