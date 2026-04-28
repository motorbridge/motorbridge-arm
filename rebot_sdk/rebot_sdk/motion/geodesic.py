from __future__ import annotations

from dataclasses import dataclass

from ..types import Pose6D
from .planner import interpolate_pose_geodesic
from .stats import TrajectoryStats, compute_trajectory_stats


@dataclass(slots=True)
class CliKParams:
    max_iter: int = 200
    tolerance: float = 1e-4
    damping: float = 1e-6
    step_size: float = 0.8
    null_gain: float = 0.1


@dataclass(slots=True)
class JointTrajectoryPoint:
    time: float
    q: list[float]
    ik_success: bool


def plan_se3_geodesic(start: Pose6D, end: Pose6D, duration_s: float, dt_s: float = 0.02) -> list[Pose6D]:
    n = max(2, int(duration_s / max(dt_s, 1e-4)) + 1)
    return interpolate_pose_geodesic(start, end, n, profile="geodesic")


def track_with_clik(model, end_frame_id: int, poses: list[Pose6D], q0: list[float], kin, params: CliKParams | None = None) -> list[JointTrajectoryPoint]:
    if params is None:
        params = CliKParams()
    out: list[JointTrajectoryPoint] = []
    q = list(q0)
    if len(poses) <= 1:
        return [JointTrajectoryPoint(time=0.0, q=q, ik_success=True)]
    dt = 1.0 / (len(poses) - 1)

    pin = getattr(kin, "_pin", None)
    if pin is None or model is None:
        # fallback path: per-waypoint IK
        for i, p in enumerate(poses):
            qn = kin.inverse(p, q)
            ok = len(qn) == len(q)
            q = list(qn)
            out.append(JointTrajectoryPoint(time=i * dt, q=list(q), ik_success=ok))
        return out

    import numpy as np

    data = model.createData()
    qv = np.array(q, dtype=float)
    nq = model.nq
    if len(qv) < nq:
        qq = np.zeros(nq, dtype=float)
        qq[: len(qv)] = qv
        qv = qq

    def _joint_limit_grad(model_, q_):
        lo = np.array([float(x) for x in model_.lowerPositionLimit])
        hi = np.array([float(x) for x in model_.upperPositionLimit])
        valid = np.isfinite(lo) & np.isfinite(hi)
        dl = q_ - lo
        dh = hi - q_
        mask = valid & (dl > 1e-6) & (dh > 1e-6)
        g = np.zeros(model_.nv)
        g[mask] = (dh[mask] - dl[mask]) / (dl[mask] * dh[mask])
        return g

    def _clamp_config(model_, q_):
        lo = np.array([float(x) if np.isfinite(x) else 0.0 for x in model_.lowerPositionLimit])
        hi = np.array([float(x) if np.isfinite(x) else 0.0 for x in model_.upperPositionLimit])
        qc = q_.copy()
        valid = np.isfinite(qc) & (lo <= hi)
        qc[valid] = np.clip(qc[valid], lo[valid], hi[valid])
        return qc

    for i, p in enumerate(poses):
        R = (
            pin.utils.rotate("x", p.roll)
            @ pin.utils.rotate("y", p.pitch)
            @ pin.utils.rotate("z", p.yaw)
        )
        T_target = pin.SE3(R, np.array([p.x, p.y, p.z], dtype=float))
        converged = False
        for _ in range(params.max_iter):
            pin.computeJointJacobians(model, data, qv)
            pin.updateFramePlacements(model, data)
            oMf = data.oMf[end_frame_id]
            err = pin.log6(oMf.inverse() * T_target).vector
            if float(np.linalg.norm(err)) < params.tolerance:
                converged = True
                break

            J = pin.getFrameJacobian(model, data, end_frame_id, pin.ReferenceFrame.LOCAL)
            err_norm = float(np.linalg.norm(err))
            lam = params.damping * max(1.0, err_norm * 10.0)
            JJT = J @ J.T
            JJT[np.diag_indices_from(JJT)] += lam
            dq = params.step_size * J.T @ np.linalg.solve(JJT, err)

            if params.null_gain > 0.0:
                g = _joint_limit_grad(model, qv)
                dq += params.null_gain * (g - J.T @ np.linalg.solve(JJT, J @ g))

            qv = _clamp_config(model, pin.integrate(model, qv, dq))

        out.append(JointTrajectoryPoint(time=i * dt, q=[float(v) for v in qv], ik_success=converged))
    return out


def compute_geodesic_stats(reference: list[Pose6D], actual: list[Pose6D], success_flags: list[bool] | None = None) -> TrajectoryStats:
    return compute_trajectory_stats(reference, actual, success_flags=success_flags)
