from __future__ import annotations

from dataclasses import dataclass

from ..model.inverse_kinematics import IKParams
from ..types import Pose6D


@dataclass(slots=True)
class JointTrajectoryPoint:
    time: float
    q: list[float]
    ik_success: bool


def _joint_limit_grad(model, qv, np):
    lo = np.array([float(x) for x in model.lowerPositionLimit])
    hi = np.array([float(x) for x in model.upperPositionLimit])
    valid = np.isfinite(lo) & np.isfinite(hi)
    dl = qv - lo
    dh = hi - qv
    mask = valid & (dl > 1e-6) & (dh > 1e-6)
    g = np.zeros(model.nv)
    g[mask] = (dh[mask] - dl[mask]) / (dl[mask] * dh[mask])
    return g


def _clamp_config(model, qv, np):
    lo = np.array([float(x) if np.isfinite(x) else 0.0 for x in model.lowerPositionLimit])
    hi = np.array([float(x) if np.isfinite(x) else 0.0 for x in model.upperPositionLimit])
    qc = qv.copy()
    valid = np.isfinite(qc) & (lo <= hi)
    qc[valid] = np.clip(qc[valid], lo[valid], hi[valid])
    return qc


def track_trajectory(
    model,
    end_frame_id: int,
    traj,
    q_init: list[float],
    kin,
    ik_params: IKParams | None = None,
    null_gain: float = 0.1,
) -> list[JointTrajectoryPoint]:
    if ik_params is None:
        ik_params = IKParams()

    out: list[JointTrajectoryPoint] = []
    q = list(q_init)

    pin = getattr(kin, "_pin", None)
    if pin is None or model is None:
        for pt in traj.points():
            qn = kin.inverse(pt.pose, q)
            ok = len(qn) == len(q)
            q = list(qn)
            out.append(JointTrajectoryPoint(time=pt.time, q=list(q), ik_success=ok))
        return out

    import numpy as np

    data = model.createData()
    qv = np.array(q, dtype=float)
    if len(qv) < model.nq:
        qq = np.zeros(model.nq, dtype=float)
        qq[: len(qv)] = qv
        qv = qq

    for pt in traj.points():
        p: Pose6D = pt.pose
        R = pin.utils.rotate("x", p.roll) @ pin.utils.rotate("y", p.pitch) @ pin.utils.rotate("z", p.yaw)
        T_target = pin.SE3(R, np.array([p.x, p.y, p.z], dtype=float))

        converged = False
        for _ in range(ik_params.max_iter):
            pin.computeJointJacobians(model, data, qv)
            pin.updateFramePlacements(model, data)
            oMf = data.oMf[end_frame_id]
            err = pin.log6(oMf.inverse() * T_target).vector
            err_norm = float(np.linalg.norm(err))
            if err_norm < ik_params.tolerance:
                converged = True
                break

            J = pin.getFrameJacobian(model, data, end_frame_id, pin.ReferenceFrame.LOCAL)
            lam = ik_params.damping * max(1.0, err_norm * 10.0)
            JJT = J @ J.T
            JJT[np.diag_indices_from(JJT)] += lam
            dq = ik_params.step_size * J.T @ np.linalg.solve(JJT, err)

            if null_gain > 0.0:
                g = _joint_limit_grad(model, qv, np)
                dq += null_gain * (g - J.T @ np.linalg.solve(JJT, J @ g))

            qv = _clamp_config(model, pin.integrate(model, qv, dq), np)

        out.append(JointTrajectoryPoint(time=pt.time, q=[float(v) for v in qv], ik_success=converged))

    return out
