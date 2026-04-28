from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from ..types import Pose6D


def _rot_to_rpy(R) -> tuple[float, float, float]:
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


@dataclass(slots=True)
class _SimpleChain:
    link_lengths: tuple[float, ...] = (0.22, 0.20, 0.10)

    def forward(self, q: list[float]) -> Pose6D:
        q0, q1, q2 = (q + [0.0, 0.0, 0.0])[:3]
        l1, l2, l3 = self.link_lengths
        a1 = q1
        a2 = q1 + q2
        x = l1 * math.cos(a1) + l2 * math.cos(a2) + l3
        z = 0.10 + l1 * math.sin(a1) + l2 * math.sin(a2)
        y = 0.0
        return Pose6D(x=x, y=y, z=z, roll=0.0, pitch=0.0, yaw=q0)


class Kinematics:
    def __init__(self, urdf_path: str | None = None, ee_frame: str = "tool0") -> None:
        self._pin = None
        self._model = None
        self._data = None
        self._frame_id = None
        self._simple = _SimpleChain()
        self._ee_frame = ee_frame
        if urdf_path:
            self._try_load_pinocchio(urdf_path, ee_frame)

    def _try_load_pinocchio(self, urdf_path: str, ee_frame: str) -> None:
        try:
            import pinocchio as pin
        except Exception:
            return
        p = Path(urdf_path)
        if not p.exists():
            return
        model = pin.buildModelFromUrdf(str(p))
        data = model.createData()
        if ee_frame not in [f.name for f in model.frames]:
            ee_frame = model.frames[-1].name
        frame_id = model.getFrameId(ee_frame)
        self._pin = pin
        self._model = model
        self._data = data
        self._frame_id = frame_id
        self._ee_frame = ee_frame

    @property
    def has_pinocchio(self) -> bool:
        return self._pin is not None

    def forward(self, q: list[float]) -> Pose6D:
        if self._pin is None:
            return self._simple.forward(q)
        nq = self._model.nq
        try:
            import numpy as np
        except Exception:
            return self._simple.forward(q)
        qv = np.zeros(nq)
        n = min(nq, len(q))
        qv[:n] = np.array(q[:n], dtype=float)
        self._pin.forwardKinematics(self._model, self._data, qv)
        self._pin.updateFramePlacements(self._model, self._data)
        oMf = self._data.oMf[self._frame_id]
        t = oMf.translation
        R = oMf.rotation
        roll, pitch, yaw = _rot_to_rpy(R)
        return Pose6D(x=float(t[0]), y=float(t[1]), z=float(t[2]), roll=roll, pitch=pitch, yaw=yaw)

    def inverse(self, target: Pose6D, q_seed: list[float]) -> list[float]:
        if not q_seed:
            q_seed = [0.0] * 6
        if self._pin is not None:
            q = self._inverse_pinocchio(target, q_seed)
            if q is not None:
                return q
        return self._inverse_simple(target, q_seed)

    def _inverse_simple(self, target: Pose6D, q_seed: list[float]) -> list[float]:
        out = list(q_seed)
        if len(out) >= 3:
            out[0] = target.yaw
            out[1] = max(-2.6, min(2.6, (target.z - 0.10) * 2.0))
            out[2] = max(-2.6, min(2.6, (target.x - 0.2) * 2.0 - out[1]))
        return out

    def _inverse_pinocchio(self, target: Pose6D, q_seed: list[float]) -> list[float] | None:
        try:
            import numpy as np
        except Exception:
            return None

        pin = self._pin
        model = self._model
        data = self._data
        nq = model.nq
        q = np.zeros(nq)
        n = min(nq, len(q_seed))
        q[:n] = np.array(q_seed[:n], dtype=float)

        R = (
            pin.utils.rotate("x", target.roll)
            @ pin.utils.rotate("y", target.pitch)
            @ pin.utils.rotate("z", target.yaw)
        )
        T_target = pin.SE3(R, np.array([target.x, target.y, target.z], dtype=float))

        for _ in range(120):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            T_cur = data.oMf[self._frame_id]
            err = pin.log6(T_cur.inverse() * T_target).vector
            if float(np.linalg.norm(err)) < 1e-4:
                return [float(v) for v in q]

            J = pin.computeFrameJacobian(model, data, q, self._frame_id, pin.ReferenceFrame.LOCAL)
            lam = 1e-6
            JJT = J @ J.T
            JJT[np.diag_indices_from(JJT)] += lam
            dq = 0.6 * J.T @ np.linalg.solve(JJT, err)
            q = pin.integrate(model, q, dq)
            lo = np.array([float(x) for x in model.lowerPositionLimit])
            hi = np.array([float(x) for x in model.upperPositionLimit])
            mask = np.isfinite(lo) & np.isfinite(hi)
            q[mask] = np.clip(q[mask], lo[mask], hi[mask])
        return None
