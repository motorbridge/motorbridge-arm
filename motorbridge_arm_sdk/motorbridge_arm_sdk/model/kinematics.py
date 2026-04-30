from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

from ..types import Pose6D
from .inverse_kinematics import IKParams, IKResult, solve_ik_advanced

logger = logging.getLogger(__name__)


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
        except ImportError:
            logger.warning("pinocchio not available; kinematics will use simplified chain fallback")
            return
        p = Path(urdf_path)
        if not p.exists():
            logger.warning("URDF file not found: %s; kinematics will use simplified chain fallback", urdf_path)
            return
        model = pin.buildModelFromUrdf(str(p))
        data = model.createData()
        resolved_frame = ee_frame
        if ee_frame not in [f.name for f in model.frames]:
            resolved_frame = model.frames[-1].name
            logger.warning(
                "Requested ee_frame '%s' not found in URDF; falling back to '%s'",
                ee_frame, resolved_frame,
            )
        frame_id = model.getFrameId(resolved_frame)
        self._pin = pin
        self._model = model
        self._data = data
        self._frame_id = frame_id
        self._ee_frame = resolved_frame
        logger.info("Loaded Pinocchio model from %s with ee_frame='%s' (frame_id=%d)", urdf_path, resolved_frame, frame_id)

    @property
    def has_pinocchio(self) -> bool:
        return self._pin is not None

    @property
    def pinocchio_model(self):
        """The loaded Pinocchio model, or ``None`` if Pinocchio is unavailable."""
        return self._model

    @property
    def end_frame_id(self) -> int | None:
        """The resolved end-effector frame ID, or ``None`` if not loaded."""
        return self._frame_id

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
        # Use fresh data per call for thread safety, same pattern as inverse.
        data = self._model.createData()
        self._pin.forwardKinematics(self._model, data, qv)
        self._pin.updateFramePlacements(self._model, data)
        oMf = data.oMf[self._frame_id]
        t = oMf.translation
        R = oMf.rotation
        roll, pitch, yaw = _rot_to_rpy(R)
        return Pose6D(x=float(t[0]), y=float(t[1]), z=float(t[2]), roll=roll, pitch=pitch, yaw=yaw)

    def inverse(self, target: Pose6D, q_seed: list[float]) -> list[float]:
        if not q_seed:
            q_seed = [0.0] * 6
        if self._pin is not None:
            r = self.inverse_result(target, q_seed)
            if r.success:
                return r.q
        return self._inverse_simple(target, q_seed)

    def inverse_result(self, target: Pose6D, q_seed: list[float]) -> IKResult:
        if not q_seed:
            q_seed = [0.0] * 6
        if self._pin is not None:
            q = self._inverse_pinocchio_result(target, q_seed)
            if q is not None:
                return q
        q_fb = self._inverse_simple(target, q_seed)
        return IKResult(q=q_fb, success=False, error=float("inf"), iterations=0)

    def _inverse_simple(self, target: Pose6D, q_seed: list[float]) -> list[float]:
        out = list(q_seed)
        if len(out) >= 3:
            out[0] = target.yaw
            out[1] = max(-2.6, min(2.6, (target.z - 0.10) * 2.0))
            out[2] = max(-2.6, min(2.6, (target.x - 0.2) * 2.0 - out[1]))
        return out

    def _inverse_pinocchio_result(self, target: Pose6D, q_seed: list[float]) -> IKResult | None:
        try:
            import numpy as np
        except Exception:
            return None

        pin = self._pin
        model = self._model
        # NOTE: Create fresh data per call so concurrent inverse() calls do not
        # race on shared mutable pinocchio Data.  The overhead is negligible
        # compared to the IK loop itself.
        data = model.createData()
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

        res = solve_ik_advanced(
            pin=pin,
            model=model,
            data=data,
            frame_id=self._frame_id,
            target_se3=T_target,
            q_seed=[float(v) for v in q],
            params=IKParams(),
        )
        return res
