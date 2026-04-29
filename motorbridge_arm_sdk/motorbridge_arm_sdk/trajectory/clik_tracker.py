"""CLIK (Closed-Loop Inverse Kinematics) trajectory tracker with null-space
joint limit avoidance.
/ 带零空间关节限位规避的 CLIK 轨迹跟踪器。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model.inverse_kinematics import IKParams
from ..model.qp_solver import QPSolver
from ..types import Pose6D


@dataclass(slots=True)
class JointTrajectoryPoint:
    """A single point on a joint-space trajectory. / 关节空间轨迹上的一个点。

    Attributes:
        time: Time stamp in seconds. / 时间戳（秒）。
        q: Joint configuration at this time step. / 该时间步的关节配置。
        ik_success: Whether IK converged within tolerance at this point.
            / 该点 IK 是否在容差内收敛。
    """

    time: float
    q: list[float]
    ik_success: bool


def _joint_limit_grad(model, qv, np):
    """Compute a null-space gradient that pushes joints away from limits.
    / 计算将关节推离限位的零空间梯度。

    The gradient increases as a joint approaches its limit, providing
    a repulsive potential that is projected into the null-space of the
    task Jacobian during CLIK iteration.

    Args:
        model: Pinocchio model with ``lowerPositionLimit`` and
            ``upperPositionLimit`` attributes.
            / 含有 ``lowerPositionLimit`` 和 ``upperPositionLimit`` 属性的 Pinocchio 模型。
        qv: Current joint configuration vector.
            / 当前关节配置向量。
        np: NumPy module reference.
            / NumPy 模块引用。

    Returns:
        Gradient vector of shape ``(nv,)``.
        / 梯度向量，形状 ``(nv,)``。
    """
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
    """Clamp joint values to their position limits. / 将关节值夹紧到位置限位内。

    Args:
        model: Pinocchio model with ``lowerPositionLimit`` and
            ``upperPositionLimit`` attributes.
            / 含有 ``lowerPositionLimit`` 和 ``upperPositionLimit`` 属性的 Pinocchio 模型。
        qv: Joint configuration vector to clamp.
            / 待夹紧的关节配置向量。
        np: NumPy module reference.
            / NumPy 模块引用。

    Returns:
        Clamped joint configuration vector.
        / 夹紧后的关节配置向量。
    """
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
    """Track a Cartesian trajectory using CLIK inverse kinematics. / 使用 CLIK 逆运动学跟踪笛卡尔轨迹。

    For each Cartesian waypoint in ``traj``, solves an iterative IK
    problem using a damped-least-squares Jacobian pseudo-inverse with
    optional null-space joint-limit avoidance.  The solution from the
    previous waypoint is used as the initial guess for the next,
    ensuring temporal continuity.

    When Pinocchio is unavailable, falls back to the simpler
    ``kin.inverse`` solver without null-space optimization.

    Args:
        model: Pinocchio model (or ``None`` for fallback path).
            / Pinocchio 模型（回退路径下可为 ``None``）。
        end_frame_id: Frame ID of the end-effector in the Pinocchio model.
            / Pinocchio 模型中末端执行器的帧 ID。
        traj: Cartesian trajectory object with a ``points()`` method
            returning an iterable of ``CartesianPoint`` (each having
            ``time`` and ``pose`` attributes).
            / 具有 ``points()`` 方法的笛卡尔轨迹对象，返回 ``CartesianPoint`` 可迭代对象。
        q_init: Initial joint configuration used as seed for the
            first IK solve.
            / 第一次 IK 求解的初始关节配置种子。
        kin: Kinematics solver instance (must provide
            ``inverse(pose, q_seed)`` when Pinocchio is unavailable).
            / 运动学求解器实例（Pinocchio 不可用时须提供 ``inverse(pose, q_seed)``）。
        ik_params: IK solver parameters (max iterations, tolerance,
            damping, step size).  Uses defaults when ``None``.
            / IK 求解器参数（最大迭代次数、容差、阻尼、步长）。为 ``None`` 时使用默认值。
        null_gain: Gain for the null-space joint-limit avoidance
            gradient.  Set to ``0.0`` to disable.
            / 零空间关节限位规避梯度的增益。设为 ``0.0`` 可禁用。

    Returns:
        List of ``JointTrajectoryPoint`` objects, one per Cartesian
        waypoint, each containing ``time``, ``q``, and ``ik_success``.
        / ``JointTrajectoryPoint`` 对象列表，每个笛卡尔航点一个，包含 ``time``、``q`` 和 ``ik_success``。
    """
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

    # Instantiate the QP solver for singularity-robust IK.
    # 创建 QP 求解器实例以实现奇异性鲁棒逆运动学。
    _qp = QPSolver(
        damping_base=ik_params.damping,
        manipulability_threshold=0.01,
    )
    _dq_prev: np.ndarray | None = None

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

            # Use QPSolver for singularity-robust damped least-squares
            # with null-space joint-limit avoidance.
            # 使用 QPSolver 进行奇异性鲁棒阻尼最小二乘求解，
            # 并通过零空间进行关节限位规避。
            if null_gain > 0.0:
                g = _joint_limit_grad(model, qv, np)
                dq = _qp.solve_with_nullspace_gradient(
                    jacobian=J,
                    error=err,
                    dq_prev=_dq_prev,
                    null_gradient=g,
                    null_gain=null_gain,
                    damping=ik_params.damping * max(1.0, err_norm * 10.0),
                )
            else:
                dq = _qp.solve(
                    jacobian=J,
                    error=err,
                    dq_prev=_dq_prev,
                    damping=ik_params.damping * max(1.0, err_norm * 10.0),
                )

            dq = ik_params.step_size * dq
            _dq_prev = dq.copy()

            qv = _clamp_config(model, pin.integrate(model, qv, dq), np)

        out.append(JointTrajectoryPoint(time=pt.time, q=[float(v) for v in qv], ik_success=converged))

    return out
