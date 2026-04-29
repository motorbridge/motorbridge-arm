"""Unified joint-space trajectory planner combining Cartesian geodesic
sampling with CLIK tracking.
/ 统一关节空间轨迹规划器，结合笛卡尔测地线采样与 CLIK 跟踪。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..motion.stats import TrajectoryStats, compute_trajectory_stats
from ..types import Pose6D
from .clik_tracker import IKParams, JointTrajectoryPoint, track_trajectory
from .sampler import TrajPlanParams, plan_cartesian_geodesic_trajectory


@dataclass(slots=True)
class TrajStats:
    """Summary statistics for a tracked trajectory. / 跟踪轨迹的统计摘要。

    Attributes:
        total_points: Total number of trajectory waypoints.
            / 轨迹航点总数。
        success_count: Number of waypoints where IK converged.
            / IK 收敛的航点数。
        success_rate: Fraction of successful IK solves.
            / IK 成功率。
        max_ik_error: Maximum end-effector error over all waypoints.
            / 所有航点中最大末端误差。
        avg_ik_error: Average end-effector error over all waypoints.
            / 所有航点中平均末端误差。
    """

    total_points: int = 0
    success_count: int = 0
    success_rate: float = 0.0
    max_ik_error: float = 0.0
    avg_ik_error: float = 0.0


def plan_joint_space_trajectory(
    model,
    end_frame_id: int,
    q_start: list[float],
    q_end: list[float],
    duration: float,
    kin,
    params: TrajPlanParams | None = None,
    ik_params: IKParams | None = None,
    null_gain: float = 0.1,
    start_pose: Pose6D | None = None,
    end_pose: Pose6D | None = None,
) -> list[JointTrajectoryPoint]:
    """Plan a joint-space trajectory by combining Cartesian geodesic
    sampling with CLIK tracking. / 结合笛卡尔测地线采样与 CLIK 跟踪规划关节空间轨迹。

    The pipeline is:

    1. Determine the start and end Cartesian poses (using forward
       kinematics from ``q_start`` / ``q_end`` unless overridden).
    2. Sample a Cartesian trajectory via SE(3) geodesic interpolation.
    3. Track the Cartesian trajectory in joint space using CLIK.

    Args:
        model: Pinocchio model (or ``None`` for fallback path).
            / Pinocchio 模型（回退路径下可为 ``None``）。
        end_frame_id: Frame ID of the end-effector.
            / 末端执行器的帧 ID。
        q_start: Start joint configuration.
            / 起始关节配置。
        q_end: End joint configuration.
            / 终止关节配置。
        duration: Trajectory duration in seconds.  Must be > 0.
            / 轨迹持续时间（秒）。必须大于 0。
        kin: Kinematics solver instance.
            / 运动学求解器实例。
        params: Trajectory sampling parameters (time step, motion profile).
            Uses defaults when ``None``.
            / 轨迹采样参数（时间步长、运动曲线）。为 ``None`` 时使用默认值。
        ik_params: IK solver parameters.  Uses defaults when ``None``.
            / IK 求解器参数。为 ``None`` 时使用默认值。
        null_gain: Null-space joint-limit avoidance gain.
            / 零空间关节限位规避增益。
        start_pose: Optional override for the start Cartesian pose.
            If ``None``, computed via ``kin.forward(q_start)``.
            / 可选的起始笛卡尔位姿覆盖。为 ``None`` 时通过 ``kin.forward(q_start)`` 计算。
        end_pose: Optional override for the end Cartesian pose.
            If ``None``, computed via ``kin.forward(q_end)``.
            / 可选的终止笛卡尔位姿覆盖。为 ``None`` 时通过 ``kin.forward(q_end)`` 计算。

    Returns:
        List of ``JointTrajectoryPoint`` objects representing the
        tracked joint-space trajectory.
        / 表示跟踪关节空间轨迹的 ``JointTrajectoryPoint`` 对象列表。

    Raises:
        ValueError: If ``duration`` is not positive.
            / 若 ``duration`` 非正则抛出。
    """
    if duration <= 0.0:
        raise ValueError("duration must be > 0")
    if params is None:
        params = TrajPlanParams()
    if ik_params is None:
        ik_params = IKParams()

    T_start = start_pose if start_pose is not None else kin.forward(q_start)
    T_end = end_pose if end_pose is not None else kin.forward(q_end)
    cart_result = plan_cartesian_geodesic_trajectory(T_start, T_end, duration, params)
    return track_trajectory(
        model=model,
        end_frame_id=end_frame_id,
        traj=cart_result.trajectory,
        q_init=q_start,
        kin=kin,
        ik_params=ik_params,
        null_gain=null_gain,
    )


def compute_traj_stats(reference: list[Pose6D], actual: list[Pose6D], success_flags: list[bool] | None = None) -> TrajectoryStats:
    """Compute tracking error statistics between reference and actual poses.
    / 计算参考位姿与实际位姿之间的跟踪误差统计。

    Delegates to the motion-layer ``compute_trajectory_stats`` function
    and wraps the result in a ``TrajStats`` dataclass.

    Args:
        reference: List of reference (desired) Cartesian poses.
            / 参考（期望）笛卡尔位姿列表。
        actual: List of actually-achieved Cartesian poses.
            / 实际达到的笛卡尔位姿列表。
        success_flags: Optional per-point success flags indicating
            whether IK converged.  If ``None``, all points are
            considered successful.
            / 可选的逐点成功标志，指示 IK 是否收敛。为 ``None`` 时所有点视为成功。

    Returns:
        ``TrajectoryStats`` object with error metrics.
        / 包含误差指标的 ``TrajectoryStats`` 对象。
    """
    return compute_trajectory_stats(reference, actual, success_flags=success_flags)
