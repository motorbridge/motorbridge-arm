"""Quadratic Programming based singularity-robust IK solver.
/ 基于二次规划的奇异性鲁棒逆运动学求解器。

This module formulates the joint-velocity IK problem as a damped
least-squares (ridge-regression) optimisation with null-space
projection for secondary objectives such as joint-limit avoidance.
No external QP library is required -- the solution is computed in
closed form using NumPy only.

该模块将关节速度逆运动学问题建模为阻尼最小二乘（岭回归）优化，
并通过零空间投影处理关节限位规避等次级目标。
无需外部 QP 库——仅使用 NumPy 以闭式解计算。

Mathematical formulation / 数学公式化
-------------------------------------
    minimize   ||J * dq - dx||^2  +  lambda * ||dq||^2

where:
    J       task-space Jacobian (6 x nv)
            任务空间雅可比矩阵 (6 x nv)
    dq      joint velocity vector (nv,)
            关节速度向量 (nv,)
    dx      desired Cartesian velocity / error (6,)
            期望笛卡尔速度 / 误差 (6,)
    lambda  damping factor (increases near singularities)
            阻尼因子（在奇异位形附近增大）

Closed-form solution / 闭式解
    dq = J^T (J J^T + lambda I)^{-1} dx
"""

from __future__ import annotations

import math


class QPSolver:
    """Singularity-robust QP-based IK solver using damped least-squares
    with null-space projection.

    使用阻尼最小二乘与零空间投影的奇异性鲁棒 QP 逆运动学求解器。

    The solver monitors the manipulability index of the Jacobian and
    automatically increases damping when the robot approaches a
    singularity, ensuring bounded joint velocities even in
    ill-conditioned configurations.

    求解器监测雅可比矩阵的可操作度指标，并在机器人接近奇异
    位形时自动增大阻尼，确保在病态配置下关节速度仍然有界。

    Attributes:
        damping_base (float): Base damping coefficient.
            / 基础阻尼系数。
        manipulability_threshold (float): Below this value the solver
            begins to increase damping adaptively.
            / 低于该值时求解器开始自适应增大阻尼。
        joint_vel_limit (float | None): Per-joint maximum velocity
            (rad/s).  ``None`` disables velocity clamping.
            / 每关节最大速度 (rad/s)。``None`` 禁用速度夹紧。
    """

    def __init__(
        self,
        damping_base: float = 1e-3,
        manipulability_threshold: float = 0.01,
        joint_vel_limit: float | None = None,
    ) -> None:
        """Initialise the QP solver.

        / 初始化 QP 求解器。

        Args:
            damping_base: Base damping coefficient applied to the
                regularisation term.  Higher values suppress joint
                velocities more aggressively.
                / 施加于正则化项的基础阻尼系数。值越大对关节
                速度的抑制越强。
            manipulability_threshold: Manipulability index threshold
                below which adaptive damping is triggered.
                / 可操作度指标阈值，低于此值触发自适应阻尼。
            joint_vel_limit: Maximum joint velocity in rad/s used for
                box-constraint clamping.  ``None`` disables clamping.
                / 用于箱约束夹紧的最大关节速度 (rad/s)。
                ``None`` 禁用夹紧。
        """
        self.damping_base = damping_base
        self.manipulability_threshold = manipulability_threshold
        self.joint_vel_limit = joint_vel_limit

    # ------------------------------------------------------------------
    # Private helpers / 私有辅助方法
    # ------------------------------------------------------------------

    def _adaptive_damping(self, jacobian, damping: float) -> float:
        """Compute adaptive damping based on manipulability. / 根据可操作度计算自适应阻尼。"""
        import numpy as np

        w = self.manipulability(jacobian)
        if w < self.manipulability_threshold:
            adaptive = self.damping_base * (
                self.manipulability_threshold / max(w, 1e-12)
            )
            return max(damping, adaptive)
        return damping

    def _null_projector(self, J, lam) -> tuple:
        """Compute the damped least-squares pseudo-inverse and null-space projector. / 计算阻尼最小二乘伪逆和零空间投影矩阵。"""
        import numpy as np

        m, n = J.shape
        JJT_reg = J @ J.T + lam * np.eye(m)
        J_pinv = J.T @ np.linalg.solve(JJT_reg, J)  # (n, n)
        N = np.eye(n) - J_pinv
        return JJT_reg, J_pinv, N

    # ------------------------------------------------------------------
    # Public API / 公开接口
    # ------------------------------------------------------------------

    def manipulability(self, jacobian: "np.ndarray") -> float:  # noqa: F821
        """Compute the manipulability index via SVD.

        / 通过 SVD 计算可操作度指标。

        Uses the product of singular values of *J* which is numerically
        more stable than the determinant-based formulation for
        ill-conditioned Jacobians.

        使用 *J* 奇异值的乘积，对于病态雅可比矩阵比基于行列式的
        公式数值更稳定。

        Args:
            jacobian: Task-space Jacobian of shape ``(m, n)`` where
                ``m`` is the task-space dimension (typically 6) and
                ``n`` is the number of joints.
                / 任务空间雅可比矩阵，形状 ``(m, n)``，其中 ``m``
                为任务空间维度（通常为 6），``n`` 为关节数。

        Returns:
            Manipulability index (non-negative scalar).
            / 可操作度指标（非负标量）。
        """
        import numpy as np

        s = np.linalg.svd(jacobian, compute_uv=False)
        return float(np.prod(s))

    def singularity_index(self, jacobian: "np.ndarray") -> float:  # noqa: F821
        """Return a normalised singularity index in [0, 1].

        / 返回 [0, 1] 范围内的归一化奇异性指标。

        * ``0`` means the manipulator is at (or very near) a full
          singularity.
          / ``0`` 表示机械臂处于（或非常接近）完全奇异位形。
        * ``1`` means the manipulator is far from any singularity.

          / ``1`` 表示机械臂远离任何奇异位形。

        The index is computed as ``1 - exp(-w / threshold)`` where
        *w* is the manipulability and *threshold* is
        ``manipulability_threshold``.  This maps [0, inf) smoothly
        onto [0, 1).

        该指标通过 ``1 - exp(-w / threshold)`` 计算，其中 *w*
        为可操作度，*threshold* 为
        ``manipulability_threshold``。这可将 [0, inf) 平滑映射
        至 [0, 1)。

        Args:
            jacobian: Task-space Jacobian ``(m, n)``.
                / 任务空间雅可比矩阵 ``(m, n)``。

        Returns:
            Singularity index in [0, 1].
            / 奇异性指标，范围 [0, 1]。
        """
        w = self.manipulability(jacobian)
        return 1.0 - math.exp(-w / self.manipulability_threshold)

    def solve(
        self,
        jacobian: "np.ndarray",  # noqa: F821
        error: "np.ndarray",  # noqa: F821
        dq_prev: "np.ndarray | None",  # noqa: F821
        damping: float = 1e-3,
        dt: float = 0.002,
    ) -> "np.ndarray":  # noqa: F821
        """Solve the damped least-squares IK step with null-space
        projection.

        / 使用零空间投影求解阻尼最小二乘 IK 步。

        The core optimisation is::

            minimize  ||J dq - dx||^2 + lambda ||dq||^2

        When the manipulability falls below the configured threshold,
        the damping factor is automatically increased to maintain
        numerical stability near singularities.

        核心优化为：

            minimize  ||J dq - dx||^2 + lambda ||dq||^2

        当可操作度低于配置阈值时，阻尼因子会自动增大以在奇异
        位形附近保持数值稳定性。

        Args:
            jacobian: Task-space Jacobian ``(m, n)``.
                / 任务空间雅可比矩阵 ``(m, n)``。
            error: Cartesian error vector ``(m,)`` (serves as *dx*).
                / 笛卡尔误差向量 ``(m,)``（即 *dx*）。
            dq_prev: Previous joint velocity vector ``(n,)`` or
                ``None``.  Used for velocity-continuity regularisation
                when provided.
                / 上一时刻关节速度向量 ``(n,)`` 或 ``None``。
                提供时用于速度连续性正则化。
            damping: Base damping coefficient.  The effective damping
                is ``max(damping, adaptive_damping)`` where the
                adaptive component depends on manipulability.
                / 基础阻尼系数。有效阻尼为
                ``max(damping, adaptive_damping)``，其中自适应分量
                取决于可操作度。
            dt: Timestep in seconds, used to convert joint-velocity
                limits into per-step displacement limits for clamping.
                / 时间步长（秒），用于将关节速度限制转换为每步
                位移限制以进行夹紧。

        Returns:
            Joint velocity vector ``(n,)``.
            / 关节速度向量 ``(n,)``。
        """
        import numpy as np

        J = jacobian
        dx = error
        m, n = J.shape

        # --- Adaptive damping / 自适应阻尼 ---
        lam = self._adaptive_damping(J, damping)

        # --- Damped least-squares solution / 阻尼最小二乘解 ---
        # dq = J^T (J J^T + lambda I)^{-1} dx
        JJT_reg, _, N = self._null_projector(J, lam)
        dq_primary = J.T @ np.linalg.solve(JJT_reg, dx)

        # Secondary task: joint-limit avoidance via null-space gradient.
        # 次级任务：通过零空间梯度进行关节限位规避。
        dq_null = np.zeros(n)

        # Optionally regularise towards previous velocity for smooth
        # trajectories.
        # 可选：通过正则化趋向上一时刻速度以获得平滑轨迹。
        if dq_prev is not None:
            dq_null = dq_prev.copy()

        dq = dq_primary + N @ dq_null

        # --- Joint velocity clamping (box constraints) / 关节速度
        #     夹紧（箱约束）---
        if self.joint_vel_limit is not None:
            max_dq = self.joint_vel_limit * dt
            dq = np.clip(dq, -max_dq, max_dq)

        return dq

    # ------------------------------------------------------------------
    # Convenience helpers / 便捷辅助方法
    # ------------------------------------------------------------------

    def solve_with_nullspace_gradient(
        self,
        jacobian: "np.ndarray",  # noqa: F821
        error: "np.ndarray",  # noqa: F821
        dq_prev: "np.ndarray | None",  # noqa: F821
        null_gradient: "np.ndarray",  # noqa: F821
        null_gain: float = 0.1,
        damping: float = 1e-3,
        dt: float = 0.002,
    ) -> "np.ndarray":  # noqa: F821
        """Solve with an explicit null-space gradient (e.g. joint-limit
        repulsive potential).

        / 使用显式零空间梯度（如关节限位排斥势）求解。

        This is a convenience wrapper around :meth:`solve` that projects
        a user-supplied gradient into the null-space of the Jacobian
        before combining it with the primary task solution.

        这是 :meth:`solve` 的便捷封装，将用户提供的梯度投影到
        雅可比矩阵的零空间后与主任务解组合。

        Args:
            jacobian: Task-space Jacobian ``(m, n)``.
                / 任务空间雅可比矩阵 ``(m, n)``。
            error: Cartesian error ``(m,)``.
                / 笛卡尔误差 ``(m,)``。
            dq_prev: Previous joint velocity ``(n,)`` or ``None``.
                / 上一时刻关节速度 ``(n,)`` 或 ``None``。
            null_gradient: Gradient vector ``(n,)`` to project into the
                null-space (e.g. from a joint-limit potential).
                / 待投影到零空间的梯度向量 ``(n,)``（如来自关节限位势场）。
            null_gain: Scaling factor for the null-space gradient.
                / 零空间梯度的缩放因子。
            damping: Base damping coefficient.
                / 基础阻尼系数。
            dt: Timestep (seconds).
                / 时间步长（秒）。

        Returns:
            Joint velocity ``(n,)`` with null-space contribution.
            / 包含零空间贡献的关节速度 ``(n,)``。
        """
        import numpy as np

        J = jacobian
        dx = error

        # Adaptive damping.
        # 自适应阻尼。
        lam = self._adaptive_damping(J, damping)

        # Primary solution.
        # 主任务解。
        JJT_reg, _, N = self._null_projector(J, lam)
        dq_primary = J.T @ np.linalg.solve(JJT_reg, dx)

        # Null-space term: project gradient, optionally blend with
        # dq_prev for continuity.
        # 零空间项：投影梯度，可选地与 dq_prev 混合以保持连续性。
        dq_null = null_gain * null_gradient
        if dq_prev is not None:
            dq_null = dq_null + 0.5 * dq_prev
        dq = dq_primary + N @ dq_null

        # Box-constraint clamping.
        # 箱约束夹紧。
        if self.joint_vel_limit is not None:
            max_dq = self.joint_vel_limit * dt
            dq = np.clip(dq, -max_dq, max_dq)

        return dq
