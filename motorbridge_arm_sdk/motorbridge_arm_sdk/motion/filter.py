"""Moving-average filter and velocity estimation utilities.

Provides an O(1) circular-buffer moving average filter (inspired by
arx5-sdk's ``MovingAverageXd``) and a 4-point central-difference
velocity estimator.
/ O(1) 循环缓冲区滑动平均滤波器（灵感来自 arx5-sdk 的 ``MovingAverageXd``）
和四点中心差分速度估计器。
"""

from __future__ import annotations


class MovingAverage:
    """O(1) circular-buffer moving average filter.

    Maintains a fixed-size sliding window over the most recent *window_size*
    samples.  Each sample is a vector of *dof* scalar values.  Updating the
    average is O(1) because only the difference between the newest and oldest
    sample is applied to the running sum.

    / O(1) 循环缓冲区滑动平均滤波器。
    维护最近 *window_size* 个样本的固定大小滑动窗口。
    每个样本是 *dof* 个标量值的向量。更新平均值为 O(1)，
    因为只需将最新样本与最旧样本的差值应用于运行总和。

    Args:
        dof: Dimensionality of each sample vector.
            / 每个样本向量的维度。
        window_size: Number of samples in the sliding window.  Must be >= 2.
            Defaults to 10.
            / 滑动窗口中的样本数。必须 >= 2。默认为 10。

    Raises:
        ValueError: If *dof* < 1 or *window_size* < 2.
    """

    def __init__(self, dof: int, window_size: int = 10) -> None:
        if dof < 1:
            raise ValueError(f"dof must be >= 1, got {dof}")
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")

        self._dof: int = dof
        self._window_size: int = window_size
        self._buf: list[list[float]] = [[0.0] * dof for _ in range(window_size)]
        self._running_sum: list[float] = [0.0] * dof
        self._count: int = 0
        self._head: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(self, new_data: list[float]) -> list[float]:
        """Push *new_data* into the buffer and return the current average.

        / 将 *new_data* 推入缓冲区并返回当前平均值。

        Args:
            new_data: A list of *dof* floats.

        Returns:
            A list of *dof* floats representing the averaged values over
            the current window.

        Raises:
            ValueError: If *new_data* length does not match *dof*.
        """
        if len(new_data) != self._dof:
            raise ValueError(
                f"new_data length {len(new_data)} != dof {self._dof}"
            )

        oldest = self._buf[self._head]

        # Subtract the oldest sample from the running sum.
        for i in range(self._dof):
            self._running_sum[i] -= oldest[i]

        # Overwrite the oldest slot with the new data.
        for i in range(self._dof):
            self._buf[self._head][i] = new_data[i]
            self._running_sum[i] += new_data[i]

        # Advance the circular index.
        self._head = (self._head + 1) % self._window_size
        if self._count < self._window_size:
            self._count += 1

        # Compute average.
        return [s / self._count for s in self._running_sum]

    def set_window_size(self, size: int) -> None:
        """Reset the filter with a new window size.

        / 使用新的窗口大小重置滤波器。

        Args:
            size: New window size.  Must be >= 2.

        Raises:
            ValueError: If *size* < 2.
        """
        if size < 2:
            raise ValueError(f"window_size must be >= 2, got {size}")
        self._window_size = size
        self._buf = [[0.0] * self._dof for _ in range(size)]
        self._running_sum = [0.0] * self._dof
        self._count = 0
        self._head = 0

    def reset(self) -> None:
        """Clear the buffer and running sums without changing the window size.

        / 清空缓冲区和运行总和，不改变窗口大小。
        """
        self._buf = [[0.0] * self._dof for _ in range(self._window_size)]
        self._running_sum = [0.0] * self._dof
        self._count = 0
        self._head = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def window_size(self) -> int:
        """Current window size. / 当前窗口大小。"""
        return self._window_size

    @property
    def dof(self) -> int:
        """Dimensionality of each sample. / 每个样本的维度。"""
        return self._dof

    @property
    def count(self) -> int:
        """Number of samples currently in the buffer (<= window_size).

        / 缓冲区中当前的样本数（<= window_size）。
        """
        return self._count


def estimate_velocity(
    positions: list[list[float]],
    dt: float,
    window: int = 5,
) -> list[float]:
    """Estimate velocity from position history using 4-point central difference.

    Uses two central-difference estimates (over the full *window* and over half
    the window) and averages them to produce a smoother derivative.

    / 使用四点中心差分从位置历史估计速度。
    使用两个中心差分估计（完整 *window* 和半窗口）并取平均值以产生更平滑的导数。

    Args:
        positions: Ordered list of position vectors, from oldest to newest.
            Must contain at least *window* + 1 entries.
            / 从旧到新的位置向量有序列表。必须至少包含 *window* + 1 个条目。
        dt: Time step between consecutive position samples in seconds.
            / 连续位置样本之间的时间步长（秒）。
        window: Number of samples to span for the full-window central
            difference.  Must be >= 2.  Defaults to 5.
            / 全窗口中心差分所跨越的样本数。必须 >= 2。默认为 5。

    Returns:
        A list of floats (same dimensionality as each position vector)
        representing the estimated velocity at the newest sample.

    Raises:
        ValueError: If there are not enough position samples or *dt* <= 0.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    needed = window + 1
    if len(positions) < needed:
        raise ValueError(
            f"Need at least {needed} position samples (window={window}), "
            f"got {len(positions)}"
        )

    dof = len(positions[0])
    newest = positions[-1]

    # Full-window central difference: (p[-1] - p[-1-window]) / (window * dt)
    oldest_full = positions[-1 - window]
    half = window // 2
    # Half-window central difference: (p[-1] - p[-1-half]) / (half * dt)
    oldest_half = positions[-1 - half]

    velocity: list[float] = []
    for i in range(dof):
        v_full = (newest[i] - oldest_full[i]) / (window * dt)
        v_half = (newest[i] - oldest_half[i]) / (half * dt)
        velocity.append(0.5 * (v_full + v_half))

    return velocity
