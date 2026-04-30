from __future__ import annotations

"""Thread-safe FPS monitor for named channels.

线程安全的帧率（FPS）监测器，支持按名称追踪多个通道的帧率。

Each channel maintains a sliding window of the most recent *window_size*
monotonic timestamps.  The FPS for a channel is computed as the number of
recorded timestamps divided by the time span covered by that window.

每个通道维护一个滑动窗口，保存最近 *window_size* 个单调时间戳。
通道的 FPS 计算方式为：时间戳数量除以窗口所覆盖的时间跨度。

Typical usage / 典型用法::

    monitor = FPSMonitor(window_size=100)
    monitor.tick("feedback")   # call every frame / 每帧调用
    print(monitor.fps("feedback"))
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ChannelWindow:
    """Internal sliding-window storage for a single channel.

    单个通道的内部滑动窗口存储。

    Attributes:
        timestamps: Monotonic timestamps of the last N frames.
                    最近 N 帧的单调时间戳。
    """

    timestamps: deque[float]


class FPSMonitor:
    """Thread-safe frames-per-second tracker for named channels.

    线程安全的帧率追踪器，支持按名称管理多个通道。

    FPS is computed from a sliding window of the last *window_size* timestamps
    per channel: ``fps = count / (t_latest - t_oldest)``.  If a channel has
    fewer than two timestamps the reported FPS is ``0.0``.

    FPS 通过每个通道最近 *window_size* 个时间戳的滑动窗口计算：
    ``fps = 数量 / (最新时间戳 - 最早时间戳)``。
    若某通道的时间戳不足两个，则返回 ``0.0``。

    All public methods are safe to call from any thread.

    所有公开方法均可从任意线程安全调用。

    Args:
        window_size: Maximum number of timestamps retained per channel.
                     Must be >= 2.  Defaults to 100.
                     每个通道保留的最大时间戳数量。必须 >= 2。默认 100。

    Raises:
        ValueError: If *window_size* is less than 2.

    Example / 示例::

        mon = FPSMonitor(window_size=60)
        mon.tick("render")
        time.sleep(1.0)
        mon.tick("render")
        print(mon.fps("render"))  # ≈ 2.0
    """

    def __init__(self, window_size: int = 100) -> None:
        if window_size < 2:
            raise ValueError(
                f"window_size must be >= 2, got {window_size}"
            )
        self._window_size = window_size
        self._channels: dict[str, _ChannelWindow] = {}
        self._lock = threading.Lock()

    # -- public API / 公开接口 ------------------------------------------------

    def tick(self, channel: str) -> None:
        """Record a frame event on *channel*.

        在指定通道上记录一帧事件。

        Appends the current monotonic time to the channel's sliding window.
        If the channel does not yet exist it is created automatically.

        将当前单调时间追加到该通道的滑动窗口中。
        若通道尚不存在，则会自动创建。

        Args:
            channel: Name of the channel to record a frame on.
                     要记录帧的通道名称。
        """
        now = time.monotonic()
        with self._lock:
            win = self._channels.get(channel)
            if win is None:
                logger.warning(
                    "Auto-creating unknown FPS channel %r; "
                    "consider pre-registering it.",
                    channel,
                )
                win = _ChannelWindow(timestamps=deque(maxlen=self._window_size))
                self._channels[channel] = win
            win.timestamps.append(now)

    def fps(self, channel: str) -> float:
        """Return the current FPS for *channel*.

        返回指定通道的当前 FPS。

        Computed as ``count / (t_latest - t_oldest)`` over the sliding
        window.  Returns ``0.0`` when fewer than two timestamps are available
        for the channel, or when the channel is unknown.

        通过滑动窗口按 ``数量 / (最新时间 - 最早时间)`` 计算。
        当通道的时间戳不足两个或通道不存在时返回 ``0.0``。

        Args:
            channel: Name of the channel to query.
                     要查询的通道名称。

        Returns:
            Frames-per-second value for the channel.
            该通道的每秒帧数。
        """
        with self._lock:
            win = self._channels.get(channel)
            if win is None or len(win.timestamps) < 2:
                return 0.0
            span = win.timestamps[-1] - win.timestamps[0]
            if span <= 0.0:
                return 0.0
            return (len(win.timestamps) - 1) / span

    def all_fps(self) -> dict[str, float]:
        """Return a snapshot of FPS values for all channels.

        返回所有通道的 FPS 快照。

        Returns:
            A dict mapping each channel name to its current FPS.
            键为通道名称、值为当前 FPS 的字典。
        """
        with self._lock:
            result: dict[str, float] = {}
            for name, win in self._channels.items():
                if len(win.timestamps) < 2:
                    result[name] = 0.0
                    continue
                span = win.timestamps[-1] - win.timestamps[0]
                if span <= 0.0:
                    result[name] = 0.0
                else:
                    result[name] = (len(win.timestamps) - 1) / span
            return result

    def reset(self, channel: str | None = None) -> None:
        """Clear recorded timestamps for one or all channels.

        清除一个或全部通道的已记录时间戳。

        Args:
            channel: Name of the channel to reset.  If ``None``, all channels
                     are cleared.
                     要重置的通道名称。若为 ``None``，则清除所有通道。
        """
        with self._lock:
            if channel is None:
                self._channels.clear()
            else:
                win = self._channels.get(channel)
                if win is not None:
                    win.timestamps.clear()

    def remove_channel(self, channel: str) -> bool:
        """Remove a channel entirely from the monitor.

        从监测器中完全移除一个通道。

        Args:
            channel: Name of the channel to remove.
                     要移除的通道名称。

        Returns:
            ``True`` if the channel existed and was removed, ``False`` otherwise.
            如果通道存在且被移除返回 ``True``，否则返回 ``False``。
        """
        with self._lock:
            if channel in self._channels:
                del self._channels[channel]
                return True
            return False
