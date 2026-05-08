"""Lock-free LIFO shared-memory ring buffer for cross-process data sharing.

用于跨进程数据共享的无锁 LIFO（后进先出）共享内存环形缓冲区。

This module provides :class:`SharedRingBuffer`, a ring buffer backed by
``multiprocessing.shared_memory`` that stores per-joint observations
(positions, velocities, torques) and supports LIFO (last-in-first-out)
read semantics so consumers always retrieve the **most recent** data.

本模块提供 :class:`SharedRingBuffer`，它使用 ``multiprocessing.shared_memory``
支持的环形缓冲区，存储每关节观测数据（位置、速度、力矩），并支持 LIFO（后进先出）
读取语义，使消费者始终检索**最新**数据。

Inspired by arx5-sdk's SharedMemoryRingBuffer, but re-implemented using
Python's ``multiprocessing.shared_memory`` module for portability.

受 arx5-sdk 的 SharedMemoryRingBuffer 启发，但使用 Python 的
``multiprocessing.shared_memory`` 模块重新实现以保证可移植性。

Data layout (binary, little-endian)::

    Header (8 bytes):
        tick        8B  uint64   monotonic write counter

    Slots (repeated *buffer_size* times):
        slot_tick   8B  uint64   tick value when this slot was written
        positions   num_joints * 8B  float64  joint positions (rad)
        velocities  num_joints * 8B  float64  joint velocities (rad/s)
        torques     num_joints * 8B  float64  joint torques (Nm)

    Slot size = 8 + num_joints * 3 * 8 = 8 + num_joints * 24 bytes
    Total size = 8 + buffer_size * slot_size
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / 常量
# ---------------------------------------------------------------------------

_TICK_FMT = "<Q"  # uint64
_TICK_SIZE = struct.calcsize(_TICK_FMT)  # 8

# Per-slot layout: tick(8) + positions(n*8) + velocities(n*8) + torques(n*8)
# 每槽布局: tick(8) + 位置(n*8) + 速度(n*8) + 力矩(n*8)
_SLOT_HEADER_FMT = "<Q"  # slot_tick uint64
_SLOT_HEADER_SIZE = struct.calcsize(_SLOT_HEADER_FMT)  # 8

# Max retries for consistent reads / 一致性读取的最大重试次数
_MAX_RETRIES = 10_000
_RETRY_SLEEP = 1e-6  # 1 us yield

# Try importing shared_memory; if unavailable the class becomes a graceful no-op.
# 尝试导入 shared_memory；如果不可用，类将退化为空操作。
try:
    from multiprocessing.shared_memory import SharedMemory
except ImportError:  # pragma: no cover
    SharedMemory = None  # type: ignore[assignment,misc]


def _slot_size(num_joints: int) -> int:
    """Compute the byte size of one ring-buffer slot.

    计算一个环形缓冲区槽的字节大小。
    """
    return _SLOT_HEADER_SIZE + num_joints * 3 * 8


def _total_size(num_joints: int, buffer_size: int) -> int:
    """Compute the total shared-memory segment size.

    计算共享内存段的总大小。
    """
    return _TICK_SIZE + buffer_size * _slot_size(num_joints)


class SharedRingBuffer:
    """Lock-free LIFO shared memory ring buffer for cross-process data.

    用于跨进程数据的无锁 LIFO 共享内存环形缓冲区。

    Each slot stores one observation (positions, velocities, torques).
    Writers advance a monotonic tick counter and write to the next slot
    in a circular fashion.  Readers always read the most recent slot
    (LIFO semantics), or the last *k* slots.

    每个槽存储一次观测（位置、速度、力矩）。写入者递增单调 tick 计数器
    并以循环方式写入下一个槽。读取者始终读取最近的槽（LIFO 语义），
    或最近 *k* 个槽。

    A writer process creates an instance with ``create=True`` (the default)
    which allocates the shared-memory block.  Any number of reader processes
    can attach by constructing with ``create=False`` and the same *name*.

    写进程以 ``create=True``（默认值）创建实例，分配共享内存块。
    任意数量的读进程可以用 ``create=False`` 和相同的 *name* 来附加。

    Thread safety is provided via ``threading.Lock`` to prevent concurrent
    writers from corrupting the tick counter.  Cross-process safety relies
    on the assumption of a single writer process.

    线程安全通过 ``threading.Lock`` 提供，防止并发写入者破坏 tick 计数器。
    跨进程安全依赖于单写入进程的假设。

    If ``multiprocessing.shared_memory`` is not available (e.g. Python < 3.8),
    the instance degrades gracefully: all operations become no-ops and
    :meth:`get` returns an empty dict.

    如果 ``multiprocessing.shared_memory`` 不可用（例如 Python < 3.8），
    实例会优雅降级：所有操作变为空操作，:meth:`get` 返回空字典。

    Args:
        name: Shared-memory segment name.  Must be unique system-wide.
              共享内存段名称，必须在整个系统中唯一。
        num_joints: Number of joints per observation.
                    每次观测的关节数量。
        buffer_size: Number of ring-buffer slots (default 100).
                     环形缓冲区槽数（默认 100）。
        create: ``True`` (default) to *create* a new segment; ``False`` to
                attach to an existing one.
                ``True``（默认）创建新段；``False`` 附加到已有段。
    """

    def __init__(
        self,
        name: str,
        num_joints: int,
        buffer_size: int = 100,
        create: bool = True,
    ) -> None:
        self._name = name
        self._num_joints = num_joints
        self._buffer_size = buffer_size
        self._slot_sz = _slot_size(num_joints)
        self._total_sz = _total_size(num_joints, buffer_size)
        self._shm: Any | None = None
        self._active = False
        self._created = create
        self._lock = threading.Lock()

        if SharedMemory is None:
            logger.warning(
                "multiprocessing.shared_memory is unavailable; "
                "SharedRingBuffer('%s') will operate as a no-op. "
                "multiprocessing.shared_memory 不可用，SharedRingBuffer 将作为空操作运行。",
                name,
            )
            return

        try:
            if create:
                self._shm = SharedMemory(
                    name=name, create=True, size=self._total_sz
                )
                self._init_buffer()
            else:
                self._shm = SharedMemory(name=name, create=False)
            self._active = True
        except FileExistsError:
            logger.debug(
                "Shared memory '%s' already exists, attaching. "
                "共享内存 '%s' 已存在，正在附加。",
                name,
            )
            self._shm = SharedMemory(name=name, create=False)
            self._active = True
        except Exception as exc:
            logger.warning(
                "Failed to initialise shared memory '%s': %s. "
                "Operating as no-op. "
                "初始化共享内存 '%s' 失败: %s，将作为空操作运行。",
                name,
                exc,
                name,
                exc,
            )

    # ------------------------------------------------------------------
    # Public API / 公共接口
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """Return ``True`` if the shared-memory segment is usable.

        当共享内存段可用时返回 ``True``。
        """
        return self._active

    @property
    def name(self) -> str:
        """Shared-memory segment name. / 共享内存段名称。"""
        return self._name

    @property
    def num_joints(self) -> int:
        """Number of joints per observation. / 每次观测的关节数量。"""
        return self._num_joints

    @property
    def buffer_size(self) -> int:
        """Number of ring-buffer slots. / 环形缓冲区槽数。"""
        return self._buffer_size

    @property
    def tick(self) -> int:
        """Current monotonic write tick counter.

        当前的单调递增写入 tick 计数器。
        """
        if not self._active:
            return 0
        buf = self._shm.buf
        return struct.unpack_from(_TICK_FMT, buf, 0)[0]

    def put(
        self,
        positions: list[float] | tuple[float, ...],
        velocities: list[float] | tuple[float, ...],
        torques: list[float] | tuple[float, ...],
    ) -> None:
        """Write a new observation to the next slot, advancing the tick.

        将新观测写入下一个槽，递增 tick。

        Args:
            positions: Joint positions in radians. / 关节位置（弧度）。
            velocities: Joint velocities in rad/s. / 关节速度（弧度/秒）。
            torques: Joint torques in Nm. / 关节力矩（牛顿·米）。
        """
        if not self._active:
            return

        n = self._num_joints
        if len(positions) < n or len(velocities) < n or len(torques) < n:
            logger.warning(
                "put(): insufficient data length (expected %d joints). Skipping. "
                "put(): 数据长度不足（期望 %d 个关节），跳过。",
                n,
                n,
            )
            return

        buf = self._shm.buf

        with self._lock:
            # Read and increment tick / 读取并递增 tick
            current_tick = struct.unpack_from(_TICK_FMT, buf, 0)[0]
            new_tick = current_tick + 1

            # Compute slot index / 计算槽索引
            slot_idx = new_tick % self._buffer_size
            slot_offset = _TICK_SIZE + slot_idx * self._slot_sz

            # Write slot data / 写入槽数据
            # First the slot tick / 先写槽 tick
            struct.pack_into(_SLOT_HEADER_FMT, buf, slot_offset, new_tick)

            # Then joint data: positions, velocities, torques
            # 然后写入关节数据：位置、速度、力矩
            data_offset = slot_offset + _SLOT_HEADER_SIZE
            for i in range(n):
                struct.pack_into(
                    "<d", buf, data_offset + i * 8, float(positions[i])
                )
            data_offset += n * 8
            for i in range(n):
                struct.pack_into(
                    "<d", buf, data_offset + i * 8, float(velocities[i])
                )
            data_offset += n * 8
            for i in range(n):
                struct.pack_into(
                    "<d", buf, data_offset + i * 8, float(torques[i])
                )

            # Publish tick last (store-release ordering via Python GIL)
            # 最后发布 tick（通过 Python GIL 实现存储释放排序）
            struct.pack_into(_TICK_FMT, buf, 0, new_tick)

    def get(self) -> dict[str, Any]:
        """Read the most recent observation (LIFO semantics).

        读取最近的观测（LIFO 语义）。

        Returns:
            A dict with keys ``"tick"``, ``"positions"``, ``"velocities"``,
            ``"torques"``.  If the shared memory is not active or no data has
            been written, returns an empty dict.

            包含 ``"tick"``、``"positions"``、``"velocities"``、
            ``"torques"`` 键的字典。如果共享内存未激活或没有写入数据，
            返回空字典。
        """
        if not self._active:
            return {}

        buf = self._shm.buf

        for _ in range(_MAX_RETRIES):
            # Read tick (load-acquire) / 读取 tick（加载获取）
            tick_val = struct.unpack_from(_TICK_FMT, buf, 0)[0]

            if tick_val == 0:
                return {}

            # Compute slot index / 计算槽索引
            slot_idx = tick_val % self._buffer_size
            slot_offset = _TICK_SIZE + slot_idx * self._slot_sz

            # Read slot tick / 读取槽 tick
            slot_tick = struct.unpack_from(_SLOT_HEADER_FMT, buf, slot_offset)[0]

            # Verify tick consistency / 验证 tick 一致性
            if slot_tick != tick_val:
                continue

            # Read joint data / 读取关节数据
            positions: list[float] = []
            velocities: list[float] = []
            torques: list[float] = []

            data_offset = slot_offset + _SLOT_HEADER_SIZE
            for i in range(self._num_joints):
                val = struct.unpack_from("<d", buf, data_offset + i * 8)[0]
                positions.append(val)
            data_offset += self._num_joints * 8
            for i in range(self._num_joints):
                val = struct.unpack_from("<d", buf, data_offset + i * 8)[0]
                velocities.append(val)
            data_offset += self._num_joints * 8
            for i in range(self._num_joints):
                val = struct.unpack_from("<d", buf, data_offset + i * 8)[0]
                torques.append(val)

            # Double-check tick hasn't changed / 双重检查 tick 未变化
            tick_after = struct.unpack_from(_TICK_FMT, buf, 0)[0]
            if tick_after == tick_val:
                return {
                    "tick": tick_val,
                    "positions": positions,
                    "velocities": velocities,
                    "torques": torques,
                }

        # Fallback: return potentially torn data / 降级：返回可能撕裂的数据
        logger.warning(
            "get(): failed to get consistent snapshot after %d retries. "
            "get(): 在 %d 次重试后仍无法获得一致快照。",
            _MAX_RETRIES,
            _MAX_RETRIES,
        )
        return {
            "tick": tick_val,
            "positions": positions,
            "velocities": velocities,
            "torques": torques,
        }

    def get_last_k(self, k: int) -> list[dict[str, Any]]:
        """Read the last *k* observations, newest first.

        读取最近 *k* 次观测，最新的在前。

        Args:
            k: Number of observations to retrieve. / 要检索的观测数量。

        Returns:
            List of observation dicts (same format as :meth:`get`), newest
            first.  May contain fewer than *k* entries if the buffer has not
            been filled enough.

            观测字典列表（格式与 :meth:`get` 相同），最新在前。
            如果缓冲区未填满，可能包含少于 *k* 条记录。
        """
        if not self._active:
            return []

        buf = self._shm.buf
        current_tick = struct.unpack_from(_TICK_FMT, buf, 0)[0]

        if current_tick == 0:
            return []

        # Limit k to available data / 限制 k 为可用数据量
        available = min(k, current_tick, self._buffer_size)
        results: list[dict[str, Any]] = []

        for offset in range(available):
            target_tick = current_tick - offset
            if target_tick <= 0:
                break

            slot_idx = target_tick % self._buffer_size
            slot_offset = _TICK_SIZE + slot_idx * self._slot_sz

            # Verify slot tick / 验证槽 tick
            slot_tick = struct.unpack_from(_SLOT_HEADER_FMT, buf, slot_offset)[0]
            if slot_tick != target_tick:
                # Slot has been overwritten, stop / 槽已被覆盖，停止
                break

            positions: list[float] = []
            velocities: list[float] = []
            torques: list[float] = []

            data_off = slot_offset + _SLOT_HEADER_SIZE
            for i in range(self._num_joints):
                val = struct.unpack_from("<d", buf, data_off + i * 8)[0]
                positions.append(val)
            data_off += self._num_joints * 8
            for i in range(self._num_joints):
                val = struct.unpack_from("<d", buf, data_off + i * 8)[0]
                velocities.append(val)
            data_off += self._num_joints * 8
            for i in range(self._num_joints):
                val = struct.unpack_from("<d", buf, data_off + i * 8)[0]
                torques.append(val)

            # Re-verify tick after reading to detect torn writes.
            slot_tick_after = struct.unpack_from(_SLOT_HEADER_FMT, buf, slot_offset)[0]
            if slot_tick_after != target_tick:
                break

            results.append(
                {
                    "tick": target_tick,
                    "positions": positions,
                    "velocities": velocities,
                    "torques": torques,
                }
            )

        return results

    def clear(self) -> None:
        """Reset the tick counter and zero out all slots.

        重置 tick 计数器并将所有槽清零。
        """
        if not self._active:
            return
        buf = self._shm.buf
        with self._lock:
            buf[:] = b"\x00" * self._total_sz

    def close(self) -> None:
        """Close the shared-memory segment and release resources.

        关闭共享内存段并释放资源。

        When the instance was created with ``create=True`` the underlying
        segment is also *unlinked* so the OS reclaims the memory.  Attach-only
        instances merely close their file descriptor.

        当实例以 ``create=True`` 创建时，底层段也会被 *unlink*，操作系统将
        回收内存。仅附加的实例仅关闭其文件描述符。
        """
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception as exc:
                logger.debug(
                    "Error closing shared memory '%s': %s "
                    "关闭共享内存 '%s' 时出错: %s",
                    self._name,
                    exc,
                    self._name,
                    exc,
                )
            if self._created:
                try:
                    self._shm.unlink()
                except Exception as exc:
                    logger.debug(
                        "Error unlinking shared memory '%s': %s "
                        "取消链接共享内存 '%s' 时出错: %s",
                        self._name,
                        exc,
                        self._name,
                        exc,
                    )
            self._shm = None
            self._active = False

    def unlink(self) -> None:
        """Unlink (destroy) the shared-memory segment from the OS.

        从操作系统中取消链接（销毁）共享内存段。

        Should only be called by the process that originally created the
        segment.  Safe to call even if the segment was already unlinked.

        只应由最初创建段的进程调用。即使段已被取消链接也可以安全调用。
        """
        if self._shm is not None:
            try:
                self._shm.unlink()
            except Exception as exc:
                logger.debug(
                    "Error unlinking shared memory '%s': %s "
                    "取消链接共享内存 '%s' 时出错: %s",
                    self._name,
                    exc,
                    self._name,
                    exc,
                )

    # ------------------------------------------------------------------
    # Internal helpers / 内部辅助方法
    # ------------------------------------------------------------------

    def _init_buffer(self) -> None:
        """Write zeros into a freshly-created segment.

        向新创建的段写入零值。
        """
        buf = self._shm.buf
        buf[:] = b"\x00" * self._total_sz

    def __repr__(self) -> str:
        status = "active" if self._active else "inactive"
        return (
            f"SharedRingBuffer(name={self._name!r}, "
            f"num_joints={self._num_joints}, "
            f"buffer_size={self._buffer_size}, "
            f"status={status})"
        )
