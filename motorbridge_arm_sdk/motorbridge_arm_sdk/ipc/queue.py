"""Lock-free FIFO shared-memory queue for cross-process command streaming.

用于跨进程命令流的无锁 FIFO（先进先出）共享内存队列。

This module provides :class:`SharedQueue`, a FIFO queue backed by
``multiprocessing.shared_memory`` that stores joint-position target commands
and supports batch put/get operations for efficient streaming.

本模块提供 :class:`SharedQueue`，它使用 ``multiprocessing.shared_memory``
支持的 FIFO 队列，存储关节位置目标命令，并支持批量 put/get 操作以实现高效流传输。

Data layout (binary, little-endian)::

    Header (16 bytes):
        head        8B  uint64   dequeue cursor (FIFO read position)
        tail        8B  uint64   enqueue cursor (FIFO write position)

    Slots (repeated *buffer_size* times):
        timestamp   8B  float64  command timestamp (seconds)
        positions   num_joints * 8B  float64  target joint positions (rad)

    Slot size = 8 + num_joints * 8 bytes
    Total size = 16 + buffer_size * slot_size
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

_HEAD_FMT = "<Q"  # uint64 head cursor
_TAIL_FMT = "<Q"  # uint64 tail cursor
_CURSOR_SIZE = struct.calcsize(_HEAD_FMT)  # 8

_HEADER_SIZE = 2 * _CURSOR_SIZE  # 16 bytes: head + tail

# Per-slot layout: timestamp(8) + positions(n*8)
# 每槽布局: 时间戳(8) + 位置(n*8)
_SLOT_TS_FMT = "<d"  # float64 timestamp
_SLOT_TS_SIZE = struct.calcsize(_SLOT_TS_FMT)  # 8

# Max retries / 最大重试次数
_MAX_RETRIES = 10_000
_RETRY_SLEEP = 1e-6  # 1 us yield

# Try importing shared_memory; if unavailable the class becomes a graceful no-op.
# 尝试导入 shared_memory；如果不可用，类将退化为空操作。
try:
    from multiprocessing.shared_memory import SharedMemory
except ImportError:  # pragma: no cover
    SharedMemory = None  # type: ignore[assignment,misc]


def _slot_size(num_joints: int) -> int:
    """Compute the byte size of one queue slot.

    计算一个队列槽的字节大小。
    """
    return _SLOT_TS_SIZE + num_joints * 8


def _total_size(num_joints: int, buffer_size: int) -> int:
    """Compute the total shared-memory segment size.

    计算共享内存段的总大小。
    """
    return _HEADER_SIZE + buffer_size * _slot_size(num_joints)


class SharedQueue:
    """Lock-free FIFO shared memory queue for cross-process commands.

    用于跨进程命令的无锁 FIFO 共享内存队列。

    Supports zero-copy read views and batch put/get operations.  Each slot
    stores a timestamp and an array of target joint positions.  The queue
    uses head/tail cursor arithmetic for FIFO ordering: ``tail`` is the
    next write position and ``head`` is the next read position.

    支持零拷贝读取视图和批量 put/get 操作。每个槽存储一个时间戳和一组
    目标关节位置。队列使用头/尾游标算术实现 FIFO 排序：``tail`` 是
    下一个写入位置，``head`` 是下一个读取位置。

    A writer process creates an instance with ``create=True`` (the default)
    which allocates the shared-memory block.  Any number of reader processes
    can attach by constructing with ``create=False`` and the same *name*.

    写进程以 ``create=True``（默认值）创建实例，分配共享内存块。
    任意数量的读进程可以用 ``create=False`` 和相同的 *name* 来附加。

    Thread safety is provided via ``threading.Lock`` to prevent concurrent
    writers from corrupting the tail cursor.  Cross-process safety relies
    on the assumption of a single writer and a single reader process.

    线程安全通过 ``threading.Lock`` 提供，防止并发写入者破坏尾游标。
    跨进程安全依赖于单写入和单读取进程的假设。

    If ``multiprocessing.shared_memory`` is not available (e.g. Python < 3.8),
    the instance degrades gracefully: all operations become no-ops and
    :meth:`get` returns ``None``.

    如果 ``multiprocessing.shared_memory`` 不可用（例如 Python < 3.8），
    实例会优雅降级：所有操作变为空操作，:meth:`get` 返回 ``None``。

    Args:
        name: Shared-memory segment name.  Must be unique system-wide.
              共享内存段名称，必须在整个系统中唯一。
        num_joints: Number of joints per command.
                    每条命令的关节数量。
        buffer_size: Number of queue slots (default 64).
                     队列槽数（默认 64）。
        create: ``True`` (default) to *create* a new segment; ``False`` to
                attach to an existing one.
                ``True``（默认）创建新段；``False`` 附加到已有段。
    """

    def __init__(
        self,
        name: str,
        num_joints: int,
        buffer_size: int = 64,
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
                "SharedQueue('%s') will operate as a no-op. "
                "multiprocessing.shared_memory 不可用，SharedQueue 将作为空操作运行。",
                name,
            )
            return

        try:
            if create:
                self._shm = SharedMemory(
                    name=name, create=True, size=self._total_sz
                )
                self._init_queue()
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
        """Number of joints per command. / 每条命令的关节数量。"""
        return self._num_joints

    @property
    def buffer_size(self) -> int:
        """Number of queue slots. / 队列槽数。"""
        return self._buffer_size

    def put(self, timestamp: float, positions: list[float]) -> None:
        """Enqueue one command.

        入队一条命令。

        If the queue is full (``tail - head == buffer_size``), the oldest
        command is silently dropped by advancing ``head``.

        如果队列已满（``tail - head == buffer_size``），最旧的命令会被
        静默丢弃（推进 ``head``）。

        Args:
            timestamp: Command timestamp in seconds. / 命令时间戳（秒）。
            positions: Target joint positions in radians. / 目标关节位置（弧度）。
        """
        if not self._active:
            return

        n = self._num_joints
        if len(positions) < n:
            logger.warning(
                "put(): insufficient positions length (expected %d joints). Skipping. "
                "put(): 位置长度不足（期望 %d 个关节），跳过。",
                n,
                n,
            )
            return

        buf = self._shm.buf

        with self._lock:
            # Read cursors / 读取游标
            head = struct.unpack_from(_HEAD_FMT, buf, 0)[0]
            tail = struct.unpack_from(_TAIL_FMT, buf, _CURSOR_SIZE)[0]

            # Check if queue is full / 检查队列是否已满
            if tail - head >= self._buffer_size:
                # Drop oldest: advance head / 丢弃最旧命令：推进 head
                head += 1
                struct.pack_into(_HEAD_FMT, buf, 0, head)
                logger.debug(
                    "put(): queue full, dropping oldest command (head=%d). "
                    "put(): 队列已满，丢弃最旧命令 (head=%d)。",
                    head,
                    head,
                )

            # Compute slot index and write / 计算槽索引并写入
            slot_idx = tail % self._buffer_size
            slot_offset = _HEADER_SIZE + slot_idx * self._slot_sz

            # Write timestamp / 写入时间戳
            struct.pack_into(_SLOT_TS_FMT, buf, slot_offset, float(timestamp))

            # Write positions / 写入位置
            data_offset = slot_offset + _SLOT_TS_SIZE
            for i in range(n):
                struct.pack_into("<d", buf, data_offset + i * 8, float(positions[i]))

            # Advance tail / 推进尾游标
            struct.pack_into(_TAIL_FMT, buf, _CURSOR_SIZE, tail + 1)

    def put_batch(self, items: list[tuple[float, list[float]]]) -> None:
        """Enqueue multiple commands.

        入队多条命令。

        If the queue does not have enough space for all items, the oldest
        commands are silently dropped to make room.

        如果队列没有足够空间容纳所有项目，最旧的命令会被静默丢弃以腾出空间。

        Args:
            items: List of ``(timestamp, positions)`` tuples.
                   ``(timestamp, positions)`` 元组列表。
        """
        if not self._active:
            return

        for ts, pos in items:
            self.put(ts, pos)

    def get(self) -> dict[str, Any] | None:
        """Dequeue one command.  Returns ``None`` if the queue is empty.

        出队一条命令。如果队列为空，返回 ``None``。

        Returns:
            A dict with keys ``"timestamp"`` and ``"positions"``, or
            ``None`` if the queue is empty.

            包含 ``"timestamp"`` 和 ``"positions"`` 键的字典，
            如果队列为空则返回 ``None``。
        """
        if not self._active:
            return None

        buf = self._shm.buf

        # Read head / 读取头游标
        head = struct.unpack_from(_HEAD_FMT, buf, 0)[0]
        tail = struct.unpack_from(_TAIL_FMT, buf, _CURSOR_SIZE)[0]

        if head >= tail:
            return None

        # Compute slot index / 计算槽索引
        slot_idx = head % self._buffer_size
        slot_offset = _HEADER_SIZE + slot_idx * self._slot_sz

        # Read timestamp / 读取时间戳
        timestamp = struct.unpack_from(_SLOT_TS_FMT, buf, slot_offset)[0]

        # Read positions / 读取位置
        data_offset = slot_offset + _SLOT_TS_SIZE
        positions: list[float] = []
        for i in range(self._num_joints):
            val = struct.unpack_from("<d", buf, data_offset + i * 8)[0]
            positions.append(val)

        # Advance head / 推进头游标
        struct.pack_into(_HEAD_FMT, buf, 0, head + 1)

        return {
            "timestamp": timestamp,
            "positions": positions,
        }

    def get_k(self, k: int) -> list[dict[str, Any]]:
        """Dequeue up to *k* commands.

        出队最多 *k* 条命令。

        Args:
            k: Maximum number of commands to dequeue. / 最大出队命令数。

        Returns:
            List of command dicts (same format as :meth:`get`).
            May contain fewer than *k* entries if the queue has fewer items.

            命令字典列表（格式与 :meth:`get` 相同）。
            如果队列中的项目少于 *k*，可能包含少于 *k* 条记录。
        """
        if not self._active:
            return []

        results: list[dict[str, Any]] = []
        for _ in range(k):
            item = self.get()
            if item is None:
                break
            results.append(item)
        return results

    def peek(self) -> dict[str, Any] | None:
        """Peek at the front command without dequeuing.

        查看队首命令但不出队。

        Returns:
            A dict with keys ``"timestamp"`` and ``"positions"``, or
            ``None`` if the queue is empty.

            包含 ``"timestamp"`` 和 ``"positions"`` 键的字典，
            如果队列为空则返回 ``None``。
        """
        if not self._active:
            return None

        buf = self._shm.buf

        head = struct.unpack_from(_HEAD_FMT, buf, 0)[0]
        tail = struct.unpack_from(_TAIL_FMT, buf, _CURSOR_SIZE)[0]

        if head >= tail:
            return None

        slot_idx = head % self._buffer_size
        slot_offset = _HEADER_SIZE + slot_idx * self._slot_sz

        timestamp = struct.unpack_from(_SLOT_TS_FMT, buf, slot_offset)[0]

        data_offset = slot_offset + _SLOT_TS_SIZE
        positions: list[float] = []
        for i in range(self._num_joints):
            val = struct.unpack_from("<d", buf, data_offset + i * 8)[0]
            positions.append(val)

        return {
            "timestamp": timestamp,
            "positions": positions,
        }

    def qsize(self) -> int:
        """Return the number of items currently in the queue.

        返回队列中当前的项目数。
        """
        if not self._active:
            return 0
        buf = self._shm.buf
        head = struct.unpack_from(_HEAD_FMT, buf, 0)[0]
        tail = struct.unpack_from(_TAIL_FMT, buf, _CURSOR_SIZE)[0]
        return max(0, tail - head)

    def empty(self) -> bool:
        """Return ``True`` if the queue is empty.

        当队列为空时返回 ``True``。
        """
        return self.qsize() == 0

    def clear(self) -> None:
        """Reset head and tail cursors, effectively emptying the queue.

        重置头尾游标，有效清空队列。

        Note: slot data is not zeroed out, only the cursors are reset.

        注意：槽数据不会被清零，只有游标被重置。
        """
        if not self._active:
            return
        buf = self._shm.buf
        with self._lock:
            # Reset both cursors to the current tail value to maintain consistency
            # 将两个游标重置为当前尾值以保持一致性
            tail = struct.unpack_from(_TAIL_FMT, buf, _CURSOR_SIZE)[0]
            struct.pack_into(_HEAD_FMT, buf, 0, tail)

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

    def _init_queue(self) -> None:
        """Write zeros into a freshly-created segment.

        向新创建的段写入零值。
        """
        buf = self._shm.buf
        buf[:] = b"\x00" * self._total_sz

    def __repr__(self) -> str:
        status = "active" if self._active else "inactive"
        return (
            f"SharedQueue(name={self._name!r}, "
            f"num_joints={self._num_joints}, "
            f"buffer_size={self._buffer_size}, "
            f"qsize={self.qsize()}, "
            f"status={status})"
        )
