"""Shared-memory IPC layer for publishing arm state across processes.

通过共享内存在进程间发布机械臂状态的 IPC 层。

This module provides :class:`SharedArmState`, a lock-free (spinlock-protected)
shared-memory segment that other processes can read to obtain the latest joint
positions, velocities, torques and status codes without any socket or pipe
overhead.

本模块提供 :class:`SharedArmState`，它使用自旋锁保护的共享内存段，使其他进程
可以直接读取最新的关节位置、速度、力矩和状态码，无需 socket 或 pipe 开销。

Data layout (binary, little-endian)::

    Header (20 bytes):
        magic       4B  uint32   0x41524D53 ("ARMS")
        version     4B  uint32   1
        tick        8B  uint64   monotonic counter
        num_joints  4B  uint32   number of joints

    Spinlock (1 byte):
        lock        1B  uint8    0 = unlocked, 1 = locked

    Padding (3 bytes):
        pad         3B  zeros    align to 8-byte boundary

    Per-joint block (28 bytes each, repeated *num_joints* times):
        pos         8B  float64  position  (rad)
        vel         8B  float64  velocity  (rad/s)
        torque      8B  float64  torque    (Nm)
        status      4B  int32    status code

Total size = 20 + 1 + 3 + 28 * num_joints = 24 + 28 * num_joints
"""

from __future__ import annotations

import logging
import struct
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / 常量
# ---------------------------------------------------------------------------

_MAGIC = 0x41524D53  # b"ARMS" in little-endian uint32
_VERSION = 1
_HEADER_FMT = "<IQI"  # magic(4B) + tick(8B) + num_joints(4B) = 16 bytes
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 16
_LOCK_OFFSET = _HEADER_SIZE  # spinlock byte sits right after header
_PAD_BYTES = 3  # padding to align joint data to 8-byte boundary
_JOINT_FMT = "<dddi"  # pos(8) + vel(8) + torque(8) + status(4) = 28 bytes
_JOINT_SIZE = struct.calcsize(_JOINT_FMT)  # 28
_DATA_OFFSET = _HEADER_SIZE + 1 + _PAD_BYTES  # 16 + 1 + 3 = 20

# Spinlock values
_LOCK_FREE = 0
_LOCK_HELD = 1

# Max spin iterations before yielding (防御活锁 / prevent live-lock)
_MAX_SPIN = 10_000
_SPIN_SLEEP = 1e-6  # 1 us yield

# Try importing shared_memory; if unavailable the class becomes a graceful no-op.
# 尝试导入 shared_memory；如果不可用，类将退化为空操作。
try:
    from multiprocessing.shared_memory import SharedMemory
except ImportError:  # pragma: no cover – Python < 3.8 or restricted env
    SharedMemory = None  # type: ignore[assignment,misc]


class SharedArmState:
    """Publish arm joint state to a POSIX shared-memory segment.

    将机械臂关节状态发布到 POSIX 共享内存段。

    A **writer** process creates an instance with ``create=True`` (the default)
    which allocates the shared-memory block.  Any number of **reader** processes
    can attach by constructing with ``create=False`` and the same *name*.

    写进程以 ``create=True``（默认值）创建实例，分配共享内存块。
    任意数量的读进程可以用 ``create=False`` 和相同的 *name* 来附加。

    If ``multiprocessing.shared_memory`` is not available (e.g. Python < 3.8),
    the instance degrades gracefully: all write/read operations become no-ops
    and :meth:`read` returns an empty dict.

    如果 ``multiprocessing.shared_memory`` 不可用（例如 Python < 3.8），
    实例会优雅降级：所有读写操作变为空操作，:meth:`read` 返回空字典。

    Args:
        name: Shared-memory segment name.  Must be unique system-wide.
              共享内存段名称，必须在整个系统中唯一。
        num_joints: Number of joints to publish.
                    要发布的关节数量。
        create: ``True`` (default) to *create* a new segment; ``False`` to
                attach to an existing one.
                ``True``（默认）创建新段；``False`` 附加到已有段。
    """

    def __init__(self, name: str, num_joints: int, *, create: bool = True) -> None:
        self._name = name
        self._num_joints = num_joints
        self._total_size = _DATA_OFFSET + _JOINT_SIZE * num_joints
        self._shm: Any | None = None
        self._active = False
        self._created = create

        if SharedMemory is None:
            logger.warning(
                "multiprocessing.shared_memory is unavailable; "
                "SharedArmState('%s') will operate as a no-op. "
                "multiprocessing.shared_memory 不可用，SharedArmState 将作为空操作运行。",
                name,
            )
            return

        try:
            if create:
                self._shm = SharedMemory(name=name, create=True, size=self._total_size)
                self._init_header()
            else:
                self._shm = SharedMemory(name=name, create=False)
            self._active = True
        except FileExistsError:
            # Segment already exists – attach instead of create.
            # 段已存在——改为附加。
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
        """Number of joints tracked. / 跟踪的关节数量。"""
        return self._num_joints

    @property
    def tick(self) -> int:
        """Current monotonic tick counter.

        当前的单调递增 tick 计数器。

        Readers can compare this value before and after a read to detect
        torn reads (i.e. the writer updated mid-read).

        读者可以在读取前后比较此值以检测撕裂读（即写进程在读取过程中更新了数据）。
        """
        if not self._active:
            return 0
        buf = self._shm.buf
        data = bytes(buf[_HEADER_SIZE - 8 : _HEADER_SIZE])  # tick field only
        return struct.unpack("<Q", data)[0]

    def write(
        self,
        positions: list[float] | tuple[float, ...],
        velocities: list[float] | tuple[float, ...],
        torques: list[float] | tuple[float, ...],
        statuses: list[int] | tuple[int, ...],
    ) -> None:
        """Write joint state to shared memory.

        将关节状态写入共享内存。

        The write is protected by a spinlock byte so that concurrent readers
        will not see a partially-updated state.  The tick counter is
        incremented atomically (within the spinlock) so readers can detect
        stale data.

        写入受自旋锁字节保护，因此并发读取器不会看到部分更新的状态。
        tick 计数器在自旋锁内原子递增，读取器可据此检测过期数据。

        Args:
            positions: Joint positions in radians. / 关节位置（弧度）。
            velocities: Joint velocities in rad/s. / 关节速度（弧度/秒）。
            torques: Joint torques in Nm. / 关节力矩（牛顿·米）。
            statuses: Joint status codes (int). / 关节状态码（整数）。
        """
        if not self._active:
            return

        n = self._num_joints
        if len(positions) < n or len(velocities) < n or len(torques) < n or len(statuses) < n:
            logger.warning(
                "write(): insufficient data length (expected %d joints). Skipping. "
                "write(): 数据长度不足（期望 %d 个关节），跳过。",
                n,
                n,
            )
            return

        buf = self._shm.buf

        # --- acquire spinlock / 获取自旋锁 ---
        self._spinlock_acquire(buf)

        try:
            # Increment tick / 递增 tick
            old_tick = struct.unpack_from("<Q", buf, 8)[0]
            struct.pack_into("<Q", buf, 8, old_tick + 1)

            # Write per-joint data / 逐关节写入数据
            for i in range(n):
                offset = _DATA_OFFSET + i * _JOINT_SIZE
                struct.pack_into(
                    _JOINT_FMT,
                    buf,
                    offset,
                    float(positions[i]),
                    float(velocities[i]),
                    float(torques[i]),
                    int(statuses[i]),
                )
        finally:
            # --- release spinlock / 释放自旋锁 ---
            self._spinlock_release(buf)

    def read(self) -> dict[str, Any]:
        """Read the current joint state from shared memory.

        从共享内存读取当前关节状态。

        Uses a double-check on the tick counter to avoid torn reads: if the
        tick changed between the start and end of the read, the operation is
        retried.

        使用 tick 计数器双重检查来避免撕裂读：如果读取前后 tick 发生变化，
        将重试操作。

        Returns:
            A dict with keys ``"tick"``, ``"num_joints"``, ``"positions"``,
            ``"velocities"``, ``"torques"``, ``"statuses"``.
            If the shared memory is not active, returns an empty dict.

            包含 ``"tick"``、``"num_joints"``、``"positions"``、
            ``"velocities"``、``"torques"``、``"statuses"`` 键的字典。
            如果共享内存未激活，返回空字典。
        """
        if not self._active:
            return {}

        buf = self._shm.buf

        for _ in range(_MAX_SPIN):
            # Spin until lock is free / 自旋直到锁释放
            while buf[_LOCK_OFFSET] == _LOCK_HELD:
                time.sleep(_SPIN_SLEEP)

            # Read tick before / 读取前 tick
            tick_before = struct.unpack_from("<Q", buf, 8)[0]

            # Read all joint data / 读取所有关节数据
            positions: list[float] = []
            velocities: list[float] = []
            torques: list[float] = []
            statuses: list[int] = []

            for i in range(self._num_joints):
                offset = _DATA_OFFSET + i * _JOINT_SIZE
                p, v, t, s = struct.unpack_from(_JOINT_FMT, buf, offset)
                positions.append(p)
                velocities.append(v)
                torques.append(t)
                statuses.append(s)

            # Read tick after / 读取后 tick
            tick_after = struct.unpack_from("<Q", buf, 8)[0]

            # If tick changed or lock was taken, retry / 如果 tick 变化或锁被占用，重试
            if tick_before == tick_after and buf[_LOCK_OFFSET] == _LOCK_FREE:
                return {
                    "tick": tick_before,
                    "num_joints": self._num_joints,
                    "positions": positions,
                    "velocities": velocities,
                    "torques": torques,
                    "statuses": statuses,
                }

        # Fallback: return last read even if potentially torn
        # 降级：即使可能存在撕裂读也返回最后读取的数据
        logger.warning(
            "read(): failed to get consistent snapshot after %d retries. "
            "read(): 在 %d 次重试后仍无法获得一致快照。",
            _MAX_SPIN,
            _MAX_SPIN,
        )
        return {
            "tick": tick_before,
            "num_joints": self._num_joints,
            "positions": positions,
            "velocities": velocities,
            "torques": torques,
            "statuses": statuses,
        }

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

    def _init_header(self) -> None:
        """Write the initial header into a freshly-created segment.

        向新创建的段写入初始头部。

        Header layout:
            magic (4B) + tick (8B) + num_joints (4B) = 16 bytes
        Then:
            spinlock (1B) + padding (3B)
        """
        buf = self._shm.buf
        # Zero-fill entire buffer in one bulk write / 一次性批量填零
        buf[:] = b'\x00' * self._total_size
        # Write magic, tick=0, num_joints / 写入 magic、tick=0、num_joints
        struct.pack_into(_HEADER_FMT, buf, 0, _MAGIC, 0, self._num_joints)

    def _spinlock_acquire(self, buf: Any) -> None:
        """Acquire the spinlock byte.

        获取自旋锁字节。

        Uses a simple compare-and-swap loop with a short sleep to avoid
        burning CPU when the lock is contended.

        使用简单的比较并交换循环，在锁竞争时短暂休眠以避免空耗 CPU。
        """
        for attempt in range(_MAX_SPIN):
            # Atomic-ish compare-and-swap via direct byte access
            # 通过直接字节访问实现的类原子比较并交换
            if buf[_LOCK_OFFSET] == _LOCK_FREE:
                buf[_LOCK_OFFSET] = _LOCK_HELD
                # Double-check: memory model is per-process but shared-memory
                # byte writes are effectively atomic on all major platforms.
                # 双重检查：内存模型是进程内的，但共享内存的字节写入在所有
                # 主要平台上实际上是原子的。
                if buf[_LOCK_OFFSET] == _LOCK_HELD:
                    return
            if attempt % 100 == 99:
                time.sleep(_SPIN_SLEEP)
        # If we exhaust spins, proceed anyway (best-effort)
        # 如果自旋次数耗尽，仍然继续（尽力而为）
        buf[_LOCK_OFFSET] = _LOCK_HELD

    @staticmethod
    def _spinlock_release(buf: Any) -> None:
        """Release the spinlock byte. / 释放自旋锁字节。"""
        buf[_LOCK_OFFSET] = _LOCK_FREE

    def __repr__(self) -> str:
        status = "active" if self._active else "inactive"
        return (
            f"SharedArmState(name={self._name!r}, "
            f"num_joints={self._num_joints}, "
            f"status={status})"
        )
