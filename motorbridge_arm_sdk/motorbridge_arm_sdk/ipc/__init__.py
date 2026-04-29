"""IPC (Inter-Process Communication) utilities for MotorBridge Arm SDK.

MotorBridge Arm SDK 的进程间通信 (IPC) 工具。

This package provides shared-memory based IPC for publishing arm state
so that external processes (e.g. visualisation, logging, or secondary
controllers) can read joint data with minimal latency.

本包提供基于共享内存的 IPC，用于发布机械臂状态，使外部进程（如可视化、
日志记录或辅助控制器）能以极低延迟读取关节数据。
"""

from .shared_state import SharedArmState

__all__ = ["SharedArmState"]
