"""Shared protocol for controllers with lifecycle management.

为具有生命周期管理的控制器提供共享协议。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable, TypeVar


@runtime_checkable
class LifecycleController(Protocol):
    """Protocol for controllers that support start/stop lifecycle and context-manager usage.

    支持启动/停止生命周期和上下文管理器使用的控制器协议。

    All controllers with background threads or hardware state transitions should
    implement this protocol to allow interchangeable use.

    所有具有后台线程或硬件状态转换的控制器都应实现此协议，
    以允许互换使用。

    Usage::

        def run(ctrl: LifecycleController) -> None:
            with ctrl:
                # ... use controller ...
                pass
    """

    def start(self) -> None:
        """Start the controller. / 启动控制器。"""
        ...

    def stop(self) -> None:
        """Stop the controller. / 停止控制器。"""
        ...

    def __enter__(self) -> "LifecycleController":
        ...

    def __exit__(self, *args) -> None:
        ...
