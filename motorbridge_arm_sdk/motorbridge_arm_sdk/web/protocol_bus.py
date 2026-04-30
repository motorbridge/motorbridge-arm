from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BusConfig:
    tx_buffer_size: int = 1024
    rx_buffer_size: int = 1024
    channels: dict[str, bool] = field(
        default_factory=lambda: {
            "websocket": True,
            "sim": True,
            "motorbridge_py": True,
            "ros": False,
        }
    )


class ProtocolBus:
    def __init__(self, config: BusConfig | None = None) -> None:
        self.config = config or BusConfig()
        self._tx: deque[dict[str, Any]] = deque(maxlen=self.config.tx_buffer_size)
        self._rx: deque[dict[str, Any]] = deque(maxlen=self.config.rx_buffer_size)
        self._listeners: list[Callable[[str, dict[str, Any]], None]] = []
        self._lock = threading.Lock()

    def set_config(
        self,
        *,
        tx_buffer_size: int | None = None,
        rx_buffer_size: int | None = None,
        channels: dict[str, bool] | None = None,
    ) -> None:
        with self._lock:
            if tx_buffer_size is not None and tx_buffer_size > 0 and tx_buffer_size != self.config.tx_buffer_size:
                self.config.tx_buffer_size = tx_buffer_size
                self._tx = deque(self._tx, maxlen=tx_buffer_size)
            if rx_buffer_size is not None and rx_buffer_size > 0 and rx_buffer_size != self.config.rx_buffer_size:
                self.config.rx_buffer_size = rx_buffer_size
                self._rx = deque(self._rx, maxlen=rx_buffer_size)
            if channels:
                self.config.channels.update({str(k): bool(v) for k, v in channels.items()})

    def add_listener(self, fn: Callable[[str, dict[str, Any]], None]) -> None:
        with self._lock:
            self._listeners.append(fn)

    def remove_listener(self, fn: Callable[[str, dict[str, Any]], None]) -> None:
        with self._lock:
            try:
                self._listeners.remove(fn)
            except ValueError:
                pass

    def _dispatch(self, direction: str, item: dict[str, Any]) -> None:
        """Call each listener with individual exception protection."""
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(direction, item)
            except Exception:
                logger.exception("ProtocolBus listener %r raised an exception", fn)

    def publish_rx(self, channel: str, payload: dict[str, Any]) -> None:
        if not self.config.channels.get(channel, False):
            return
        item = {"ts": time.time(), "channel": channel, "dir": "rx", "payload": payload}
        with self._lock:
            self._rx.append(item)
        self._dispatch("rx", item)

    def publish_tx(self, channel: str, payload: dict[str, Any]) -> None:
        if not self.config.channels.get(channel, False):
            return
        item = {"ts": time.time(), "channel": channel, "dir": "tx", "payload": payload}
        with self._lock:
            self._tx.append(item)
        self._dispatch("tx", item)

    def snapshot(self, limit: int = 100) -> dict[str, Any]:
        n = max(1, int(limit))
        with self._lock:
            return {
                "config": {
                    "tx_buffer_size": self.config.tx_buffer_size,
                    "rx_buffer_size": self.config.rx_buffer_size,
                    "channels": dict(self.config.channels),
                },
                "tx_count": len(self._tx),
                "rx_count": len(self._rx),
                "tx_recent": list(self._tx)[-n:],
                "rx_recent": list(self._rx)[-n:],
            }
