from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable


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

    def set_config(
        self,
        *,
        tx_buffer_size: int | None = None,
        rx_buffer_size: int | None = None,
        channels: dict[str, bool] | None = None,
    ) -> None:
        if tx_buffer_size is not None and tx_buffer_size > 0 and tx_buffer_size != self.config.tx_buffer_size:
            self.config.tx_buffer_size = tx_buffer_size
            self._tx = deque(self._tx, maxlen=tx_buffer_size)
        if rx_buffer_size is not None and rx_buffer_size > 0 and rx_buffer_size != self.config.rx_buffer_size:
            self.config.rx_buffer_size = rx_buffer_size
            self._rx = deque(self._rx, maxlen=rx_buffer_size)
        if channels:
            self.config.channels.update({str(k): bool(v) for k, v in channels.items()})

    def add_listener(self, fn: Callable[[str, dict[str, Any]], None]) -> None:
        self._listeners.append(fn)

    def publish_rx(self, channel: str, payload: dict[str, Any]) -> None:
        if not self.config.channels.get(channel, False):
            return
        item = {"ts": time.time(), "channel": channel, "dir": "rx", "payload": payload}
        self._rx.append(item)
        for fn in list(self._listeners):
            fn("rx", item)

    def publish_tx(self, channel: str, payload: dict[str, Any]) -> None:
        if not self.config.channels.get(channel, False):
            return
        item = {"ts": time.time(), "channel": channel, "dir": "tx", "payload": payload}
        self._tx.append(item)
        for fn in list(self._listeners):
            fn("tx", item)

    def snapshot(self, limit: int = 100) -> dict[str, Any]:
        n = max(1, int(limit))
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
