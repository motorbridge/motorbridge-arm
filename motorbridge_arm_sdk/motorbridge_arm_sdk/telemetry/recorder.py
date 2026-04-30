from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TraceEvent:
    t: float
    name: str
    data: dict


class Recorder:
    def __init__(self, max_events: int = 10000) -> None:
        self._events: deque[TraceEvent] = deque(maxlen=max_events)
        self._overflow_warned = False
        self._lock = threading.Lock()

    def add(self, name: str, data: dict) -> None:
        with self._lock:
            if len(self._events) == self._events.maxlen and not self._overflow_warned:
                logger.warning("Recorder reached max_events=%d; oldest events will be dropped", self._events.maxlen)
                self._overflow_warned = True
            self._events.append(TraceEvent(t=time.time(), name=name, data=data))

    def save_json(self, path: str) -> None:
        with self._lock:
            logger.info("save_json(%s) — %d events", path, len(self._events))
            fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=os.path.dirname(path) or ".")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump([asdict(e) for e in self._events], f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, path)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
