from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class TraceEvent:
    t: float
    name: str
    data: dict


class Recorder:
    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    def add(self, name: str, data: dict) -> None:
        self._events.append(TraceEvent(t=time.time(), name=name, data=data))

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in self._events], f, ensure_ascii=False, indent=2)
