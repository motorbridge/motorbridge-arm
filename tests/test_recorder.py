from __future__ import annotations

import json
import logging
import os
import tempfile

from motorbridge_arm_sdk.telemetry.recorder import Recorder


def test_add_and_save():
    rec = Recorder()
    for i in range(5):
        rec.add(f"event_{i}", {"idx": i})

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        rec.save_json(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 5
        for i, entry in enumerate(data):
            assert entry["name"] == f"event_{i}"
            assert entry["data"] == {"idx": i}
    finally:
        os.unlink(path)


def test_memory_limit():
    rec = Recorder(max_events=5)
    for i in range(10):
        rec.add(f"event_{i}", {"idx": i})
    assert len(rec._events) == 5
    # The deque keeps the *last* 5 (indices 5-9)
    names = [e.name for e in rec._events]
    assert names == [f"event_{i}" for i in range(5, 10)]


def test_overflow_warning(caplog):
    rec = Recorder(max_events=3)
    with caplog.at_level(logging.WARNING, logger="motorbridge_arm_sdk.telemetry.recorder"):
        for i in range(5):
            rec.add(f"event_{i}", {"idx": i})
    assert any("max_events" in r.message for r in caplog.records)
