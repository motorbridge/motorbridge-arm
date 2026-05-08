from __future__ import annotations

import threading
from unittest.mock import MagicMock

from motorbridge_arm_sdk.motion.executor import JointMotionExecutor


def test_run_sends_all_points():
    ex = JointMotionExecutor(dt_s=0.001)
    send_fn = MagicMock()
    points = [[float(i), float(i)] for i in range(5)]
    ex.run(points, send_fn, vlim=1.0)
    assert send_fn.call_count == 5


def test_run_with_abort():
    ex = JointMotionExecutor(dt_s=0.001)
    send_fn = MagicMock()
    abort = threading.Event()
    abort.set()
    points = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    ex.run(points, send_fn, vlim=1.0, abort_event=abort)
    assert send_fn.call_count == 0


def test_interpolate_linear_produces_correct_count():
    ex = JointMotionExecutor()
    q0 = [0.0, 0.0, 0.0]
    q1 = [1.0, 2.0, 3.0]
    pts = ex.interpolate_linear(q0, q1, steps=10)
    assert len(pts) == 10


def test_interpolate_linear_endpoints():
    ex = JointMotionExecutor()
    q0 = [0.1, -0.5, 0.3]
    q1 = [1.0, 0.5, -0.3]
    pts = ex.interpolate_linear(q0, q1, steps=5)
    assert pts[0] == q0
    assert pts[-1] == q1
