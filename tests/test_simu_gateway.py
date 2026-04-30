"""Tests for the WebSocket gateway dispatch logic."""
import asyncio
from unittest.mock import MagicMock, patch

import pytest

from motorbridge_arm_sdk.types import Pose6D


def _run(coro):
    return asyncio.run(coro)


def _make_gateway():
    """Create a SimuWsGateway with a mocked SimArm."""
    with patch("motorbridge_arm_sdk.web.simu_gateway.SimArm") as MockSim:
        mock_sim = MagicMock()
        mock_sim.get_joint_positions.return_value = [0.0] * 6
        mock_sim.get_pose.return_value = Pose6D(x=0.3, y=0.0, z=0.4, roll=0.0, pitch=0.0, yaw=0.0)
        MockSim.return_value = mock_sim

        from motorbridge_arm_sdk.web.simu_gateway import SimuWsGateway
        gw = SimuWsGateway()
        return gw, mock_sim


def test_dispatch_ping():
    gw, _ = _make_gateway()
    result = _run(gw._dispatch({"op": "ping", "req_id": 42}))
    assert result["ok"] is True
    assert result["req_id"] == 42
    assert "ts" in result["data"]


def test_dispatch_state():
    gw, mock_sim = _make_gateway()
    result = _run(gw._dispatch({"op": "state", "req_id": 1}))
    assert result["ok"] is True
    data = result["data"]
    assert "q" in data
    assert "pose" in data
    assert "waypoints" in data
    mock_sim.get_joint_positions.assert_called()
    mock_sim.get_pose.assert_called()


def test_dispatch_set_joint_targets():
    gw, mock_sim = _make_gateway()
    result = _run(
        gw._dispatch({
            "op": "sim_set_joint_targets",
            "req_id": 2,
            "targets": {"joint1": 0.5, "joint3": -0.3},
        })
    )
    assert result["ok"] is True
    # __init__ calls set_joint_positions once for startup posture, dispatch calls again
    assert mock_sim.set_joint_positions.call_count == 2
    q_set = mock_sim.set_joint_positions.call_args[0][0]
    assert abs(q_set[0] - 0.5) < 1e-9
    assert abs(q_set[2] - (-0.3)) < 1e-9


def test_dispatch_waypoint_add():
    gw, _ = _make_gateway()
    result = _run(
        gw._dispatch({
            "op": "waypoint_add",
            "req_id": 3,
            "id": "wp1",
            "pose": {"x": 0.3, "y": 0.1, "z": 0.4, "roll": 0, "pitch": 0, "yaw": 0},
        })
    )
    assert result["ok"] is True
    assert "wp1" in result["data"]["waypoints"]
    assert abs(result["data"]["waypoints"]["wp1"]["x"] - 0.3) < 1e-9


def test_dispatch_waypoint_add_missing_id():
    gw, _ = _make_gateway()
    result = _run(gw._dispatch({"op": "waypoint_add", "req_id": 4, "id": ""}))
    assert result["ok"] is False
    assert result["error"] == "missing_id"


def test_dispatch_waypoint_update():
    gw, _ = _make_gateway()
    # Add first
    _run(
        gw._dispatch({
            "op": "waypoint_add",
            "req_id": 5,
            "id": "wp1",
            "pose": {"x": 0.1, "y": 0.0, "z": 0.0, "roll": 0, "pitch": 0, "yaw": 0},
        })
    )
    # Update
    result = _run(
        gw._dispatch({
            "op": "waypoint_update",
            "req_id": 6,
            "id": "wp1",
            "pose": {"x": 0.9},
        })
    )
    assert result["ok"] is True
    assert abs(result["data"]["waypoints"]["wp1"]["x"] - 0.9) < 1e-9


def test_dispatch_waypoint_update_not_found():
    gw, _ = _make_gateway()
    result = _run(gw._dispatch({"op": "waypoint_update", "req_id": 7, "id": "nonexistent"}))
    assert result["ok"] is False
    assert result["error"] == "waypoint_not_found"


def test_dispatch_waypoint_remove():
    gw, _ = _make_gateway()
    _run(
        gw._dispatch({
            "op": "waypoint_add", "req_id": 8, "id": "wp1",
            "pose": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
        })
    )
    result = _run(gw._dispatch({"op": "waypoint_remove", "req_id": 9, "id": "wp1"}))
    assert result["ok"] is True
    assert "wp1" not in result["data"]["waypoints"]


def test_dispatch_waypoint_clear():
    gw, _ = _make_gateway()
    _run(
        gw._dispatch({
            "op": "waypoint_add", "req_id": 10, "id": "a",
            "pose": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
        })
    )
    _run(
        gw._dispatch({
            "op": "waypoint_add", "req_id": 11, "id": "b",
            "pose": {"x": 1, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
        })
    )
    result = _run(gw._dispatch({"op": "waypoint_clear", "req_id": 12}))
    assert result["ok"] is True
    assert len(result["data"]["waypoints"]) == 0


def test_dispatch_waypoint_list():
    gw, _ = _make_gateway()
    result = _run(gw._dispatch({"op": "waypoint_list", "req_id": 13}))
    assert result["ok"] is True
    assert "waypoints" in result["data"]


def test_dispatch_unknown_op():
    gw, _ = _make_gateway()
    result = _run(gw._dispatch({"op": "nonexistent", "req_id": 99}))
    assert result["ok"] is False
    assert result["error"] == "unknown_op"


def test_dispatch_sim_stop():
    gw, _ = _make_gateway()
    result = _run(gw._dispatch({"op": "sim_stop", "req_id": 50}))
    assert result["ok"] is True
    assert result["data"]["status"]["running"] is False


def test_dispatch_sim_run_waypoints_missing():
    gw, _ = _make_gateway()
    result = _run(
        gw._dispatch({
            "op": "sim_run_waypoints",
            "req_id": 51,
            "from_id": "missing1",
            "to_id": "missing2",
        })
    )
    assert result["ok"] is False
    assert result["error"] == "waypoint_not_found"


def test_dispatch_sim_run_sequence_invalid_ids():
    gw, _ = _make_gateway()
    result = _run(
        gw._dispatch({
            "op": "sim_run_sequence",
            "req_id": 52,
            "ids": ["only_one"],
        })
    )
    assert result["ok"] is False
    assert result["error"] == "invalid_ids"


def test_snapshot_state():
    gw, mock_sim = _make_gateway()
    snap = gw._snapshot_state()
    assert "q" in snap
    assert "pose" in snap
    assert "waypoints" in snap
    assert "motion" in snap
    assert "ts" in snap
    assert len(snap["q"]) == 6
