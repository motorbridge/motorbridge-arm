from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from typing import Any

from ..model.profiles import rebot_arm02_dm_with_gripper

logger = logging.getLogger(__name__)

from ..sim import SimArm
from ..types import Pose6D
from .protocol_bus import ProtocolBus

_MAX_WAYPOINTS = 512


class SimuWsGateway:
    def __init__(self, host: str = "127.0.0.1", port: int = 9011, path: str = "/ws") -> None:
        self.host = host
        self.port = port
        self.path = path
        self._clients: set[Any] = set()
        self._cfg = rebot_arm02_dm_with_gripper()
        self._sim = SimArm(self._cfg)
        # Force simulator startup posture to all-zero for deterministic web/CLI initial state.
        self._sim.set_joint_positions([0.0 for _ in self._sim.get_joint_positions()])
        self._state_task: asyncio.Task | None = None
        self._bus = ProtocolBus()
        self._waypoints: dict[str, dict[str, Any]] = {}
        self._motion_task: asyncio.Task | None = None
        self._motion_stop = False
        self._motion_status: dict[str, Any] = {"running": False, "name": "idle"}

    async def start(self) -> None:
        try:
            from websockets.server import serve
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("websockets dependency missing; install with: pip install websockets") from exc

        async with serve(self._handler, self.host, self.port, ping_interval=20, ping_timeout=20):
            self._state_task = asyncio.create_task(self._state_publisher())
            logger.info("listening ws://%s:%d%s", self.host, self.port, self.path)
            try:
                await asyncio.Future()
            finally:
                if self._state_task:
                    self._state_task.cancel()

    async def _handler(self, ws) -> None:
        self._clients.add(ws)
        logger.info("client connected — %d clients total", len(self._clients))
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    await self._send_json(ws, {"ok": False, "error": "invalid_json"})
                    continue
                self._bus.publish_rx("websocket", msg)
                resp = await self._dispatch(msg)
                if resp is not None:
                    await self._send_json(ws, resp)
        finally:
            self._clients.discard(ws)
            logger.info("client disconnected — %d clients remaining", len(self._clients))

    async def _dispatch(self, req: dict[str, Any]) -> dict[str, Any] | None:
        op = str(req.get("op") or "")
        req_id = req.get("req_id")

        if op == "ping":
            return {"ok": True, "req_id": req_id, "op": op, "data": {"ts": time.time()}}

        if op == "state":
            return {"ok": True, "req_id": req_id, "op": op, "data": self._snapshot_state()}

        if op == "sim_set_joint_targets":
            targets = req.get("targets") or {}
            q = self._sim.get_joint_positions()
            joint_names = [j.name for j in self._cfg.joints]
            for i in range(len(q)):
                key = joint_names[i] if i < len(joint_names) else f"joint{i+1}"
                if key in targets:
                    try:
                        q[i] = float(targets[key])
                    except Exception:
                        pass
            # Browser UI compatibility: joint7 is the single gripper slider.
            if "joint7" in targets:
                try:
                    opening = float(targets["joint7"])
                    for name in ("gripper_joint1", "gripper_joint2"):
                        if name in joint_names:
                            q[joint_names.index(name)] = opening
                except Exception:
                    pass
            self._sim.set_joint_positions(q)
            self._bus.publish_tx("sim", {"event": "sim_set_joint_targets", "targets": targets})
            return {"ok": True, "req_id": req_id, "op": op, "data": self._snapshot_state()}

        if op == "sim_move_l":
            pose = req.get("target_pose") or {}
            duration = float(req.get("duration_s") or 2.0)
            target = Pose6D(
                x=float(pose.get("x", 0.0)),
                y=float(pose.get("y", 0.0)),
                z=float(pose.get("z", 0.0)),
                roll=float(pose.get("roll", 0.0)),
                pitch=float(pose.get("pitch", 0.0)),
                yaw=float(pose.get("yaw", 0.0)),
            )
            traj = self._sim.move_l(target=target, duration_s=duration)
            self._bus.publish_tx("sim", {"event": "sim_move_l", "duration_s": duration, "points": len(traj.points)})
            return {
                "ok": True,
                "req_id": req_id,
                "op": op,
                "data": {
                    "points": len(traj.points),
                    "duration_s": traj.duration_s,
                    "final": self._snapshot_state(),
                },
            }
        if op == "waypoint_add":
            wid = str(req.get("id") or "").strip()
            pose = req.get("pose") or {}
            if not wid:
                return {"ok": False, "req_id": req_id, "op": op, "error": "missing_id"}
            if len(self._waypoints) >= _MAX_WAYPOINTS:
                return {"ok": False, "req_id": req_id, "op": op, "error": "max_waypoints_reached"}
            label = str(req.get("label") or pose.get("label") or pose.get("name") or wid).strip() or wid
            self._waypoints[wid] = {
                "label": label,
                "x": float(pose.get("x", 0.0)),
                "y": float(pose.get("y", 0.0)),
                "z": float(pose.get("z", 0.0)),
                "roll": float(pose.get("roll", 0.0)),
                "pitch": float(pose.get("pitch", 0.0)),
                "yaw": float(pose.get("yaw", 0.0)),
            }
            await self._broadcast_event("waypoint", {"event": "added", "id": wid, "pose": self._waypoints[wid]})
            return {"ok": True, "req_id": req_id, "op": op, "data": {"waypoints": self._waypoints}}
        if op == "waypoint_update":
            wid = str(req.get("id") or "").strip()
            if not wid or wid not in self._waypoints:
                return {"ok": False, "req_id": req_id, "op": op, "error": "waypoint_not_found"}
            pose = req.get("pose") or {}
            prev = self._waypoints[wid]
            label = str(req.get("label") or pose.get("label") or pose.get("name") or prev.get("label") or wid).strip() or wid
            self._waypoints[wid] = {
                "label": label,
                "x": float(pose.get("x", prev["x"])),
                "y": float(pose.get("y", prev["y"])),
                "z": float(pose.get("z", prev["z"])),
                "roll": float(pose.get("roll", prev["roll"])),
                "pitch": float(pose.get("pitch", prev["pitch"])),
                "yaw": float(pose.get("yaw", prev["yaw"])),
            }
            await self._broadcast_event("waypoint", {"event": "updated", "id": wid, "pose": self._waypoints[wid]})
            return {"ok": True, "req_id": req_id, "op": op, "data": {"waypoints": self._waypoints}}
        if op == "waypoint_remove":
            wid = str(req.get("id") or "").strip()
            if wid in self._waypoints:
                del self._waypoints[wid]
                await self._broadcast_event("waypoint", {"event": "removed", "id": wid})
            return {"ok": True, "req_id": req_id, "op": op, "data": {"waypoints": self._waypoints}}
        if op == "waypoint_clear":
            self._waypoints.clear()
            await self._broadcast_event("waypoint", {"event": "cleared"})
            return {"ok": True, "req_id": req_id, "op": op, "data": {"waypoints": self._waypoints}}
        if op == "waypoint_list":
            return {"ok": True, "req_id": req_id, "op": op, "data": {"waypoints": self._waypoints}}
        if op == "sim_run_waypoints":
            from_id = str(req.get("from_id") or "")
            to_id = str(req.get("to_id") or "")
            duration_s = float(req.get("duration_s") or 2.0)
            profile = str(req.get("profile") or "min_jerk")
            if from_id not in self._waypoints or to_id not in self._waypoints:
                return {"ok": False, "req_id": req_id, "op": op, "error": "waypoint_not_found"}
            if self._motion_task and not self._motion_task.done():
                return {"ok": False, "req_id": req_id, "op": op, "error": "motion_busy"}
            self._motion_stop = False
            self._motion_status = {
                "running": True,
                "name": "sim_run_waypoints",
                "from_id": from_id,
                "to_id": to_id,
                "profile": profile,
            }
            await self._broadcast_event("task", {"event": "accepted", "task": self._motion_status})
            self._motion_task = asyncio.create_task(self._run_waypoints(from_id, to_id, duration_s, profile))
            return {"ok": True, "req_id": req_id, "op": op, "data": {"status": self._motion_status}}
        if op == "sim_run_sequence":
            ids = req.get("ids") or []
            duration_s = float(req.get("duration_s") or 2.0)
            profile = str(req.get("profile") or "min_jerk")
            if not isinstance(ids, list) or len(ids) < 2:
                return {"ok": False, "req_id": req_id, "op": op, "error": "invalid_ids"}
            if any(str(i) not in self._waypoints for i in ids):
                return {"ok": False, "req_id": req_id, "op": op, "error": "waypoint_not_found"}
            if self._motion_task and not self._motion_task.done():
                return {"ok": False, "req_id": req_id, "op": op, "error": "motion_busy"}
            self._motion_stop = False
            self._motion_status = {
                "running": True,
                "name": "sim_run_sequence",
                "ids": [str(i) for i in ids],
                "profile": profile,
            }
            await self._broadcast_event("task", {"event": "accepted", "task": self._motion_status})
            self._motion_task = asyncio.create_task(self._run_sequence([str(i) for i in ids], duration_s, profile))
            return {"ok": True, "req_id": req_id, "op": op, "data": {"status": self._motion_status}}
        if op == "sim_stop":
            self._motion_stop = True
            self._motion_status = {"running": False, "name": "stopped"}
            if self._motion_task and not self._motion_task.done():
                self._motion_task.cancel()
            await self._broadcast_event("task", {"event": "stopped", "task": self._motion_status})
            return {"ok": True, "req_id": req_id, "op": op, "data": {"status": self._motion_status}}
        if op == "bus_config":
            tx_size = req.get("tx_buffer_size")
            rx_size = req.get("rx_buffer_size")
            channels = req.get("channels")
            self._bus.set_config(
                tx_buffer_size=int(tx_size) if tx_size is not None else None,
                rx_buffer_size=int(rx_size) if rx_size is not None else None,
                channels=channels if isinstance(channels, dict) else None,
            )
            return {"ok": True, "req_id": req_id, "op": op, "data": self._bus.snapshot(limit=20)}

        if op == "bus_snapshot":
            limit = int(req.get("limit") or 100)
            return {"ok": True, "req_id": req_id, "op": op, "data": self._bus.snapshot(limit=limit)}

        return {"ok": False, "req_id": req_id, "op": op, "error": "unknown_op"}

    def _snapshot_state(self) -> dict[str, Any]:
        q = self._sim.get_joint_positions()
        p = self._sim.get_pose()
        joint_names = [j.name for j in self._cfg.joints]
        joint_targets = {
            (joint_names[i] if i < len(joint_names) else f"joint{i+1}"): q[i]
            for i in range(len(q))
        }
        # Keep a compact alias for the current /simu UI gripper slider.
        if "gripper_joint1" in joint_targets:
            joint_targets["joint7"] = joint_targets["gripper_joint1"]
        return {
            "q": q,
            "joint_targets": joint_targets,
            "pose": asdict(p),
            "waypoints": self._waypoints,
            "motion": self._motion_status,
            "ts": time.time(),
        }

    def _waypoint_pose(self, wid: str) -> Pose6D:
        p = self._waypoints[wid]
        return Pose6D(
            x=float(p.get("x", 0.0)),
            y=float(p.get("y", 0.0)),
            z=float(p.get("z", 0.0)),
            roll=float(p.get("roll", 0.0)),
            pitch=float(p.get("pitch", 0.0)),
            yaw=float(p.get("yaw", 0.0)),
        )

    async def _run_waypoints(self, from_id: str, to_id: str, duration_s: float, profile: str) -> None:
        try:
            # Re-validate waypoints still exist before starting motion.
            if from_id not in self._waypoints:
                raise ValueError(f"start waypoint {from_id!r} no longer exists")
            if to_id not in self._waypoints:
                raise ValueError(f"end waypoint {to_id!r} no longer exists")

            pose1 = self._waypoint_pose(from_id)
            pose2 = self._waypoint_pose(to_id)
            q1 = self._sim.solve_ik(pose1)
            self._sim.move_j(q1)
            traj = self._sim.plan_l(pose2, duration_s=duration_s, profile=profile)
            loop_dt = self._sim.loop_dt_s
            for pt in traj.points:
                if self._motion_stop:
                    self._motion_status = {"running": False, "name": "stopped", "from_id": from_id, "to_id": to_id}
                    await self._broadcast_event("task", {"event": "stopped", "task": self._motion_status})
                    return
                self._sim.set_joint_positions(pt.q)
                await asyncio.sleep(max(0.001, loop_dt))
            self._motion_status = {"running": False, "name": "done", "from_id": from_id, "to_id": to_id}
            await self._broadcast_event("task", {"event": "done", "task": self._motion_status})
        except asyncio.CancelledError:
            self._motion_status = {"running": False, "name": "stopped", "from_id": from_id, "to_id": to_id}
            await self._broadcast_event("task", {"event": "stopped", "task": self._motion_status})
            raise
        except Exception as exc:
            self._motion_status = {"running": False, "name": "error", "error": str(exc)}
            await self._broadcast_event("task", {"event": "error", "task": self._motion_status})

    async def _run_sequence(self, ids: list[str], duration_s: float, profile: str) -> None:
        try:
            if len(ids) < 2:
                self._motion_status = {"running": False, "name": "done", "ids": ids}
                await self._broadcast_event("task", {"event": "done", "task": self._motion_status})
                return

            # Re-validate all waypoints still exist before starting motion.
            for wid in ids:
                if wid not in self._waypoints:
                    raise ValueError(f"waypoint {wid!r} no longer exists")

            # Align once to the first point, then move continuously point-to-point.
            first_pose = self._waypoint_pose(ids[0])
            q0 = self._sim.solve_ik(first_pose)
            self._sim.move_j(q0)

            loop_dt = self._sim.loop_dt_s
            for i in range(1, len(ids)):
                if self._motion_stop:
                    self._motion_status = {"running": False, "name": "stopped", "ids": ids}
                    await self._broadcast_event("task", {"event": "stopped", "task": self._motion_status})
                    return

                to_id = ids[i]
                to_pose = self._waypoint_pose(to_id)
                traj = self._sim.plan_l(to_pose, duration_s=duration_s, profile=profile)
                for pt in traj.points:
                    if self._motion_stop:
                        self._motion_status = {"running": False, "name": "stopped", "ids": ids}
                        await self._broadcast_event("task", {"event": "stopped", "task": self._motion_status})
                        return
                    self._sim.set_joint_positions(pt.q)
                    await asyncio.sleep(max(0.001, loop_dt))
            self._motion_status = {"running": False, "name": "done", "ids": ids}
            await self._broadcast_event("task", {"event": "done", "task": self._motion_status})
        except asyncio.CancelledError:
            self._motion_status = {"running": False, "name": "stopped", "ids": ids}
            await self._broadcast_event("task", {"event": "stopped", "task": self._motion_status})
            raise
        except Exception as exc:
            self._motion_status = {"running": False, "name": "error", "error": str(exc)}
            await self._broadcast_event("task", {"event": "error", "task": self._motion_status})

    async def _state_publisher(self) -> None:
        while True:
            await asyncio.sleep(0.05)
            if not self._clients:
                continue
            state_msg = {"type": "state", "data": self._snapshot_state()}
            dead = []
            for ws in list(self._clients):
                try:
                    await self._send_json(ws, state_msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)

    async def _send_json(self, ws, payload: dict[str, Any]) -> None:
        self._bus.publish_tx("websocket", payload)
        await ws.send(json.dumps(payload))

    async def _broadcast_event(self, event_type: str, data: dict[str, Any]) -> None:
        payload = {"type": event_type, "data": data, "ts": time.time()}
        dead = []
        for ws in list(self._clients):
            try:
                await self._send_json(ws, payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)


def main() -> int:
    gateway = SimuWsGateway()
    asyncio.run(gateway.start())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
