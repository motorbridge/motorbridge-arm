#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any


async def ws_request(url: str, op: str, payload: dict[str, Any]) -> dict[str, Any]:
    from websockets.client import connect

    req = {"op": op, "req_id": 1}
    req.update(payload)
    async with connect(url, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps(req))
        raw = await ws.recv()
        return json.loads(raw)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simu WS CLI for motorbridge-arm")
    p.add_argument("--url", default="ws://127.0.0.1:9011/ws", help="WS gateway url")
    sub = p.add_subparsers(dest="cmd", required=True)

    add = sub.add_parser("waypoint-add", help="add/update waypoint")
    add.add_argument("--id", required=True)
    add.add_argument("--label", default="")
    add.add_argument("--x", type=float, required=True)
    add.add_argument("--y", type=float, required=True)
    add.add_argument("--z", type=float, required=True)
    add.add_argument("--roll", type=float, default=0.0)
    add.add_argument("--pitch", type=float, default=0.0)
    add.add_argument("--yaw", type=float, default=0.0)

    rem = sub.add_parser("waypoint-remove", help="remove waypoint")
    rem.add_argument("--id", required=True)
    upd = sub.add_parser("waypoint-update", help="update waypoint pose")
    upd.add_argument("--id", required=True)
    upd.add_argument("--label", default="")
    upd.add_argument("--x", type=float, required=True)
    upd.add_argument("--y", type=float, required=True)
    upd.add_argument("--z", type=float, required=True)
    upd.add_argument("--roll", type=float, default=0.0)
    upd.add_argument("--pitch", type=float, default=0.0)
    upd.add_argument("--yaw", type=float, default=0.0)

    sub.add_parser("waypoint-clear", help="clear all waypoints")
    sub.add_parser("waypoint-list", help="list waypoints")
    sub.add_parser("state", help="get state")
    sub.add_parser("stop", help="emergency stop running motion")

    run = sub.add_parser("run", help="run path from waypoint A to B")
    run.add_argument("--from-id", required=True)
    run.add_argument("--to-id", required=True)
    run.add_argument("--duration-s", type=float, default=2.0)
    run.add_argument("--profile", choices=["linear", "min_jerk", "geodesic"], default="min_jerk")
    runseq = sub.add_parser("run-seq", help="run sequence by ids")
    runseq.add_argument("--ids", required=True, help="comma-separated waypoint ids, e.g. P1,P2,P3")
    runseq.add_argument("--duration-s", type=float, default=2.0)
    runseq.add_argument("--profile", choices=["linear", "min_jerk", "geodesic"], default="min_jerk")
    return p


async def main_async() -> int:
    args = build_parser().parse_args()
    if args.cmd == "waypoint-add":
        resp = await ws_request(
            args.url,
            "waypoint_add",
            {
                "id": args.id,
                "label": args.label,
                "pose": {
                    "x": args.x,
                    "y": args.y,
                    "z": args.z,
                    "roll": args.roll,
                    "pitch": args.pitch,
                    "yaw": args.yaw,
                },
            },
        )
    elif args.cmd == "waypoint-remove":
        resp = await ws_request(args.url, "waypoint_remove", {"id": args.id})
    elif args.cmd == "waypoint-update":
        resp = await ws_request(
            args.url,
            "waypoint_update",
            {
                "id": args.id,
                "label": args.label,
                "pose": {
                    "x": args.x,
                    "y": args.y,
                    "z": args.z,
                    "roll": args.roll,
                    "pitch": args.pitch,
                    "yaw": args.yaw,
                },
            },
        )
    elif args.cmd == "waypoint-clear":
        resp = await ws_request(args.url, "waypoint_clear", {})
    elif args.cmd == "waypoint-list":
        resp = await ws_request(args.url, "waypoint_list", {})
    elif args.cmd == "run":
        resp = await ws_request(
            args.url,
            "sim_run_waypoints",
            {"from_id": args.from_id, "to_id": args.to_id, "duration_s": args.duration_s, "profile": args.profile},
        )
    elif args.cmd == "run-seq":
        ids = [x.strip() for x in str(args.ids).split(",") if x.strip()]
        resp = await ws_request(
            args.url,
            "sim_run_sequence",
            {"ids": ids, "duration_s": args.duration_s, "profile": args.profile},
        )
    elif args.cmd == "stop":
        resp = await ws_request(args.url, "sim_stop", {})
    else:
        resp = await ws_request(args.url, "state", {})
    print(json.dumps(resp, ensure_ascii=False, indent=2))
    return 0 if bool(resp.get("ok")) else 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
