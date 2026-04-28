#!/usr/bin/env python3
from __future__ import annotations

from motorbridge_arm_sdk.model.profiles import rebot_arm_robstride
from motorbridge_arm_sdk.sim import MeshCatArmVisualizer, SimArm
from motorbridge_arm_sdk.types import Pose6D


def main() -> int:
    cfg = rebot_arm_robstride()
    sim = SimArm(cfg)
    viz = MeshCatArmVisualizer(cfg.urdf_path, open_browser=True)

    print("Input target pose: x y z [roll pitch yaw], or q to quit")
    while True:
        line = input("traj> ").strip().lower()
        if line in {"q", "quit", "exit", ""}:
            break
        vals = [float(x) for x in line.split()]
        if len(vals) not in {3, 6}:
            print("format: x y z [roll pitch yaw]")
            continue
        target = Pose6D(
            x=vals[0],
            y=vals[1],
            z=vals[2],
            roll=vals[3] if len(vals) == 6 else 0.0,
            pitch=vals[4] if len(vals) == 6 else 0.0,
            yaw=vals[5] if len(vals) == 6 else 0.0,
        )
        traj = sim.move_l(target, duration_s=2.0, null_gain=0.1)
        viz.clear_paths()
        ref = [[p.pose.x, p.pose.y, p.pose.z] for p in traj.points]
        viz.draw_ref_path(ref)
        sim.playback(traj, realtime=True, visualizer=viz)
        print(f"trajectory points: {len(traj.points)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
