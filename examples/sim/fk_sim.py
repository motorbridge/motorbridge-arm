#!/usr/bin/env python3
from __future__ import annotations

import math

from rebot_sdk.model.profiles import rebot_arm_robstride
from rebot_sdk.sim import MeshCatArmVisualizer, SimArm


def main() -> int:
    cfg = rebot_arm_robstride()
    sim = SimArm(cfg)
    viz = MeshCatArmVisualizer(cfg.urdf_path, open_browser=True)

    print("Input 6 joint angles in degrees, or q to quit")
    while True:
        line = input("q(deg)> ").strip().lower()
        if line in {"q", "quit", "exit", ""}:
            break
        vals = [float(x) for x in line.split()]
        if len(vals) != len(cfg.joints):
            print(f"need {len(cfg.joints)} values")
            continue
        q = [math.radians(v) for v in vals]
        sim.set_joint_positions(q)
        viz.update(q)
        print(sim.get_pose())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
