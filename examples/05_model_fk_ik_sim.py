#!/usr/bin/env python3
from __future__ import annotations

from motorbridge_arm_sdk.model.profiles import rebot_arm_robstride
from motorbridge_arm_sdk.sim import SimArm
from motorbridge_arm_sdk.types import Pose6D


def main() -> int:
    cfg = rebot_arm_robstride()
    sim = SimArm(cfg)
    print("model:", cfg.model)
    print("urdf:", cfg.urdf_path)

    q0 = sim.get_joint_positions()
    p0 = sim.get_pose()
    print("q0:", [round(v, 3) for v in q0])
    print("p0:", p0)

    target_pose = Pose6D(x=p0.x + 0.03, y=p0.y, z=p0.z + 0.02, roll=0.0, pitch=0.0, yaw=0.2)
    q1 = sim.solve_ik(target_pose)
    sim.move_j(q1)
    p1 = sim.get_pose()

    print("q1:", [round(v, 3) for v in q1])
    print("p1:", p1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
