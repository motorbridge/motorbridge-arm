#!/usr/bin/env python3
"""Demonstrate MoveL and MoveC Cartesian motions using SimArm (no hardware needed)."""
from __future__ import annotations

from motorbridge_arm_sdk.model.profiles import rebot_arm_robstride
from motorbridge_arm_sdk.sim.simulator import SimArm
from motorbridge_arm_sdk.types import Pose6D


def main() -> None:
    config = rebot_arm_robstride()
    sim = SimArm(config)

    # Start from home position.
    home = config.default_home or [0.0] * 6
    sim.set_joint_positions(home)
    start_pose = sim.get_pose()
    print(f"Start pose: x={start_pose.x:.3f} y={start_pose.y:.3f} z={start_pose.z:.3f}")

    # MoveL: linear Cartesian motion to a target.
    target_l = Pose6D(x=0.25, y=0.1, z=0.15, roll=0.0, pitch=0.0, yaw=0.0)
    traj_l = sim.move_l(target_l, duration_s=2.0, profile="min_jerk")
    end_l = sim.get_pose()
    print(f"\nMoveL trajectory: {len(traj_l.points)} points, duration={traj_l.duration_s:.2f}s")
    print(f"MoveL end pose:   x={end_l.x:.3f} y={end_l.y:.3f} z={end_l.z:.3f}")
    if traj_l.points:
        success_count = sum(1 for p in traj_l.points if p.ik_success)
        print(f"IK success rate:  {success_count}/{len(traj_l.points)}")

    # MoveC: circular arc motion.
    sim.set_joint_positions(home)  # Reset.
    target_c = Pose6D(x=0.2, y=-0.1, z=0.15, roll=0.0, pitch=0.0, yaw=0.0)
    traj_c = sim.move_c(
        target_c,
        center_x=0.2,
        center_y=0.0,
        normal_z=1.0,
        duration_s=2.0,
        steps=60,
    )
    end_c = sim.get_pose()
    print(f"\nMoveC trajectory: {len(traj_c.points)} points, duration={traj_c.duration_s:.2f}s")
    print(f"MoveC end pose:   x={end_c.x:.3f} y={end_c.y:.3f} z={end_c.z:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
