#!/usr/bin/env python3
"""Demonstrate tool/payload configuration and gripper control.

This example uses the Arm class with a mock config to show the API.
In production, connect to real hardware and use a config with a gripper joint.
"""
from __future__ import annotations

from motorbridge_arm_sdk.model.profiles import rebot_arm_robstride
from motorbridge_arm_sdk.types import ToolConfig, PayloadConfig


def main() -> None:
    config = rebot_arm_robstride()

    # --- Tool offset configuration ---
    # Simulate a 5cm tool offset along Z and a small rotation.
    tool = ToolConfig(x=0.0, y=0.0, z=0.05, roll=0.0, pitch=0.0, yaw=0.0)
    print(f"Tool offset: dx={tool.x}, dy={tool.y}, dz={tool.z}m")

    # --- Payload configuration ---
    # Simulate a 0.5 kg payload at the end-effector.
    payload = PayloadConfig(mass=0.5, cx=0.0, cy=0.0, cz=0.025)
    print(f"Payload: mass={payload.mass}kg, CoM=({payload.cx}, {payload.cy}, {payload.cz})")

    # --- Gripper configuration ---
    # If the arm config includes a gripper_joint, the Arm class provides
    # gripper_open() and gripper_close() methods:
    #
    #   with Arm(config) as arm:
    #       arm.connect()
    #       arm.enable()
    #       arm.set_tool(tool)
    #       arm.set_payload(payload)
    #       arm.gripper_open(pos=1.0)    # fully open
    #       arm.gripper_close(pos=0.0)   # fully closed
    #
    # The gripper is identified by either:
    #   1. config.gripper_joint (explicit name), or
    #   2. Any joint with "gripper" in its name (auto-detect)

    print("\nGripper/tool API demonstrated. Connect to hardware for live control.")


if __name__ == "__main__":
    main()
