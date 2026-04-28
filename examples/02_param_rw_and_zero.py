#!/usr/bin/env python3
from __future__ import annotations

import argparse

from motorbridge_arm_sdk.arm import Arm
from motorbridge_arm_sdk.model.profiles import rebot_arm_robstride


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--joint-index", type=int, default=0)
    ap.add_argument("--param-id", default="0x200C")
    ap.add_argument("--param-type", default="u32")
    ap.add_argument("--write", type=int, default=0)
    ap.add_argument("--zero", type=int, default=0)
    args = ap.parse_args()

    param_id = int(args.param_id, 0)
    arm = Arm(rebot_arm_robstride(channel=args.channel))
    arm.connect()
    arm.enable()
    if args.write:
        arm.write_param(args.joint_index, param_id, args.write, args.param_type)
        print(f"wrote param 0x{param_id:X}={args.write}")
    value = arm.read_param(args.joint_index, param_id, args.param_type)
    print(f"read param 0x{param_id:X}={value}")
    if args.zero:
        arm.zero_calibrate(scope="joint", joint_index=args.joint_index)
        print("zero_calibrate joint done")
    arm.disable()
    arm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
