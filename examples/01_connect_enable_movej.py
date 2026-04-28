#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rebot_sdk.arm import Arm
from rebot_sdk.model.profiles import rebot_arm_robstride


def parse_q(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--target", default="0,-0.2,0.8,0,0.5,0")
    ap.add_argument("--vlim", type=float, default=1.0)
    args = ap.parse_args()

    arm = Arm(rebot_arm_robstride(channel=args.channel))
    arm.connect()
    arm.enable()
    arm.move_j(parse_q(args.target), vlim=args.vlim)
    print("move_j done")
    arm.home(vlim=args.vlim)
    print("home done")
    arm.disable()
    arm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
