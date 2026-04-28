#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rebot_sdk.model.profiles import rebot_arm_robstride
from rebot_sdk.arm import Arm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--enable", type=int, default=0)
    args = ap.parse_args()

    cfg = rebot_arm_robstride(channel=args.channel)
    arm = Arm(cfg)
    arm.connect()
    print(f"connected arm={cfg.name} joints={len(cfg.joints)} channel={cfg.channel}")
    if args.enable:
        arm.enable()
        print("enabled")
        arm.disable()
        print("disabled")
    arm.close()
    print("closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
