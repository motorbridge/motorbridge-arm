#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify motorbridge-arm runtime environment.")
    parser.add_argument("--require-pinocchio", action="store_true", help="Fail if pinocchio is unavailable.")
    args = parser.parse_args()

    ok = True

    try:
        import motorbridge  # noqa: F401
        print("[OK] motorbridge import")
    except Exception as exc:
        ok = False
        print(f"[ERR] motorbridge import failed: {exc}")

    has_pin = False
    try:
        import pinocchio as pin  # noqa: F401
        has_pin = True
        print("[OK] pinocchio import")
    except Exception as exc:
        print(f"[WARN] pinocchio import failed: {exc}")

    urdf = Path("models/urdf/reBot-DevArm_fixend_description/urdf/reBot-DevArm_fixend.urdf")
    if urdf.exists():
        print(f"[OK] urdf path exists: {urdf}")
    else:
        ok = False
        print(f"[ERR] urdf path missing: {urdf}")

    if args.require_pinocchio and not has_pin:
        ok = False
        print("[ERR] pinocchio is required but unavailable")

    if ok:
        print("[PASS] environment verification succeeded")
        return 0
    print("[FAIL] environment verification failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
