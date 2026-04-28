from __future__ import annotations

from importlib import import_module
import sys

from motorbridge_arm_sdk import *  # noqa: F401,F403


_SUBMODULES = [
    "arm",
    "errors",
    "types",
    "session",
    "runtime",
    "controllers",
    "dynamics",
    "model",
    "motion",
    "params",
    "safety",
    "sim",
    "telemetry",
    "trajectory",
    "vendors",
    "web",
]

for _name in _SUBMODULES:
    sys.modules[f"{__name__}.{_name}"] = import_module(f"motorbridge_arm_sdk.{_name}")
