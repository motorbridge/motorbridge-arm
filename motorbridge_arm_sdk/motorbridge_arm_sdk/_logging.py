"""Centralised logger factory for motorbridge_arm_sdk.

Library best practice: attach a NullHandler so that if the user does not
configure logging, nothing is emitted.  Users control output by calling
``logging.basicConfig(...)`` or configuring their own handlers.
"""
from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``motorbridge_arm_sdk`` namespace."""
    logger = logging.getLogger(f"motorbridge_arm_sdk.{name}")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger
