"""Shared fallback utilities for the dynamics subpackage.

Provides a decorator that eliminates the repetitive pinocchio/numpy fallback
boilerplate previously duplicated across every dynamics module.
"""
from __future__ import annotations

from functools import wraps
from typing import Callable

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


def _zeros_1d(n: int):
    """Return a zero vector of length *n* (list or ndarray)."""
    if np is not None:
        return np.zeros(n, dtype=float)
    return [0.0] * n


def _zeros_2d(n: int):
    """Return an n x n zero matrix (list or ndarray)."""
    if np is not None:
        return np.zeros((n, n), dtype=float)
    return [[0.0 for _ in range(n)] for _ in range(n)]


def _eye_2d(n: int):
    """Return an n x n identity matrix (list or ndarray)."""
    if np is not None:
        return np.eye(n, dtype=float)
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def pinocchio_fallback(fallback_fn: Callable):
    """Decorator for dynamics functions that require Pinocchio.

    When Pinocchio is not available (``drm.has_pinocchio`` is False), the
    decorated function is **not** called and *fallback_fn* is invoked instead
    with the same arguments.  The fallback is responsible for returning an
    appropriate zero-filled value.

    Usage::

        @pinocchio_fallback(lambda drm, q: _zeros_1d(len(_as_q(drm, q))))
        def compute_something(drm, q=None):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(drm, *args, **kwargs):
            if not drm.has_pinocchio:
                return fallback_fn(drm, *args, **kwargs)
            return func(drm, *args, **kwargs)

        return wrapper

    return decorator
