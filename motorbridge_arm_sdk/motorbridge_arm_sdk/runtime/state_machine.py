from __future__ import annotations

import logging
import threading

from ..errors import ArmError, ArmErrorCode
from ..types import ArmRunState

logger = logging.getLogger(__name__)


class RuntimeStateMachine:
    _ALLOWED: dict[ArmRunState, set[ArmRunState]] = {
        ArmRunState.DISCONNECTED: {ArmRunState.IDLE},
        ArmRunState.IDLE: {ArmRunState.DISCONNECTED, ArmRunState.ENABLED, ArmRunState.FAULT},
        ArmRunState.ENABLED: {ArmRunState.IDLE, ArmRunState.RUNNING, ArmRunState.FAULT},
        ArmRunState.RUNNING: {ArmRunState.ENABLED, ArmRunState.FAULT},
        ArmRunState.FAULT: {ArmRunState.IDLE, ArmRunState.DISCONNECTED},
    }

    def __init__(self, init_state: ArmRunState = ArmRunState.DISCONNECTED) -> None:
        self._state = init_state
        self._lock = threading.Lock()

    @property
    def state(self) -> ArmRunState:
        with self._lock:
            return self._state

    def transition(self, next_state: ArmRunState) -> ArmRunState:
        with self._lock:
            allowed = self._ALLOWED.get(self._state, set())
            if next_state == self._state:
                return self._state
            if next_state not in allowed:
                raise ArmError(
                    ArmErrorCode.ERR_STATE,
                    f"invalid state transition: {self._state.value} -> {next_state.value}",
                )
            logger.debug("state transition: %s -> %s", self._state.value, next_state.value)
            self._state = next_state
            return self._state

    def force(self, state: ArmRunState) -> ArmRunState:
        with self._lock:
            logger.warning("state force: %s -> %s", self._state.value, state.value)
            self._state = state
            return self._state
