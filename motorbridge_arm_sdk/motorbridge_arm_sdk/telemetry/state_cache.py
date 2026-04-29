from __future__ import annotations

import threading
import time

from ..types import ArmConfig, ArmRunState, ArmState, JointState


class StateCache:
    def __init__(self, config: ArmConfig) -> None:
        self._cfg = config
        self._lock = threading.Lock()
        self._state = ArmState(
            run_state=ArmRunState.DISCONNECTED,
            joints=[JointState(name=j.name, pos=None, vel=None, torq=None) for j in config.joints],
            updated_at_s=time.time(),
        )

    def update_run_state(self, run_state: ArmRunState) -> None:
        with self._lock:
            self._state.run_state = run_state
            self._state.updated_at_s = time.time()

    def update_joint(self, index: int, joint: JointState) -> None:
        with self._lock:
            self._state.joints[index] = joint
            self._state.updated_at_s = time.time()

    def snapshot(self) -> ArmState:
        with self._lock:
            return ArmState(
                run_state=self._state.run_state,
                joints=list(self._state.joints),
                updated_at_s=self._state.updated_at_s,
            )
