from __future__ import annotations

from .calibration.zeroing import ZeroCalibrator
from .motion.executor import JointMotionExecutor
from .motion.planner import estimate_steps
from .params.registry import ParamRegistry, create_default_registry
from .safety.supervisor import SafetySupervisor
from .session import ModeLike, MotorBridgeSession
from .telemetry.recorder import Recorder
from .telemetry.state_cache import StateCache
from .types import ArmConfig, ArmRunState, ArmState, JointState, Pose6D
from .model.kinematics import Kinematics


class Arm:
    def __init__(self, config: ArmConfig, registry: ParamRegistry | None = None) -> None:
        self._cfg = config
        self._session = MotorBridgeSession(config.channel)
        self._safety = SafetySupervisor(config)
        self._cache = StateCache(config)
        self._executor = JointMotionExecutor(config.loop_dt_s)
        self._zero = ZeroCalibrator(self._session)
        self._kin = Kinematics(config.urdf_path, config.ee_frame)
        self._registry = registry or create_default_registry()
        self._recorder = Recorder()

    def connect(self) -> None:
        self._session.connect()
        for j in self._cfg.joints:
            self._session.add_joint(j)
        self._cache.update_run_state(ArmRunState.IDLE)
        self._recorder.add("connect", {"channel": self._cfg.channel, "joints": len(self._cfg.joints)})

    def close(self) -> None:
        self._session.close()
        self._cache.update_run_state(ArmRunState.DISCONNECTED)

    def enable(self) -> None:
        self._session.enable_all()
        self._cache.update_run_state(ArmRunState.ENABLED)

    def disable(self) -> None:
        self._session.disable_all()
        self._cache.update_run_state(ArmRunState.IDLE)

    def estop(self) -> None:
        self._session.disable_all()
        self._cache.update_run_state(ArmRunState.FAULT)

    def refresh_state(self) -> ArmState:
        self._session.request_feedback_all()
        for i, h in enumerate(self._session.joints):
            raw = h.motor.get_state()
            if raw is None:
                continue
            q = (raw.pos - h.config.zero_offset) / h.config.direction
            self._cache.update_joint(
                i,
                JointState(
                    name=h.config.name,
                    pos=q,
                    vel=raw.vel,
                    torq=raw.torq,
                    status_code=raw.status_code,
                    t_mos=raw.t_mos,
                    t_rotor=raw.t_rotor,
                ),
            )
        return self._cache.snapshot()

    def get_state(self) -> ArmState:
        return self._cache.snapshot()

    def get_joint_positions(self) -> list[float]:
        st = self.refresh_state()
        out: list[float] = []
        for j in st.joints:
            out.append(0.0 if j.pos is None else float(j.pos))
        return out

    def get_pose(self) -> Pose6D:
        q = self.get_joint_positions()
        return self._kin.forward(q)

    def move_j(self, q_target: list[float], vlim: float = 1.0) -> None:
        q_target = self._safety.clamp_joint_targets(q_target)
        vlim = self._safety.validate_velocity_limit(vlim)
        q_now = self.get_joint_positions()
        steps = estimate_steps(q_now, q_target)
        points = self._executor.interpolate_linear(q_now, q_target, steps)
        self._session.ensure_mode_all(ModeLike.POS_VEL)
        self._cache.update_run_state(ArmRunState.RUNNING)
        self._executor.run(points, self._session.set_pos_vel_all, vlim)
        self._cache.update_run_state(ArmRunState.ENABLED)
        self._recorder.add("move_j", {"steps": steps, "vlim": vlim})

    def home(self, vlim: float = 1.0) -> None:
        home = self._cfg.default_home or [0.0 for _ in self._cfg.joints]
        self.move_j(home, vlim=vlim)

    def read_param(self, joint_index: int, param_id: int, param_type: str | None = None, timeout_ms: int = 1000):
        spec = self._registry.get(self._cfg.joints[joint_index].vendor, param_id)
        ptype = param_type or (spec.param_type if spec else "f32")
        value = self._session.get_param(joint_index, param_id, ptype, timeout_ms)
        self._recorder.add("read_param", {"joint": joint_index, "param_id": param_id, "type": ptype, "value": value})
        return value

    def write_param(self, joint_index: int, param_id: int, value: int | float, param_type: str | None = None) -> None:
        spec = self._registry.get(self._cfg.joints[joint_index].vendor, param_id)
        ptype = param_type or (spec.param_type if spec else "f32")
        self._session.set_param(joint_index, param_id, ptype, value)
        self._recorder.add("write_param", {"joint": joint_index, "param_id": param_id, "type": ptype, "value": value})

    def zero_calibrate(self, scope: str = "all", joint_index: int | None = None):
        if scope == "all":
            result = self._zero.zero_all()
        else:
            if joint_index is None:
                raise ValueError("joint_index is required for scope=joint")
            result = self._zero.zero_joint(joint_index)
        self._recorder.add("zero_calibrate", {"scope": scope, "joint_index": joint_index, "ok": result.ok})
        return result

    def save_trace(self, path: str) -> None:
        self._recorder.save_json(path)
