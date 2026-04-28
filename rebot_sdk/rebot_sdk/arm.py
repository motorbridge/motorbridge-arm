from __future__ import annotations

from dataclasses import asdict
import threading
import time

from .calibration.zeroing import ZeroCalibrator
from .motion.executor import JointMotionExecutor
from .motion.planner import ArcSpec, estimate_steps, interpolate_joint_linear, interpolate_pose_circular, interpolate_pose_geodesic, interpolate_pose_linear
from .params.registry import ParamRegistry, create_default_registry
from .runtime import RuntimeStateMachine
from .safety.supervisor import SafetySupervisor
from .session import ModeLike, MotorBridgeSession
from .telemetry.recorder import Recorder
from .telemetry.state_cache import StateCache
from .types import ArmConfig, ArmRunState, ArmState, FaultState, JointState, PayloadConfig, Pose6D, ToolConfig
from .model.kinematics import Kinematics
from .vendors import MotorAdapterRegistry, create_default_adapter_registry


class Arm:
    def __init__(
        self,
        config: ArmConfig,
        registry: ParamRegistry | None = None,
        adapter_registry: MotorAdapterRegistry | None = None,
    ) -> None:
        self._cfg = config
        self._runtime = RuntimeStateMachine(ArmRunState.DISCONNECTED)
        self._adapter_registry = adapter_registry or create_default_adapter_registry()
        self._session = MotorBridgeSession(config.channel, adapter_registry=self._adapter_registry)
        self._safety = SafetySupervisor(config)
        self._cache = StateCache(config)
        self._executor = JointMotionExecutor(config.loop_dt_s)
        self._zero = ZeroCalibrator(self._session)
        self._kin = Kinematics(config.urdf_path, config.ee_frame)
        self._registry = registry or create_default_registry()
        self._recorder = Recorder()
        self._tool = ToolConfig()
        self._payload = PayloadConfig()
        self._mode = "pos_vel"
        self._ctrl_thread: threading.Thread | None = None
        self._ctrl_running = False
        self._ctrl_fn = None

    def connect(self) -> None:
        self._runtime.transition(ArmRunState.IDLE)
        self._session.connect()
        for j in self._cfg.joints:
            self._session.add_joint(j)
        self._cache.update_run_state(ArmRunState.IDLE)
        self._recorder.add("connect", {"channel": self._cfg.channel, "joints": len(self._cfg.joints)})

    @property
    def num_joints(self) -> int:
        return len(self._cfg.joints)

    @property
    def joint_names(self) -> list[str]:
        return [j.name for j in self._cfg.joints]

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def control_loop_active(self) -> bool:
        t = self._ctrl_thread
        return t is not None and t.is_alive()

    def close(self) -> None:
        self.stop_control_loop()
        self._runtime.force(ArmRunState.DISCONNECTED)
        self._session.close()
        self._cache.update_run_state(ArmRunState.DISCONNECTED)

    def reconnect(self) -> None:
        self.close()
        self.connect()

    def enable(self) -> None:
        self._runtime.transition(ArmRunState.ENABLED)
        self._session.enable_all()
        self._cache.update_run_state(ArmRunState.ENABLED)

    def disable(self) -> None:
        if self._runtime.state == ArmRunState.RUNNING:
            self._runtime.transition(ArmRunState.ENABLED)
        self._runtime.transition(ArmRunState.IDLE)
        self._session.disable_all()
        self._cache.update_run_state(ArmRunState.IDLE)

    def estop(self) -> None:
        self._session.disable_all()
        self._runtime.force(ArmRunState.FAULT)
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

    def get_positions(self, request: bool = True):
        _ = request
        return self.get_joint_positions()

    def get_velocities(self, request: bool = True):
        _ = request
        st = self.refresh_state()
        return [0.0 if j.vel is None else float(j.vel) for j in st.joints]

    def get_torques(self, request: bool = True):
        _ = request
        st = self.refresh_state()
        return [0.0 if j.torq is None else float(j.torq) for j in st.joints]

    def get_state_vectors(self):
        return self.get_positions(), self.get_velocities(), self.get_torques()

    def get_joint_state(self, joint: int | str) -> JointState:
        st = self.refresh_state()
        idx = self._resolve_joint_index(joint)
        return st.joints[idx]

    def get_pose(self) -> Pose6D:
        q = self.get_joint_positions()
        base = self._kin.forward(q)
        return Pose6D(
            x=base.x + self._tool.x,
            y=base.y + self._tool.y,
            z=base.z + self._tool.z,
            roll=base.roll + self._tool.roll,
            pitch=base.pitch + self._tool.pitch,
            yaw=base.yaw + self._tool.yaw,
        )

    def move_j(self, q_target: list[float], vlim: float = 1.0, profile: str | None = None) -> None:
        q_target = self._safety.clamp_joint_targets(q_target)
        self._run_joint_target(q_target, vlim=vlim, profile=profile or self._cfg.motion_profile)

    def mode_mit(self) -> bool:
        self._mode = "mit"
        self._session.ensure_mode_all(ModeLike.MIT)
        return True

    def mode_pos_vel(self) -> bool:
        self._mode = "pos_vel"
        self._session.ensure_mode_all(ModeLike.POS_VEL)
        return True

    def mode_vel(self) -> bool:
        self._mode = "vel"
        self._session.ensure_mode_all(ModeLike.VEL)
        return True

    def mit(self, pos, vel=None, kp=None, kd=None, tau=None) -> None:
        n = len(self._cfg.joints)
        vel = vel if vel is not None else [0.0] * n
        kp = kp if kp is not None else [0.0] * n
        kd = kd if kd is not None else [0.0] * n
        tau = tau if tau is not None else [0.0] * n
        self.mode_mit()
        self._session.set_mit_all(list(pos), list(vel), list(kp), list(kd), list(tau))

    def pos_vel(self, pos, vlim=None) -> None:
        n = len(self._cfg.joints)
        if vlim is None:
            vmax = min(j.limit_vel for j in self._cfg.joints)
        else:
            if isinstance(vlim, (list, tuple)):
                vmax = float(vlim[0]) if len(vlim) > 0 else 1.0
            else:
                vmax = float(vlim)
        self.mode_pos_vel()
        self._session.set_pos_vel_all(list(pos), vmax)

    def set_vel(self, vel) -> None:
        self.mode_vel()
        self._session.set_vel_all(list(vel))

    def move_joint(self, joint: int | str, pos: float, vlim: float = 1.0) -> None:
        idx = self._resolve_joint_index(joint)
        q = self.get_joint_positions()
        q[idx] = pos
        self.move_j(q, vlim=vlim)

    def move_joints(self, targets: dict[int | str, float], vlim: float = 1.0) -> None:
        q = self.get_joint_positions()
        for j, p in targets.items():
            idx = self._resolve_joint_index(j)
            q[idx] = p
        self.move_j(q, vlim=vlim)

    def joint_vel(self, joint: int | str, vel: float) -> None:
        idx = self._resolve_joint_index(joint)
        self._session.ensure_mode_joint(idx, ModeLike.VEL)
        self._session.set_vel_joint(idx, vel)
        self._recorder.add("joint_vel", {"joint": self._cfg.joints[idx].name, "vel": vel})

    def joint_mit(self, joint: int | str, pos: float, vel: float, kp: float, kd: float, tau: float = 0.0) -> None:
        idx = self._resolve_joint_index(joint)
        self._session.ensure_mode_joint(idx, ModeLike.MIT)
        self._session.set_mit_joint(idx, pos=pos, vel=vel, kp=kp, kd=kd, tau=tau)
        self._recorder.add(
            "joint_mit",
            {"joint": self._cfg.joints[idx].name, "pos": pos, "vel": vel, "kp": kp, "kd": kd, "tau": tau},
        )

    def home(self, vlim: float = 1.0) -> None:
        home = self._cfg.default_home or [0.0 for _ in self._cfg.joints]
        self.move_j(home, vlim=vlim)

    def solve_ik(self, target: Pose6D) -> list[float]:
        q_now = self.get_joint_positions()
        q = self._kin.inverse(target, q_now)
        q = self._safety.clamp_joint_targets(q)
        return q

    def move_l(self, target: Pose6D, vlim: float = 1.0, step_m: float = 0.01, profile: str | None = None) -> None:
        start = self.get_pose()
        profile_name = profile or self._cfg.motion_profile
        dist = ((target.x - start.x) ** 2 + (target.y - start.y) ** 2 + (target.z - start.z) ** 2) ** 0.5
        steps = max(2, int(dist / max(step_m, 1e-4)) + 1)
        if profile_name.lower() == "geodesic":
            poses = interpolate_pose_geodesic(start, target, steps, profile=profile_name)
        else:
            poses = interpolate_pose_linear(start, target, steps, profile=profile_name)
        q_now = self.get_joint_positions()
        joint_points: list[list[float]] = []
        for pose in poses:
            q_now = self._kin.inverse(pose, q_now)
            joint_points.append(self._safety.clamp_joint_targets(q_now))
        self._run_joint_points(joint_points, vlim=vlim, motion_name="move_l")

    def move_c(
        self,
        target: Pose6D,
        center_x: float,
        center_y: float,
        normal_z: float = 1.0,
        vlim: float = 1.0,
        steps: int = 80,
        profile: str | None = None,
    ) -> None:
        start = self.get_pose()
        poses = interpolate_pose_circular(
            start,
            target,
            ArcSpec(center_x=center_x, center_y=center_y, normal_z=normal_z),
            steps,
            profile=profile or self._cfg.motion_profile,
        )
        q_now = self.get_joint_positions()
        joint_points: list[list[float]] = []
        for pose in poses:
            q_now = self._kin.inverse(pose, q_now)
            joint_points.append(self._safety.clamp_joint_targets(q_now))
        self._run_joint_points(joint_points, vlim=vlim, motion_name="move_c")

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

    def set_tool(self, tool: ToolConfig) -> None:
        self._tool = tool
        self._recorder.add("set_tool", asdict(tool))

    def set_payload(self, payload: PayloadConfig) -> None:
        self._payload = payload
        self._recorder.add("set_payload", asdict(payload))

    def get_faults(self) -> FaultState:
        st = self.refresh_state()
        bad: list[str] = []
        for j in st.joints:
            if j.status_code is None:
                continue
            if j.status_code != 1:
                bad.append(j.name)
        return FaultState(has_fault=len(bad) > 0, faulted_joints=bad)

    def clear_faults(self) -> None:
        self._session.clear_fault_all()
        self._recorder.add("clear_faults", {"ok": True})

    def gripper_open(self, pos: float = 1.0, vlim: float = 1.0) -> None:
        self._gripper_move(pos=pos, vlim=vlim)

    def gripper_close(self, pos: float = 0.0, vlim: float = 1.0) -> None:
        self._gripper_move(pos=pos, vlim=vlim)

    def save_trace(self, path: str) -> None:
        self._recorder.save_json(path)

    def start_control_loop(self, callback, rate_hz: float | None = None) -> None:
        if self.control_loop_active:
            self.stop_control_loop()
        self._ctrl_fn = callback
        self._ctrl_running = True
        dt = self._cfg.loop_dt_s if rate_hz is None or rate_hz <= 0 else 1.0 / rate_hz

        def _loop():
            while self._ctrl_running:
                t0 = time.perf_counter()
                if self._ctrl_fn is not None:
                    self._ctrl_fn(self, dt)
                dt_sleep = dt - (time.perf_counter() - t0)
                if dt_sleep > 0:
                    time.sleep(dt_sleep)

        self._ctrl_thread = threading.Thread(target=_loop, daemon=True)
        self._ctrl_thread.start()

    def stop_control_loop(self) -> None:
        self._ctrl_running = False
        if self._ctrl_thread is not None:
            if threading.current_thread() is not self._ctrl_thread:
                self._ctrl_thread.join(timeout=1.0)
            self._ctrl_thread = None

    def _run_joint_target(self, q_target: list[float], vlim: float, profile: str = "linear") -> None:
        vlim = self._safety.validate_velocity_limit(vlim)
        q_now = self.get_joint_positions()
        steps = estimate_steps(q_now, q_target)
        points = interpolate_joint_linear(q_now, q_target, steps, profile=profile)
        self._run_joint_points(points, vlim=vlim, motion_name="move_j", steps=steps)

    def _run_joint_points(self, points: list[list[float]], vlim: float, motion_name: str, steps: int | None = None) -> None:
        self._runtime.transition(ArmRunState.RUNNING)
        self._session.ensure_mode_all(ModeLike.POS_VEL)
        self._cache.update_run_state(ArmRunState.RUNNING)
        try:
            self._executor.run(points, self._session.set_pos_vel_all, vlim)
        finally:
            self._runtime.transition(ArmRunState.ENABLED)
            self._cache.update_run_state(ArmRunState.ENABLED)
        payload = {"vlim": vlim, "points": len(points)}
        if steps is not None:
            payload["steps"] = steps
        self._recorder.add(motion_name, payload)

    def _gripper_move(self, pos: float, vlim: float) -> None:
        idx = self._find_gripper_index()
        if idx is None:
            raise ValueError("gripper_joint is not configured")
        q = self.get_joint_positions()
        q[idx] = pos
        self.move_j(q, vlim=vlim)

    def _find_gripper_index(self) -> int | None:
        if self._cfg.gripper_joint:
            for i, j in enumerate(self._cfg.joints):
                if j.name == self._cfg.gripper_joint:
                    return i
        for i, j in enumerate(self._cfg.joints):
            if "gripper" in j.name.lower():
                return i
        return None

    def _resolve_joint_index(self, joint: int | str) -> int:
        if isinstance(joint, int):
            if 0 <= joint < len(self._cfg.joints):
                return joint
            raise IndexError(f"joint index out of range: {joint}")
        for i, jc in enumerate(self._cfg.joints):
            if jc.name == joint:
                return i
        raise KeyError(f"unknown joint name: {joint}")

    def __enter__(self) -> "Arm":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Arm(name={self._cfg.name!r}, joints={len(self._cfg.joints)}, mode={self._mode!r})"
