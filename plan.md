# motorbridge-arm Rewrite Plan

Last Updated: 2026-04-28
Owner: motorbridge-arm team

## 1) Goal

Build a complete, production-grade arm SDK on top of current `motorbridge-arm` architecture.

Core constraints:

- Motor control substrate must be `motorbridge/bindings/python` only.
- Arm-level API stays stable (`Arm` as primary facade).
- Architecture must support multi-vendor motors and future multi-arm products.

---

## 2) Target Architecture

- `Arm` facade: user-facing product API.
- `runtime`: state machine, lifecycle guard, fault transitions.
- `session`: bus/controller lifecycle + joint handles.
- `vendors`: vendor adapter registry (plugin-like expansion).
- `safety`: limits, clamp, pre-dispatch checks.
- `motion`: planner/executor abstraction.
- `model`: URDF profile + FK/IK backend.
- `telemetry`: state cache, recorder, trace export.
- `calibration`: zeroing and calibration flows.

---

## 3) Rewrite Scope and Feature Parity

Required parity:

- [x] Connect/enable/disable/estop lifecycle
- [x] `move_j` + home
- [x] typed param read/write bridge
- [x] zero calibration
- [x] FK runtime pose query
- [x] IK runtime solve + arm end pose controller
- [x] Cartesian trajectory (`move_l`)
- [x] Circular trajectory (`move_c`)
- [x] Trajectory strategy options (linear/min-jerk/geodesic)
- [ ] Unified fault recovery workflow
- [ ] Hardware-in-loop integration tests

Nice-to-have parity:

- [ ] gravity compensation controller hooks
- [ ] richer multi-language docs/examples

---

## 4) Phase Plan

## Phase A - Foundation (Done/In Progress)

- [x] Introduce vendor adapter registry (remove hardcoded vendor branches in session add_motor path).
- [x] Add runtime state machine and integrate base transitions.
- [x] Set default runtime state to `DISCONNECTED`.
- [x] Add tests for adapter registry and state machine transitions.
- [x] Normalize session-level timeout/retry/error mapping for all vendor operations.

## Phase B - Controller Pipeline

- [x] Add controller abstraction for high-level motion commands.
- [x] Implement `move_j` pipeline as planner -> safety gate -> executor.
- [x] Introduce `move_l` pipeline (Cartesian interpolation to joint commands).
- [x] Introduce `move_c` pipeline (arc constraints + discretization).
- [ ] Add command cancellation and stop hooks.

## Phase C - Kinematics and Trajectory Upgrade

- [x] Replace current minimal IK fallback with robust Pinocchio-based solver path.
- [x] Add trajectory profiles: linear, min-jerk, geodesic.
- [x] Add CLIK tracker option with joint-limit-aware null-space term.
- [x] Add trajectory stats (success rate, max/avg tracking error).
- [x] Add unified trajectory pipeline modules:
  `trajectory/sampler.py`, `trajectory/clik_tracker.py`, `trajectory/trajectory_planner.py`.

## Phase E - Advanced Packages

- [x] Add independent `ArmEndPos` high-level controller.
- [x] Upgrade dynamics subpackage to engineering-grade implementation:
  robot_model/inertia/inverse_dynamics/forward_dynamics/energy/centroidal/derivatives
  with shape checks, all-terms APIs, centroidal APIs, derivative APIs, and compatibility aliases.

## Phase D - Productization and Multi-Arm Readiness

- [ ] Multi-profile model registry (`ArmConfig` loaders for multiple arm products).
- [ ] Vendor capability matrix in code (mode/param support declarative checks).
- [ ] Replay/trace tooling improvements.
- [ ] Cross-platform CI matrix and packaging verification.
- [x] 3D simulation visualization baseline: MeshCat real-time rendering support (`rebot_sdk.sim.MeshCatArmVisualizer`).
- [x] Simulation upgrade: FK/IK/trajectory planning+playback in `SimArm` and `examples/sim/*`.

---

## 5) Coding Rules for This Rewrite

- Never bypass `motorbridge` Python binding for motor/bus control.
- No direct vendor logic in `Arm` facade.
- No silent exception swallowing in critical control path.
- Every new feature must include tests and doc updates.
- API changes must be reflected in `spec/api.md`.

---

## 6) Immediate Next 3 Tasks

1. Implement unified timeout/retry wrapper in `session` for mode switch and param IO.
2. Add controller abstraction and first `move_l` implementation.
3. Add IK + trajectory tests (model-only CI first, hardware optional later).
