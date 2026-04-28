# motorbridge-arm

Product-form serial robotic arm SDK built on `motorbridge` Python binding.

## Scope

This repository targets the product form layer:

- Arm-level lifecycle (`connect/enable/disable/estop`)
- Joint-space execution (`move_j/home`)
- Parameter registry and typed read/write bridge
- Zero calibration flow with guard checks
- Safety supervision and state cache

## Quick Start

```bash
git clone https://github.com/motorbridge/motorbridge-arm.git
cd motorbridge-arm
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
python examples/00_session_smoke.py --channel can0
```

If you already cloned into another folder, just `cd <your-path>/motorbridge-arm` first.

## Layout

- `spec/`: frozen product/API spec
- `models/`: arm product model assets (`urdf/`, `meshes/`)
- `rebot_sdk/`: implementation package
- `examples/`: runnable examples
- `tests/`: unit tests

## Model and Pinocchio

- Default product model:
  - `models/urdf/rebot_arm_robstride.urdf`
- Profile binding:
  - `rebot_sdk/model/profiles.py` injects `urdf_path` and `ee_frame` into `ArmConfig`
- Kinematics:
  - `rebot_sdk/model/kinematics.py`
  - Auto-uses Pinocchio when available and URDF exists
  - Falls back to simplified chain model if Pinocchio is missing

Install with kinematics extras:

```bash
pip install -e .[kinematics]
```

Run model-only simulation (no CAN, no motor hardware):

```bash
python examples/05_model_fk_ik_sim.py
```

## Naming Conventions

- Product repository: `motorbridge-arm`
- Python package namespace: `rebot_sdk` (current implementation root)
- Main product-level class: `Arm`

Why `Arm`:
- This repository is the robotic-arm product form layer.
- At this layer, user-facing semantics should be arm-level (`Arm.connect()`, `Arm.move_j()`), not single-motor-level APIs.
