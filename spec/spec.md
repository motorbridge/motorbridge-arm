# motorbridge-arm Product Spec

## Product Position

`motorbridge-arm` is the product-form layer for serial robotic arms.
It consumes `motorbridge` as the motor/bus substrate and provides turnkey arm-level APIs.

## Product Model Assets

- URDF path convention: `models/urdf/<product_model>.urdf`
- Mesh assets convention: `models/meshes/*`
- Profile binding must map:
  - `ArmConfig.urdf_path`
  - `ArmConfig.ee_frame`
  - motor topology (`esc_id`, `feedback_id`, `direction`, limits)

This ensures hardware control and model-based FK/IK share the same product definition.

## Minimum API

- `Arm.connect()`
- `Arm.close()`
- `Arm.enable()`
- `Arm.disable()`
- `Arm.estop()`
- `Arm.refresh_state()`
- `Arm.get_state()`
- `Arm.get_pose()`
- `Arm.move_j(q_target, vlim)`
- `Arm.home(vlim)`
- `Arm.read_param(joint_index, param_id, param_type=None)`
- `Arm.write_param(joint_index, param_id, value, param_type=None)`
- `Arm.zero_calibrate(scope="all"|"joint", joint_index=None)`

## State Machine

- `DISCONNECTED`
- `IDLE`
- `ENABLED`
- `RUNNING`
- `FAULT`

## Safety Rules

- Joint target is always clamped to configured position limits.
- Velocity limit is capped by configured per-joint max velocity.
- `estop()` transitions to `FAULT` and disables all motors.

## Simulation Mode

- `rebot_sdk.sim.SimArm` provides model-only operation.
- Simulation mode uses `Kinematics` and bypasses `motorbridge` bus commands.
- Intended for:
  - FK/IK validation
  - trajectory dry run
  - CI checks without hardware
