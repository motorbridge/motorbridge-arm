# API Contract (Initial)

## ArmConfig

- `name: str`
- `model: str`
- `channel: str`
- `joints: list[JointConfig]`
- `default_home: list[float]`
- `loop_dt_s: float`
- `motion_profile: str` (`linear|min_jerk|geodesic`)

## JointConfig

- `name`
- `vendor`
- `model`
- `esc_id`
- `feedback_id`
- `direction`
- `zero_offset`
- `limit_pos_min`
- `limit_pos_max`
- `limit_vel`
- `limit_tau`

## Errors

Errors are raised via `ArmError(code, message)`.
See `error_codes.md`.

## Motion API Extensions

- `Arm.solve_ik(target_pose: Pose6D) -> list[float]`
- `Arm.move_j(q_target: list[float], vlim: float = 1.0, profile: str | None = None) -> None`
- `Arm.move_l(target_pose: Pose6D, vlim: float = 1.0, step_m: float = 0.01, profile: str | None = None) -> None`
- `Arm.move_c(target_pose: Pose6D, center_x: float, center_y: float, normal_z: float = 1.0, vlim: float = 1.0, steps: int = 80, profile: str | None = None) -> None`
