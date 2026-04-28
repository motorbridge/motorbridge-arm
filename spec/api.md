# API Contract (Initial)

## ArmConfig

- `name: str`
- `model: str`
- `channel: str`
- `joints: list[JointConfig]`
- `default_home: list[float]`
- `loop_dt_s: float`

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
