from motorbridge_arm_sdk.motion.planner import estimate_steps


def test_estimate_steps_min_two():
    assert estimate_steps([0.0], [0.0]) == 2


def test_estimate_steps_growth():
    n = estimate_steps([0.0, 0.0], [0.2, 0.0], step_rad=0.02)
    assert n >= 11
