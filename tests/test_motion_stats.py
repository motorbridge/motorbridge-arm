from motorbridge_arm_sdk.motion.stats import compute_trajectory_stats
from motorbridge_arm_sdk.types import Pose6D


def test_compute_trajectory_stats_basic():
    ref = [
        Pose6D(0.0, 0.0, 0.0, 0, 0, 0),
        Pose6D(1.0, 0.0, 0.0, 0, 0, 0),
    ]
    act = [
        Pose6D(0.0, 0.0, 0.0, 0, 0, 0),
        Pose6D(1.1, 0.0, 0.0, 0, 0, 0),
    ]
    s = compute_trajectory_stats(ref, act)
    assert s.total_points == 2
    assert s.success_count == 2
    assert s.success_rate == 1.0
    assert s.max_position_error >= 0.1 - 1e-9
    assert s.avg_position_error >= 0.05 - 1e-9
    assert s.max_orientation_error == 0.0


def test_compute_trajectory_stats_with_success_flags_and_orientation_error():
    ref = [
        Pose6D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        Pose6D(0.0, 0.0, 0.0, 0.1, 0.2, 0.3),
    ]
    act = [
        Pose6D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        Pose6D(0.0, 0.0, 0.0, 0.3, 0.2, 0.3),
    ]
    s = compute_trajectory_stats(ref, act, success_flags=[True, False])
    assert s.total_points == 2
    assert s.success_count == 1
    assert abs(s.success_rate - 0.5) < 1e-12
    assert s.max_orientation_error > 0.0
