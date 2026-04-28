from motorbridge_arm_sdk.motion.planner import ArcSpec, interpolate_joint_linear, interpolate_pose_circular, interpolate_pose_geodesic, interpolate_pose_linear
from motorbridge_arm_sdk.types import Pose6D


def test_linear_pose_interpolation_endpoints():
    a = Pose6D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    b = Pose6D(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
    pts = interpolate_pose_linear(a, b, 5)
    assert len(pts) == 5
    assert pts[0] == a
    assert pts[-1] == b


def test_circular_pose_interpolation_endpoints():
    a = Pose6D(1.0, 0.0, 0.2, 0.0, 0.0, 0.0)
    b = Pose6D(0.0, 1.0, 0.4, 0.0, 0.0, 1.57)
    arc = ArcSpec(center_x=0.0, center_y=0.0, normal_z=1.0)
    pts = interpolate_pose_circular(a, b, arc, 10)
    assert len(pts) == 10
    assert abs(pts[0].x - a.x) < 1e-6
    assert abs(pts[0].y - a.y) < 1e-6
    assert abs(pts[-1].x - b.x) < 1e-6
    assert abs(pts[-1].y - b.y) < 1e-6


def test_min_jerk_profile_joint_interpolation():
    pts = interpolate_joint_linear([0.0], [1.0], 5, profile="min_jerk")
    assert len(pts) == 5
    assert abs(pts[0][0] - 0.0) < 1e-12
    assert abs(pts[-1][0] - 1.0) < 1e-12
    # Monotonic non-decreasing
    vals = [p[0] for p in pts]
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


def test_geodesic_profile_pose_endpoints():
    a = Pose6D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    b = Pose6D(0.2, -0.1, 0.5, 0.1, -0.2, 0.3)
    pts = interpolate_pose_linear(a, b, 7, profile="geodesic")
    assert pts[0] == a
    assert pts[-1] == b


def test_geodesic_interpolation_endpoints():
    a = Pose6D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    b = Pose6D(0.1, 0.2, 0.3, 0.2, -0.1, 0.4)
    pts = interpolate_pose_geodesic(a, b, 6, profile="geodesic")
    assert len(pts) == 6
    assert abs(pts[0].x - a.x) < 1e-9
    assert abs(pts[0].y - a.y) < 1e-9
    assert abs(pts[-1].x - b.x) < 1e-9
    assert abs(pts[-1].y - b.y) < 1e-9
