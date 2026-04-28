from rebot_sdk.motion.geodesic import CliKParams, track_with_clik
from rebot_sdk.types import Pose6D


class _FakeKin:
    _pin = None

    def inverse(self, p, q):
        return [q[0] + 0.01, q[1] + 0.01]


def test_track_with_clik_fallback_path():
    poses = [
        Pose6D(0, 0, 0, 0, 0, 0),
        Pose6D(0.1, 0, 0, 0, 0, 0),
        Pose6D(0.2, 0, 0, 0, 0, 0),
    ]
    out = track_with_clik(model=None, end_frame_id=0, poses=poses, q0=[0.0, 0.0], kin=_FakeKin(), params=CliKParams())
    assert len(out) == 3
    assert out[0].ik_success
    assert out[-1].q[0] > out[0].q[0]
