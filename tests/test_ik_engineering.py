from rebot_sdk.model.inverse_kinematics import IKParams, clamp_config_safe


class _FakeModel:
    lowerPositionLimit = [float("nan"), -1.0, -2.0]
    upperPositionLimit = [float("nan"), 1.0, 2.0]


def test_ik_default_params_match_engineering_target():
    p = IKParams()
    assert p.max_iter == 1000
    assert p.tolerance == 1e-4
    assert p.line_search_steps == 4
    assert p.retry_count == 8


def test_clamp_config_safe_handles_nan_limits_and_q():
    q = [float("nan"), 3.0, -3.0]
    qc = clamp_config_safe(_FakeModel(), q)
    assert qc[0] == 0.0
    assert qc[1] == 1.0
    assert qc[2] == -2.0
