from rebot_sdk.params.registry import create_default_registry


def test_registry_has_robstride_timeout():
    reg = create_default_registry()
    spec = reg.get("robstride", 0x200C)
    assert spec is not None
    assert spec.param_type == "u32"
