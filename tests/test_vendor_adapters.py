from rebot_sdk.vendors import create_default_adapter_registry


class _FakeController:
    def add_robstride_motor(self, esc_id, feedback_id, model):
        return ("robstride", esc_id, feedback_id, model)


def test_vendor_adapter_create_motor():
    reg = create_default_adapter_registry()
    ctrl = _FakeController()
    m = reg.create_motor(ctrl, "robstride", 1, 0xFD, "rs-00")
    assert m == ("robstride", 1, 0xFD, "rs-00")
