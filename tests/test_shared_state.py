"""Tests for the shared-memory IPC layer."""
import struct
from unittest.mock import MagicMock, patch

from motorbridge_arm_sdk.ipc.shared_state import (
    SharedArmState,
    _DATA_OFFSET,
    _HEADER_FMT,
    _HEADER_SIZE,
    _JOINT_FMT,
    _JOINT_SIZE,
    _LOCK_FREE,
    _LOCK_HELD,
    _LOCK_OFFSET,
    _MAGIC,
)


def test_constants_consistency():
    assert _HEADER_SIZE == 16
    assert _JOINT_SIZE == 28
    assert _DATA_OFFSET == _HEADER_SIZE + 1 + 3
    assert _MAGIC == 0x41524D53


def test_write_read_roundtrip():
    try:
        from multiprocessing.shared_memory import SharedMemory
    except ImportError:
        return  # skip on platforms without SharedMemory

    name = "test_arm_shm_roundtrip"
    # Clean up any stale segment
    try:
        SharedMemory(name=name).unlink()
    except Exception:
        pass

    writer = SharedArmState(name=name, num_joints=3, create=True)
    assert writer.active

    try:
        writer.write(
            positions=[1.0, 2.0, 3.0],
            velocities=[0.1, 0.2, 0.3],
            torques=[5.0, 6.0, 7.0],
            statuses=[1, 1, 0],
        )

        reader = SharedArmState(name=name, num_joints=3, create=False)
        assert reader.active

        result = reader.read()
        assert result["tick"] == 1
        assert result["num_joints"] == 3
        assert len(result["positions"]) == 3
        assert abs(result["positions"][0] - 1.0) < 1e-9
        assert abs(result["positions"][1] - 2.0) < 1e-9
        assert abs(result["positions"][2] - 3.0) < 1e-9
        assert abs(result["velocities"][0] - 0.1) < 1e-9
        assert abs(result["torques"][2] - 7.0) < 1e-9
        assert result["statuses"] == [1, 1, 0]

        reader.close()
    finally:
        writer.close()
        writer.unlink()


def test_tick_increments_on_each_write():
    try:
        from multiprocessing.shared_memory import SharedMemory
    except ImportError:
        return

    name = "test_arm_shm_tick"
    try:
        SharedMemory(name=name).unlink()
    except Exception:
        pass

    shm = SharedArmState(name=name, num_joints=2, create=True)
    assert shm.active

    try:
        assert shm.tick == 0
        shm.write([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0, 0])
        assert shm.tick == 1
        shm.write([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0, 0])
        assert shm.tick == 2
    finally:
        shm.close()
        shm.unlink()


def test_write_insufficient_length_skipped():
    try:
        from multiprocessing.shared_memory import SharedMemory
    except ImportError:
        return

    name = "test_arm_shm_short"
    try:
        SharedMemory(name=name).unlink()
    except Exception:
        pass

    shm = SharedArmState(name=name, num_joints=3, create=True)
    assert shm.active

    try:
        # Only provide 2 values when 3 are expected
        shm.write([1.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0, 0])
        # Tick should not have incremented
        assert shm.tick == 0
    finally:
        shm.close()
        shm.unlink()


def test_inactive_when_shared_memory_unavailable():
    with patch("motorbridge_arm_sdk.ipc.shared_state.SharedMemory", None):
        shm = SharedArmState(name="test_noop", num_joints=2)
        assert not shm.active
        assert shm.tick == 0
        # write and read should be no-ops
        shm.write([1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0, 0])
        assert shm.read() == {}


def test_properties():
    try:
        from multiprocessing.shared_memory import SharedMemory
    except ImportError:
        return

    name = "test_arm_shm_props"
    try:
        SharedMemory(name=name).unlink()
    except Exception:
        pass

    shm = SharedArmState(name=name, num_joints=6, create=True)
    try:
        assert shm.name == name
        assert shm.num_joints == 6
        assert shm.active is True
    finally:
        shm.close()
        shm.unlink()


def test_repr():
    try:
        from multiprocessing.shared_memory import SharedMemory
    except ImportError:
        return

    name = "test_arm_shm_repr"
    try:
        SharedMemory(name=name).unlink()
    except Exception:
        pass

    shm = SharedArmState(name=name, num_joints=3, create=True)
    try:
        r = repr(shm)
        assert "active" in r
        assert name in r
    finally:
        shm.close()
        shm.unlink()


def test_close_sets_inactive():
    try:
        from multiprocessing.shared_memory import SharedMemory
    except ImportError:
        return

    name = "test_arm_shm_close"
    try:
        SharedMemory(name=name).unlink()
    except Exception:
        pass

    shm = SharedArmState(name=name, num_joints=2, create=True)
    assert shm.active
    shm.close()
    assert not shm.active
