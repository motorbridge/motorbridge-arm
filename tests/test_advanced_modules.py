from motorbridge_arm_sdk import ArmEndPos
from motorbridge_arm_sdk.dynamics import (
    aba_acceleration,
    centroidal_momentum,
    inverse_dynamics_derivatives,
    kinetic_energy,
    load_dynamics_robot_model,
    mass_matrix,
    potential_energy,
    rnea_torque,
    total_energy,
)


def test_arm_endpos_symbol_available():
    assert ArmEndPos is not None


def test_dynamics_fallback_outputs():
    drm = load_dynamics_robot_model("/tmp/nonexistent.urdf")
    q = [0.0, 0.0]
    dq = [0.0, 0.0]
    ddq = [0.0, 0.0]
    tau = [0.0, 0.0]
    M = mass_matrix(drm, q)
    assert len(M) == 2 and len(M[0]) == 2
    assert rnea_torque(drm, q, dq, ddq) == [0.0, 0.0]
    assert aba_acceleration(drm, q, dq, tau) == [0.0, 0.0]
    assert isinstance(kinetic_energy(drm, q, dq), float)
    assert isinstance(potential_energy(drm, q), float)
    assert isinstance(total_energy(drm, q, dq), float)
    assert centroidal_momentum(drm, q, dq) == [0.0] * 6
    d = inverse_dynamics_derivatives(drm, q, dq, ddq)
    assert "dtau_dq" in d and "dtau_ddq" in d and "dtau_dddq" in d
