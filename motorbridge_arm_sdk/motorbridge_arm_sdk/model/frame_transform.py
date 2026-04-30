"""TCP/EE coordinate frame conversion utilities.
/ TCP/EE 坐标系转换工具。"""

from __future__ import annotations

import math


def _rotation_matrix_to_rpy(R: list[list[float]]) -> tuple[float, float, float]:
    """Extract roll, pitch, yaw (ZYX convention) from a 3x3 rotation matrix.

    / 从 3x3 旋转矩阵中提取横滚、俯仰、偏航角（ZYX 约定）。
    """
    sy = math.sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0])
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2][1], R[2][2])
        pitch = math.atan2(-R[2][0], sy)
        yaw = math.atan2(R[1][0], R[0][0])
    else:
        roll = math.atan2(-R[1][2], R[1][1])
        pitch = math.atan2(-R[2][0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def _rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    """Build a 3x3 rotation matrix from roll, pitch, yaw (ZYX convention).

    / 从横滚、俯仰、偏航角构建 3x3 旋转矩阵（ZYX 约定）。
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _mat_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Multiply two 3x3 matrices. / 两个 3x3 矩阵相乘。"""
    result = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += A[i][k] * B[k][j]
    return result


def _transpose(R: list[list[float]]) -> list[list[float]]:
    """Transpose a 3x3 matrix. / 3x3 矩阵转置。"""
    return [[R[j][i] for j in range(3)] for i in range(3)]


def _rotation_shortest_path(
    roll: float, pitch: float, yaw: float,
) -> tuple[float, float, float]:
    """Normalize rotation angles to the shortest angular path.

    Ensures each component lies within [-pi, pi].

    / 将旋转角度归一化为最短角路径，确保每个分量在 [-pi, pi] 内。
    """
    return (
        math.atan2(math.sin(roll), math.cos(roll)),
        math.atan2(math.sin(pitch), math.cos(pitch)),
        math.atan2(math.sin(yaw), math.cos(yaw)),
    )


# Fixed rotation from EE frame to TCP frame (ZYX convention).
# R_tcp_ee rotates a vector expressed in EE frame to TCP frame.
R_TCP_EE: list[list[float]] = [
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
]

# Inverse (transpose, since R is orthogonal).
R_EE_TCP: list[list[float]] = _transpose(R_TCP_EE)


def ee_to_tcp(ee_pose: list[float]) -> list[float]:
    """Convert EE pose (x, y, z, roll, pitch, yaw) to TCP pose.

    Applies the fixed rotation ``R_tcp_ee`` between EE and TCP frames
    and inverts the rotation vector to use the shorter angular path.

    / 将末端执行器位姿 (x, y, z, roll, pitch, yaw) 转换为工具中心点位姿。
    应用 EE 到 TCP 的固定旋转，并将旋转向量归一化为最短路径。

    Args:
        ee_pose: ``[x, y, z, roll, pitch, yaw]`` in EE frame.
                 EE 坐标系下的 ``[x, y, z, roll, pitch, yaw]``。

    Returns:
        ``[x, y, z, roll, pitch, yaw]`` in TCP frame.
        TCP 坐标系下的 ``[x, y, z, roll, pitch, yaw]``。
    """
    if len(ee_pose) != 6:
        raise ValueError(f"ee_pose must have 6 elements, got {len(ee_pose)}")

    x, y, z, roll, pitch, yaw = ee_pose

    # Rotate the position vector from EE frame to TCP frame.
    p_ee = [x, y, z]
    p_tcp = [
        R_TCP_EE[0][0] * p_ee[0] + R_TCP_EE[0][1] * p_ee[1] + R_TCP_EE[0][2] * p_ee[2],
        R_TCP_EE[1][0] * p_ee[0] + R_TCP_EE[1][1] * p_ee[1] + R_TCP_EE[1][2] * p_ee[2],
        R_TCP_EE[2][0] * p_ee[0] + R_TCP_EE[2][1] * p_ee[1] + R_TCP_EE[2][2] * p_ee[2],
    ]

    # Compose rotations: R_tcp = R_tcp_ee * R_ee
    R_ee = _rpy_to_rotation_matrix(roll, pitch, yaw)
    R_tcp = _mat_mul(R_TCP_EE, R_ee)

    # Extract RPY from the composed rotation.
    r_tcp, p_tcp_rot, y_tcp = _rotation_matrix_to_rpy(R_tcp)

    # Use the shortest angular path.
    r_tcp, p_tcp_rot, y_tcp = _rotation_shortest_path(r_tcp, p_tcp_rot, y_tcp)

    return [p_tcp[0], p_tcp[1], p_tcp[2], r_tcp, p_tcp_rot, y_tcp]


def tcp_to_ee(tcp_pose: list[float]) -> list[float]:
    """Convert TCP pose (x, y, z, roll, pitch, yaw) back to EE pose.

    Applies the inverse rotation ``R_ee_tcp`` and normalizes to the
    shortest angular path.

    / 将工具中心点位姿 (x, y, z, roll, pitch, yaw) 转换回末端执行器位姿。
    应用逆旋转 ``R_ee_tcp`` 并归一化为最短角路径。

    Args:
        tcp_pose: ``[x, y, z, roll, pitch, yaw]`` in TCP frame.
                  TCP 坐标系下的 ``[x, y, z, roll, pitch, yaw]``。

    Returns:
        ``[x, y, z, roll, pitch, yaw]`` in EE frame.
        EE 坐标系下的 ``[x, y, z, roll, pitch, yaw]``。
    """
    if len(tcp_pose) != 6:
        raise ValueError(f"tcp_pose must have 6 elements, got {len(tcp_pose)}")

    x, y, z, roll, pitch, yaw = tcp_pose

    # Rotate position from TCP frame back to EE frame.
    p_tcp = [x, y, z]
    p_ee = [
        R_EE_TCP[0][0] * p_tcp[0] + R_EE_TCP[0][1] * p_tcp[1] + R_EE_TCP[0][2] * p_tcp[2],
        R_EE_TCP[1][0] * p_tcp[0] + R_EE_TCP[1][1] * p_tcp[1] + R_EE_TCP[1][2] * p_tcp[2],
        R_EE_TCP[2][0] * p_tcp[0] + R_EE_TCP[2][1] * p_tcp[1] + R_EE_TCP[2][2] * p_tcp[2],
    ]

    # Compose rotations: R_ee = R_ee_tcp * R_tcp
    R_tcp = _rpy_to_rotation_matrix(roll, pitch, yaw)
    R_ee = _mat_mul(R_EE_TCP, R_tcp)

    # Extract RPY.
    r_ee, p_ee_rot, y_ee = _rotation_matrix_to_rpy(R_ee)

    # Use the shortest angular path.
    r_ee, p_ee_rot, y_ee = _rotation_shortest_path(r_ee, p_ee_rot, y_ee)

    return [p_ee[0], p_ee[1], p_ee[2], r_ee, p_ee_rot, y_ee]
