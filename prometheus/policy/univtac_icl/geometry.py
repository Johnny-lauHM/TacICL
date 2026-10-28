"""Quaternion helpers (wxyz) for base-frame EE deltas."""
import numpy as np


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def rotvec_to_quat(rotvec):
    rotvec = np.asarray(rotvec, dtype=float)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rotvec / angle
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def quat_to_rotvec(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    angle = 2 * np.arctan2(np.linalg.norm(q[1:]), q[0])
    if angle < 1e-12:
        return np.zeros(3)
    return angle * q[1:] / np.linalg.norm(q[1:])


def world_delta(q_from, q_to):
    """Rotation vector r such that q_to = exp(r) * q_from (base frame)."""
    conj = np.array([q_from[0], -q_from[1], -q_from[2], -q_from[3]])
    return quat_to_rotvec(quat_mul(q_to, conj))


def apply_row(position, quat, row):
    """Accumulate one [dx_mm,dy_mm,dz_mm,rx_deg,ry_deg,rz_deg,...] row."""
    position = np.asarray(position, dtype=float) + np.asarray(row[:3], dtype=float) / 1000
    quat = quat_mul(rotvec_to_quat(np.radians(row[3:6])), quat)
    return position, quat / np.linalg.norm(quat)
