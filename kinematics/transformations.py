"""Homogeneous-transform helpers (SE(3)). Frames: T_a_b maps points from b to a."""
import numpy as np


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rpy_to_R(r, p, y):
    """URDF convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return Rz(y) @ Ry(p) @ Rx(r)


def make_T(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def inv_T(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def axis_angle_R(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def rotvec_to_R(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    return axis_angle_R(w / th, th)


def R_to_rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if np.pi - th < 1e-6:  # near 180 deg: axis from the symmetric part
        A = (R + np.eye(3)) / 2
        k = int(np.argmax(np.diag(A)))
        axis = A[:, k] / np.sqrt(max(A[k, k], 1e-12))
        return axis * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def pose_error(T_cur, T_tgt):
    """6-vector [dp, dtheta] in the base frame that moves T_cur onto T_tgt."""
    dp = T_tgt[:3, 3] - T_cur[:3, 3]
    dr = R_to_rotvec(T_tgt[:3, :3] @ T_cur[:3, :3].T)
    return np.concatenate([dp, dr])


def interpolate_pose(T0, T1, s):
    T = np.eye(4)
    T[:3, 3] = (1 - s) * T0[:3, 3] + s * T1[:3, 3]
    T[:3, :3] = T0[:3, :3] @ rotvec_to_R(s * R_to_rotvec(T0[:3, :3].T @ T1[:3, :3]))
    return T


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))
