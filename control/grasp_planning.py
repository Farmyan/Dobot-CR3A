"""Top-down parallel-jaw grasps for cubes.
Convention: TCP z axis points DOWN at grasp, so R = Rz(yaw) @ Rx(pi)."""
import numpy as np
from kinematics.transformations import Rz, Rx, make_T


def grasp_pose(cube_pos, yaw):
    return make_T(Rz(yaw) @ Rx(np.pi), cube_pos)


def offset_z(T, dz):
    out = T.copy()
    out[2, 3] += dz
    return out


def grasp_candidates(cube_pos, yaw):
    """Cube is symmetric under 90 deg; a parallel gripper under 180 deg. Offer the two
    distinct finger orientations (yaw, yaw+90) and let IK pick the reachable/closest one."""
    return [grasp_pose(cube_pos, yaw + k * np.pi / 2) for k in (0, 1, -1, 2)]


def place_pose(slot_xy, table_z, cube_size, yaw=0.0):
    return grasp_pose(np.array([slot_xy[0], slot_xy[1], table_z + cube_size / 2]), yaw)
