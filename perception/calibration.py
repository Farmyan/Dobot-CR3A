"""Pinhole camera model and pixel -> base-frame geometry.
Camera frame = OpenCV convention (x right, y down, z forward).
P_base = T_base_cam @ P_cam."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @classmethod
    def from_fovy(cls, fovy_deg, width, height):
        fy = (height / 2) / np.tan(np.radians(fovy_deg) / 2)
        return cls(fy, fy, width / 2, height / 2, width, height)


def pixel_to_ray_cam(uv, intr):
    u, v = uv
    d = np.array([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1.0])
    return d / np.linalg.norm(d)


def ray_plane_intersection(origin, direction, plane_point, plane_normal=(0, 0, 1)):
    n = np.asarray(plane_normal, float)
    denom = float(n @ direction)
    if abs(denom) < 1e-9:
        raise ValueError("ray parallel to plane")
    t = float(n @ (np.asarray(plane_point, float) - origin)) / denom
    if t <= 0:
        raise ValueError("plane behind camera")
    return origin + t * direction


def pixel_to_base_on_plane(uv, intr, T_base_cam, plane_z):
    """Ray-plane intersection with the horizontal plane z = plane_z (base frame)."""
    d_cam = pixel_to_ray_cam(uv, intr)
    origin = T_base_cam[:3, 3]
    d_base = T_base_cam[:3, :3] @ d_cam
    return ray_plane_intersection(origin, d_base, [0, 0, plane_z])


def project_base_to_pixel(p_base, intr, T_base_cam):
    from kinematics.transformations import inv_T
    pc = inv_T(T_base_cam) @ np.append(p_base, 1.0)
    return np.array([intr.fx * pc[0] / pc[2] + intr.cx, intr.fy * pc[1] / pc[2] + intr.cy])
