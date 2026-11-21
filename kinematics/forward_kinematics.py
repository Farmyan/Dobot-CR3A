"""Forward kinematics T_0^6 = A1(q1)...A6(q6) built directly from the URDF joint
chain (origin transforms + joint axes). No hand-typed DH numbers, so it is exactly
the model MuJoCo simulates."""
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import numpy as np
from .transformations import rpy_to_R, make_T, axis_angle_R


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    type: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float = -2 * np.pi
    upper: float = 2 * np.pi
    velocity: float | None = None
    effort: float | None = None


def _floats(s, default):
    return np.array([float(v) for v in s.split()]) if s else np.array(default, float)


def parse_urdf_joints(path):
    root = ET.parse(str(path)).getroot()
    joints = []
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = _floats(o.get("xyz") if o is not None else None, [0, 0, 0])
        rpy = _floats(o.get("rpy") if o is not None else None, [0, 0, 0])
        ax = j.find("axis")
        axis = _floats(ax.get("xyz") if ax is not None else None, [1, 0, 0])
        jt = Joint(j.get("name"), j.find("parent").get("link"), j.find("child").get("link"),
                   j.get("type"), make_T(rpy_to_R(*rpy), xyz), axis / np.linalg.norm(axis))
        lim = j.find("limit")
        if lim is not None and jt.type in ("revolute", "prismatic"):
            jt.lower = float(lim.get("lower", jt.lower))
            jt.upper = float(lim.get("upper", jt.upper))
            if lim.get("velocity"):
                jt.velocity = float(lim.get("velocity"))
            if lim.get("effort"):
                jt.effort = float(lim.get("effort"))
        joints.append(jt)
    return joints


class SerialChain:
    def __init__(self, elements, tcp=None):
        self.elements = elements
        self.actuated = [j for j in elements if j.type in ("revolute", "continuous", "prismatic")]
        self.n = len(self.actuated)
        self.tcp = np.eye(4) if tcp is None else tcp
        self.lower = np.array([j.lower for j in self.actuated])
        self.upper = np.array([j.upper for j in self.actuated])
        self.names = [j.name for j in self.actuated]

    @classmethod
    def from_urdf(cls, path, base_link, tip_link, tcp=None):
        joints = parse_urdf_joints(path)
        by_child = {j.child: j for j in joints}
        chain, link = [], tip_link
        while link != base_link:
            if link not in by_child:
                links = sorted({j.child for j in joints} | {j.parent for j in joints})
                raise ValueError(f"link '{link}' not connected to '{base_link}'. Links: {links}")
            chain.append(by_child[link])
            link = by_child[link].parent
        return cls(chain[::-1], tcp)

    def vel_limits(self, default):
        return np.array([j.velocity if j.velocity else default for j in self.actuated])

    def fk(self, q, return_all=False):
        """4x4 pose of the TCP in the base frame (optionally joint frames + points)."""
        T = np.eye(4)
        i = 0
        frames, pts = [], [T[:3, 3].copy()]
        for j in self.elements:
            T = T @ j.origin
            if j.type in ("revolute", "continuous"):
                frames.append((T[:3, :3] @ j.axis, T[:3, 3].copy()))
                T = T @ make_T(axis_angle_R(j.axis, q[i]))
                i += 1
            elif j.type == "prismatic":
                raise NotImplementedError("prismatic joints are not used by the CR3A")
            pts.append(T[:3, 3].copy())
        T = T @ self.tcp
        return (T, frames, pts) if return_all else T

    def link_points(self, q):
        return self.fk(q, True)[2]

    def jacobian(self, q):
        """Geometric Jacobian (6 x n) in the base frame."""
        T, frames, _ = self.fk(q, True)
        pe = T[:3, 3]
        J = np.zeros((6, self.n))
        for i, (z, p) in enumerate(frames):
            J[:3, i] = np.cross(z, pe - p)
            J[3:, i] = z
        return J
