"""Central configuration. Every value marked PLACEHOLDER must be replaced with
real data from the CR3A datasheet / your calibration before real-robot use."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent

CUBE_COLORS = {
    "red":     (0.95, 0.10, 0.10, 1.0),
    "yellow":  (0.96, 0.88, 0.05, 1.0),
    "green":   (0.08, 0.85, 0.15, 1.0),
    "blue":    (0.08, 0.22, 0.96, 1.0),
}


@dataclass
class RobotConfig:
    urdf_path: Path = ROOT / "assets/urdf/cr3a.urdf"
    base_link: str = "base_link"
    flange_link: str = "Link6"
    # Link6-to-physical-AG95-TCP transform measured from simulation/robot.xml.
    tcp_xyz: tuple = (0.00213861, -0.00708698, 0.15982866)
    tcp_rpy: tuple = (-0.04513534, -0.01024624, -3.07159361)
    vel_scale: float = 0.36
    acc_scale: float = 0.36
    default_vmax: float = 3.14
    default_amax: float = 6.28
    joint_sign: tuple = (1, 1, 1, 1, 1, 1)
    joint_offset_deg: tuple = (0, 0, 0, 0, 0, 0)


@dataclass
class WorkspaceConfig:
    table_z: float = 0.0
    cube_size: float = 0.025               # 25 mm sponge cube
    zone_half_size: float = 0.060          # 120 x 120 mm colored floor pad (fits 4 cubes with 39 mm gap)
    # Sweet-spot top-down reachability window (never too close to base or too far)
    spawn_x: tuple = (0.20, 0.60)
    spawn_y: tuple = (-0.50, 0.50)
    min_r: float = 0.32
    max_r: float = 0.48
    min_cube_spacing: float = 0.068        # > 55 mm open gripper width so fingers never clip neighbors
    min_zone_spacing: float = 0.145        # keeps 120x120 mm pads well separated
    place_xy: tuple = (0.15, -0.30)
    place_spacing: float = 0.045
    observe_xyz: tuple = (0.05, 0.45, 0.22)
    # Physical TCP is 30 mm above the cube centre for the top-down grasp.
    tcp_to_cube_center_z: float = 0.030
    approach_height: float = 0.08
    lift_height: float = 0.10
    tcp_box: tuple = (-0.1, 0.6, -0.55, 0.55, 0.005, 0.55)
    link_min_z: float = 0.02


@dataclass
class CameraConfig:
    name: str = "vx500_cam"
    width: int = 1280
    height: int = 960
    fovy_deg: float = 54.0
    real_intrinsics: tuple | None = None
    real_T_base_cam: np.ndarray | None = None


@dataclass
class GripperConfig:
    open_width: float = 0.055
    close_width: float = 0.0242            # ~0.8 mm gentle sponge compression on 25 mm cube
    grasp_force_note: str = "light sponge compression (24.0-24.5 mm)"
    min_grasp_width: float = 0.0220
    max_grasp_width: float = 0.0260


ROBOT = RobotConfig()
WORKSPACE = WorkspaceConfig()
CAMERA = CameraConfig()
GRIPPER = GripperConfig()



