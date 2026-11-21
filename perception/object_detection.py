"""Multi-color cube and target floor-zone detection + ray-plane localisation."""
from dataclasses import dataclass
import numpy as np
import cv2
from .calibration import pixel_to_base_on_plane

COLOR_HSV_RANGES = {
    # Keep red's high-hue band separate from magenta so pink surfaces are
    # not lost at the edge of a narrow HSV interval. OpenCV hue is 0..179.
    "red":     [((0, 90, 50), (10, 255, 255)), ((174, 90, 50), (179, 255, 255))],
    "yellow":  [((18, 75, 60), (38, 255, 255))],
    "green":   [((45, 65, 45), (84, 255, 255))],
    "blue":    [((100, 75, 45), (135, 255, 255))],
    "magenta": [((130, 55, 45), (173, 255, 255))],
}


@dataclass
class CubeDetection:
    pixel: tuple
    position: np.ndarray    # cube centre, base frame (m)
    yaw: float              # rotation about base z, folded to [-45deg, 45deg)
    area_px: float
    color: str = "red"


@dataclass
class ZoneDetection:
    color: str
    position: np.ndarray    # zone centre on table (x, y, table_z)
    area_px: float


def fold_yaw(yaw):
    """Cube has 90-degree symmetry."""
    return (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4


class CubeDetector:
    def __init__(self, cube_size, table_z, min_area_px=120, zone_area_thresh_px=4500):
        self.size, self.table_z = cube_size, table_z
        self.min_area = min_area_px
        self.zone_thresh = zone_area_thresh_px

    def color_mask(self, hsv, color_name):
        m = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in COLOR_HSV_RANGES[color_name]:
            m = cv2.bitwise_or(m, cv2.inRange(hsv, lo, hi))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    def detect_all(self, rgb, intr, T_base_cam):
        """Return (cubes, zones) detected across all configured colors."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        top_z = self.table_z + self.size
        cubes, zones = [], {}

        for color_name in COLOR_HSV_RANGES:
            mask = self.color_mask(hsv, color_name)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < self.min_area:
                    continue
                (u, v), (w, h), ang = cv2.minAreaRect(cv2.convexHull(c))
                if area >= self.zone_thresh:
                    p_zone = pixel_to_base_on_plane((u, v), intr, T_base_cam, self.table_z)
                    zones[color_name] = ZoneDetection(color_name, p_zone, area)
                else:
                    top = pixel_to_base_on_plane((u, v), intr, T_base_cam, top_z)
                    a = np.radians(ang)
                    du = np.array([np.cos(a), np.sin(a)]) * 10
                    p2 = pixel_to_base_on_plane((u + du[0], v + du[1]), intr, T_base_cam, top_z)
                    yaw = fold_yaw(np.arctan2(p2[1] - top[1], p2[0] - top[0]))
                    centre = np.array([top[0], top[1], self.table_z + self.size / 2])
                    cubes.append(CubeDetection((u, v), centre, yaw, area, color=color_name))
        return cubes, zones

    def detect(self, rgb, intr, T_base_cam):
        cubes, _ = self.detect_all(rgb, intr, T_base_cam)
        return cubes
