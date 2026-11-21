"""Camera abstraction shared by simulation and the real VX500."""
from abc import ABC, abstractmethod
import numpy as np
from .calibration import Intrinsics


class Camera(ABC):
    @abstractmethod
    def capture(self) -> np.ndarray:
        """Return an RGB uint8 image (H, W, 3)."""

    @property
    @abstractmethod
    def intrinsics(self) -> Intrinsics: ...

    @abstractmethod
    def T_base_cam(self) -> np.ndarray:
        """4x4 pose of the camera (OpenCV convention) in the robot base frame."""


class RealVX500Camera(Camera):
    """Stub. Fill with the DOBOT VX500 SDK grab call, its factory/own intrinsics and a
    hand-eye (eye-to-hand) calibration result."""

    def __init__(self, grab_fn, intrinsics: Intrinsics, T_base_cam: np.ndarray):
        self._grab, self._intr, self._T = grab_fn, intrinsics, T_base_cam

    def capture(self):
        return self._grab()

    @property
    def intrinsics(self):
        return self._intr

    def T_base_cam(self):
        return self._T
