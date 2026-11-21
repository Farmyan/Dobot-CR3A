"""Hardware-agnostic robot interface + DOBOT TCP/IP implementation.

The high-level controller only talks to RobotInterface, so the same code runs in
simulation (SimulationInterface) and on the real CR3A (DobotInterface).

!! DobotInterface: the command strings below follow the DOBOT TCP/IP protocol (V4,
dashboard port 29999) as I know it, but I could not verify them against your
controller firmware. Check every entry of COMMANDS against the official
"TCP/IP Remote Control Interface" PDF for your firmware BEFORE moving the robot.
Real robot angles are in DEGREES; this class converts to/from URDF radians using
the sign/offset conventions in config.RobotConfig."""
from abc import ABC, abstractmethod
import socket
import re
import time
import numpy as np


class RobotInterface(ABC):
    @abstractmethod
    def get_joint_positions(self) -> np.ndarray: ...          # rad (URDF convention)

    @abstractmethod
    def execute_trajectory(self, traj) -> bool: ...           # blocks until done; False on error

    @abstractmethod
    def set_gripper(self, width: float, wait: bool = True) -> None: ...

    @abstractmethod
    def get_gripper_width(self) -> float: ...

    @abstractmethod
    def get_errors(self) -> list: ...

    @abstractmethod
    def stop(self) -> None: ...                               # safe stop of current motion


class DobotInterface(RobotInterface):
    COMMANDS = {                                              # VERIFY against protocol doc
        "enable": "EnableRobot()",
        "disable": "DisableRobot()",
        "clear_error": "ClearError()",
        "get_angle": "GetAngle()",
        "get_error": "GetErrorID()",
        "servoj": "ServoJ({j},t={dt})",
        "stop": "Stop()",
        "do": "DO({idx},{val})",
    }

    def __init__(self, ip="192.168.5.1", port=29999, gripper_do=1, sign=(1,) * 6,
                 offset_deg=(0,) * 6, gripper_open_width=0.06, speed_scale=0.1):
        self.sock = socket.create_connection((ip, port), timeout=5)
        self.sign, self.offset = np.array(sign, float), np.array(offset_deg, float)
        self.gripper_do, self.open_w = gripper_do, gripper_open_width
        self._width = gripper_open_width
        self.speed_scale = speed_scale          # >1 stretches time => slower
        self._send(self.COMMANDS["clear_error"])
        self._send(self.COMMANDS["enable"])

    def _send(self, cmd):
        self.sock.sendall((cmd + "\n").encode())
        return self.sock.recv(4096).decode(errors="ignore")

    def _to_robot(self, q):
        return self.sign * np.degrees(q) + self.offset

    def _from_robot(self, deg):
        return np.radians((np.array(deg) - self.offset) / self.sign)

    def get_joint_positions(self):
        resp = self._send(self.COMMANDS["get_angle"])
        nums = re.findall(r"-?\d+\.?\d*", resp.split("{")[1].split("}")[0])
        return self._from_robot([float(x) for x in nums[:6]])

    def execute_trajectory(self, traj):
        dt = traj.dt / self.speed_scale
        for q in traj.q:
            j = ",".join(f"{v:.4f}" for v in self._to_robot(q))
            self._send(self.COMMANDS["servoj"].format(j=j, dt=f"{dt:.3f}"))
            time.sleep(dt)
            if self.get_errors():
                self.stop()
                return False
        return True

    def set_gripper(self, width, wait=True):
        closed = width < self.open_w / 2        # digital-output gripper: open/close only
        self._send(self.COMMANDS["do"].format(idx=self.gripper_do, val=0 if closed else 1))
        self._width = width
        if wait:
            time.sleep(0.8)

    def get_gripper_width(self):
        return self._width                      # replace with real feedback if available

    def get_errors(self):
        resp = self._send(self.COMMANDS["get_error"])
        ids = re.findall(r"\d+", resp.split("{")[1].split("}")[0]) if "{" in resp else []
        return [i for i in ids if i != "0"]

    def stop(self):
        self._send(self.COMMANDS["stop"])
