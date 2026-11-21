"""RobotInterface implementation backed by MuJoCo (keeps UI responsive during trajectories)."""
import numpy as np
from deployment.robot_interface import RobotInterface


class SimulationInterface(RobotInterface):
    def __init__(self, env, viewer=None, ui_pump=None):
        self.env, self.viewer, self.ui_pump = env, viewer, ui_pump
        self.max_track_err = 0.0
        self.lead = 0.06
        self.abort_requested = False

    def _pump(self):
        if self.viewer is not None:
            self.viewer.sync()
        if self.ui_pump is not None:
            self.ui_pump()

    def get_joint_positions(self):
        return self.env.joint_positions()

    def execute_trajectory(self, traj):
        self.env.phase = getattr(self, "stage", "")
        qd = np.gradient(traj.q, traj.dt, axis=0)
        for step_i, (q, v) in enumerate(zip(traj.q, qd)):
            if self.abort_requested:
                self.stop()
                return False
            self.env.step_to(q + self.lead * v, traj.dt)
            self.max_track_err = max(self.max_track_err, float(np.abs(self.env.joint_positions() - q).max()))
            if step_i % 2 == 0:
                self._pump()
        self._settle(traj.q[-1])
        return True

    def _settle(self, q_target, tol=2e-3, max_s=2.0):
        self.env.step_to(q_target, 0.1)
        t = 0.0
        while t < max_s and np.abs(self.env.joint_positions() - q_target).max() > tol:
            if self.abort_requested:
                return
            self.env.step_to(q_target, 0.05)
            self._pump()
            t += 0.05

    def set_gripper(self, width, wait=True):
        # Match the validated AG95 position controller.
        self.env.model.actuator_gainprm[self.env.grip_act, 0] = 100.0
        self.env.model.actuator_biasprm[self.env.grip_act, 1] = -100.0
        self.env.model.actuator_biasprm[self.env.grip_act, 2] = -2.0

        # Public API width is the physical jaw gap in meters.
        angle = self.env.gripper_angle_for_width(width)
        self.env.data.ctrl[self.env.grip_act] = angle

        qadr = self.env.grip_q[0]
        print(
            f"[GRIPPER START] requested={width*1000:.1f}mm "
            f"target_angle={angle:.5f} actual_angle={self.env.data.qpos[qadr]:.5f}",
            flush=True,
        )

        if wait:
            q_hold = self.env.joint_positions()
            # Closing against a cube needs additional settling time.
            steps = 200 if width < 0.040 else 40
            for _ in range(steps):
                if self.abort_requested:
                    return
                self.env.step_to(q_hold, 0.03)
                self._pump()

        print(
            f"[GRIPPER END] actual_angle={self.env.data.qpos[qadr]:.5f} "
            f"joint_velocity={self.env.data.qvel[self.env.model.joint('ag95_finger1_joint').dofadr[0]]:.5f} "
            f"actuator_force={self.env.data.actuator_force[self.env.grip_act]:.3f} "
            f"measured_width={self.env.gripper_width()*1000:.1f}mm",
            flush=True,
        )

    def get_gripper_width(self):
        return self.env.gripper_width()

    def get_errors(self):
        return []

    def stop(self):
        self.env.data.ctrl[self.env.arm_act] = self.env.joint_positions()

