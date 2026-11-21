from pathlib import Path

def patch(path, pairs):
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        print(("ok      " if a in s else "MISSING ") + path + ": " + a.strip()[:55])
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8")

patch("tools/prepare_assets.py", [
    ('ap.add_argument("--kp", type=float, default=2000.0)', 'ap.add_argument("--kp", type=float, default=1000.0)'),
    ('ap.add_argument("--kv", type=float, default=200.0)', 'ap.add_argument("--kv", type=float, default=60.0)'),
    ('            arm_joints.append((j.get("name"), float(lo), float(hi)))',
     '            arm_joints.append((j.get("name"), float(lo), float(hi)))\n            j.set("armature", "0.1")'),
    ('kp="500", kv="20",', 'kp="1000", kv="40",'),
    ('forcerange="-30 30"', 'forcerange="-40 40"'),
])
patch("assets/gripper/gripper.xml", [
    ('type="slide" axis="0 1 0" range="0 0.03"', 'type="slide" axis="0 1 0" range="0 0.03" armature="0.005"'),
    ('type="slide" axis="0 -1 0" range="0 0.03"', 'type="slide" axis="0 -1 0" range="0 0.03" armature="0.005"'),
])
patch("deployment/simulation_interface.py", [
    ("self.env.step_to(traj.q[-1], 0.15)", "self._settle(traj.q[-1])"),
    ("    def set_gripper(self, width, wait=True):",
     "    def _settle(self, q_target, tol=2e-3, max_s=2.0):\n"
     "        self.env.step_to(q_target, 0.1)\n"
     "        t = 0.0\n"
     "        while t < max_s and np.abs(self.env.joint_positions() - q_target).max() > tol:\n"
     "            self.env.step_to(q_target, 0.05)\n"
     "            t += 0.05\n\n"
     "    def set_gripper(self, width, wait=True):"),
])
patch("control/pick_and_place.py", [
    ("    def compute_observe_pose(self, q_seed):",
     "    def _holding(self):\n"
     "        w = self.robot.get_gripper_width()\n"
     "        return self.grip.min_grasp_width <= w <= self.grip.max_grasp_width\n\n"
     "    def compute_observe_pose(self, q_seed):"),
    ("            self._line_move(offset_z(Tg, self.ws.lift_height), stage, allow_low=True)\n",
     "            self._line_move(offset_z(Tg, self.ws.lift_height), stage, allow_low=True)\n"
     "            if not self._holding():\n"
     "                return PickResult(False, \"lift\", \"cube lost during lift\", det.position)\n"),
    ("            self._joint_move(qp.q, stage)\n",
     "            self._joint_move(qp.q, stage)\n"
     "            if not self._holding():\n"
     "                return PickResult(False, \"transport\", \"cube dropped during transport\", det.position)\n"),
])
