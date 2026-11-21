from pathlib import Path

def patch(path, pairs):
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        print(("ok      " if s.count(a) == 1 else "PROBLEM(%d) " % s.count(a)) + path + ": " + a.strip()[:60])
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8")

patch("simulation/environment.py", [
    ("        self.unsafe_contacts = 0\n        self.joint_violations = 0\n",
     "        self.unsafe_contacts = 0\n        self.joint_violations = 0\n        self.unsafe_pairs = {}\n        self.phase = \"\"\n"),
    ("        self.unsafe_contacts = self.joint_violations = 0\n",
     "        self.unsafe_contacts = self.joint_violations = 0\n        self.unsafe_pairs = {}\n"),
    ("                self.unsafe_contacts += 1          # link-link self collision\n",
     "                self.unsafe_contacts += 1          # link-link self collision\n                self._note(\"self\", c)\n"),
    ("                        self.unsafe_contacts += 1\n",
     "                        self.unsafe_contacts += 1\n                        self._note(\"table\" if other_table else \"cube\", c)\n"),
    ("    def config_collision_free(self, q):\n",
     "    def _note(self, kind, c):\n"
     "        m = self.model\n"
     "        g = lambda i: f\"{m.body(m.geom_bodyid[i]).name}/{m.geom(i).name or 'mesh'}\"\n"
     "        key = f\"[{self.phase}] {kind}: {g(c.geom1)} <-> {g(c.geom2)}\"\n"
     "        self.unsafe_pairs[key] = self.unsafe_pairs.get(key, 0) + 1\n\n"
     "    def config_collision_free(self, q):\n"),
])
patch("deployment/simulation_interface.py", [
    ("        self.max_track_err = 0.0\n",
     "        self.max_track_err = 0.0\n        self.lead = 0.06        # s, velocity lead = kv/kp of the arm actuators (cancels tracking lag)\n"),
    ("        for q in traj.q:\n            self.env.step_to(q, traj.dt)\n",
     "        self.env.phase = getattr(self, \"stage\", \"\")\n"
     "        qd = np.gradient(traj.q, traj.dt, axis=0)\n"
     "        for q, v in zip(traj.q, qd):\n"
     "            self.env.step_to(q + self.lead * v, traj.dt)\n"),
])
patch("control/pick_and_place.py", [
    ("    def _run(self, traj, stage, allow_low=False):\n        self._validate(traj, stage, allow_low)\n",
     "    def _run(self, traj, stage, allow_low=False):\n        self.robot.stage = stage\n        self._validate(traj, stage, allow_low)\n"),
])
patch("deployment/safety.py", [
    ("            step = max(1, len(q) // 40)\n            for qi in list(q[::step]) + [q[-1]]:\n",
     "            dq = np.abs(np.diff(q, axis=0)).max(axis=1) if len(q) > 1 else np.zeros(0)\n"
     "            cum = np.concatenate([[0.0], np.cumsum(dq)])\n"
     "            marks = np.arange(0.0, cum[-1] + 1e-9, 0.02)          # a collision check every ~0.02 rad\n"
     "            idx = np.unique(np.clip(np.searchsorted(cum, marks), 0, len(q) - 1))\n"
     "            for qi in list(q[idx]) + [q[-1]]:\n"),
])
patch("main.py", [
    ("unsafe_contacts=env.unsafe_contacts,\n",
     "unsafe_contacts=env.unsafe_contacts,\n                        unsafe_pairs=dict(env.unsafe_pairs),\n"),
])
