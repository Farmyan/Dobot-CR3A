from pathlib import Path

def patch(path, pairs):
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        print(("ok      " if s.count(a) == 1 else "PROBLEM(%d) " % s.count(a)) + path + ": " + a.strip()[:60])
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8")

patch("simulation/environment.py", [
    ("        self._scratch = mujoco.MjData(self._cm)\n",
     "        self._scratch = mujoco.MjData(self._cm)\n"
     "        fl = m.body(robot_cfg.flange_link).id\n"
     "        self.wrist_bodies = {fl, int(m.body_parentid[fl])}   # gripper is rigidly mounted here\n"),
    ("            if g1 and g2:\n                continue\n            if r1 and r2:\n",
     "            if g1 and g2:\n                continue\n"
     "            if (g1 and b2 in self.wrist_bodies) or (g2 and b1 in self.wrist_bodies):\n"
     "                continue                                   # gripper is rigidly mounted on the wrist\n"
     "            if r1 and r2:\n"),
    ("    # ---- sensing ----\n",
     "    def explain_collision(self, q):\n"
     "        \"\"\"Names of all part pairs within the planning margin at configuration q (for debugging).\"\"\"\n"
     "        s, m = self._scratch, self._cm\n"
     "        s.qpos[:] = self.data.qpos\n"
     "        for a, v in zip(self.jadr, q):\n"
     "            s.qpos[a] = v\n"
     "        mujoco.mj_fwdPosition(m, s)\n"
     "        g = lambda i: f\"{m.body(m.geom_bodyid[i]).name}/{m.geom(i).name or 'mesh'}\"\n"
     "        return sorted({f\"{g(s.contact[i].geom1)} <-> {g(s.contact[i].geom2)} dist={s.contact[i].dist*1000:.1f}mm\"\n"
     "                       for i in range(s.ncon)})\n\n"
     "    # ---- sensing ----\n"),
])
patch("main.py", [
    ("    print('clearance margin %.1f mm; observe pose passes the check: %s' % (margin * 1000, env.config_collision_free(sol.q)))\n",
     "    ok_obs = env.config_collision_free(sol.q)\n"
     "    print('clearance margin %.1f mm; observe pose passes the check: %s' % (margin * 1000, ok_obs))\n"
     "    if not ok_obs:\n"
     "        print('  parts within the margin at the observe pose:')\n"
     "        for line in env.explain_collision(sol.q):\n"
     "            print('   ', line)\n"
     "        raise SystemExit('Lower the margin (--margin 0.0015) or send me the list above.')\n"),
])
