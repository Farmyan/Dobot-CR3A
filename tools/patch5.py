from pathlib import Path

def patch(path, pairs):
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        print(("ok      " if s.count(a) == 1 else "PROBLEM(%d) " % s.count(a)) + path + ": " + a.strip()[:55])
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8")

patch("kinematics/inverse_kinematics.py", [
    ("        self.margin = margin\n", "        self.margin = margin\n        self.is_valid = None       # optional callback q -> bool (e.g. collision check)\n"),
    ("        first = self.solve_from(T_target, q_seed)\n"
     "        if first.converged and first.sigma_min >= self.sigma_thresh:\n"
     "            return first\n"
     "        sols = self.solve_all(T_target, q_seed)\n"
     "        return sols[0] if sols else None\n",
     "        ok = (lambda q: True) if self.is_valid is None else self.is_valid\n"
     "        first = self.solve_from(T_target, q_seed)\n"
     "        if first.converged and first.sigma_min >= self.sigma_thresh and ok(first.q):\n"
     "            return first\n"
     "        for s in self.solve_all(T_target, q_seed):\n"
     "            if ok(s.q):\n"
     "                return s\n"
     "        return None\n"),
])
patch("deployment/safety.py", [
    ("        self.margin = joint_margin\n", "        self.margin = joint_margin\n        self.collision_fn = None   # optional callback q -> True if collision-free\n"),
    ("        x0, x1, y0, y1, z0, z1 = self.ws.tcp_box\n",
     "        if self.collision_fn is not None:\n"
     "            step = max(1, len(q) // 40)\n"
     "            for qi in list(q[::step]) + [q[-1]]:\n"
     "                if not self.collision_fn(qi):\n"
     "                    r.append('configuration in collision (self/table) along trajectory')\n"
     "                    break\n"
     "        x0, x1, y0, y1, z0, z1 = self.ws.tcp_box\n"),
])
patch("simulation/environment.py", [
    ("        self.dt = m.opt.timestep\n", "        self.dt = m.opt.timestep\n        self._scratch = mujoco.MjData(self.model)\n"),
    ("            for ba, bb, gb in ((b1, b2, c.geom2), (b2, b1, c.geom1)):\n",
     "            if b1 in self.robot_bodies and b2 in self.robot_bodies:\n"
     "                self.unsafe_contacts += 1          # link-link self collision\n"
     "                continue\n"
     "            for ba, bb, gb in ((b1, b2, c.geom2), (b2, b1, c.geom1)):\n"),
    ("    # ---- sensing ----\n",
     "    def config_collision_free(self, q):\n"
     "        \"\"\"True if joint vector q has no self/table collision (cubes and gripper-gripper ignored).\"\"\"\n"
     "        s, m = self._scratch, self.model\n"
     "        s.qpos[:] = self.data.qpos\n"
     "        for a, v in zip(self.jadr, q):\n"
     "            s.qpos[a] = v\n"
     "        mujoco.mj_fwdPosition(m, s)\n"
     "        for i in range(s.ncon):\n"
     "            c = s.contact[i]\n"
     "            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]\n"
     "            if b1 in self.cube_bodies or b2 in self.cube_bodies:\n"
     "                continue\n"
     "            if b1 in self.gripper_bodies and b2 in self.gripper_bodies:\n"
     "                continue\n"
     "            if b1 in self.robot_bodies and b2 in self.robot_bodies:\n"
     "                return False\n"
     "            if (b1 in self.robot_bodies or b2 in self.robot_bodies) and self.table_geom in (c.geom1, c.geom2):\n"
     "                if self.base_body not in (b1, b2):\n"
     "                    return False\n"
     "        return True\n\n"
     "    # ---- sensing ----\n"),
])
patch("main.py", [
    ("    pnp.q_observe = sol.q\n",
     "    pnp.q_observe = sol.q\n    pnp.ik.is_valid = env.config_collision_free\n    pnp.safety.collision_fn = env.config_collision_free\n"),
    ('grasp_ok=sum(r.success or r.stage not in ("grasp",) and r.stage != "grasp_plan" for r in results),',
     'grasp_ok=sum(r.success or r.stage not in ("grasp_plan", "approach", "descend", "grasp", "lift") for r in results),'),
    ('                   pick_place_success=sum(l["placed"] for l in log) / max(att, 1),',
     '                   grasp_success=sum(l["grasp_ok"] for l in log) / max(att, 1),\n'
     '                   pick_place_success=sum(l["placed"] for l in log) / max(att, 1),\n'
     '                   failure_stages=dict(__import__("collections").Counter(f[0] for l in log for f in l["failures"])),'),
])
patch("tools/debug_episode.py", [
    ("pnp.q_observe = sol.q\n", "pnp.q_observe = sol.q\npnp.ik.is_valid = env.config_collision_free\npnp.safety.collision_fn = env.config_collision_free\n"),
])
