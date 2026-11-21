from pathlib import Path

def patch(path, pairs):
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        print(("ok      " if s.count(a) == 1 else "PROBLEM(%d) " % s.count(a)) + path + ": " + a.strip()[:60])
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8")

patch("simulation/environment.py", [
    ("    def __init__(self, scene_path, robot_cfg, ws, cam_cfg, chain, seed=0):\n",
     "    def __init__(self, scene_path, robot_cfg, ws, cam_cfg, chain, seed=0, plan_margin=0.003):\n"),
    ("        self._scratch = mujoco.MjData(self.model)\n",
     "        self._cm = mujoco.MjModel.from_xml_path(str(scene_path))   # copy used only for clearance checks\n"
     "        self._cm.geom_margin[:] = plan_margin\n"
     "        self._scratch = mujoco.MjData(self._cm)\n"),
])

NEW = '''    def config_collision_free(self, q):
        """True if joint vector q keeps a clearance of plan_margin from self/table/cube collisions
        (gripper-gripper and gripper-cube contacts are allowed)."""
        s, m = self._scratch, self._cm
        s.qpos[:] = self.data.qpos
        for a, v in zip(self.jadr, q):
            s.qpos[a] = v
        mujoco.mj_fwdPosition(m, s)
        for i in range(s.ncon):
            c = s.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            if not (r1 or r2):
                continue                                   # cube-table, cube-cube
            g1, g2 = b1 in self.gripper_bodies, b2 in self.gripper_bodies
            if b1 in self.cube_bodies or b2 in self.cube_bodies:
                if g1 or g2:
                    continue                               # gripper may touch the cube
                return False                               # arm hitting a cube
            if g1 and g2:
                continue
            if r1 and r2:
                return False                               # link-link
            if self.table_geom in (c.geom1, c.geom2) and self.base_body not in (b1, b2):
                return False
        return True

'''
p = Path("simulation/environment.py")
s = p.read_text(encoding="utf-8")
i = s.index("    def config_collision_free(self, q):\n")
j = s.index("    # ---- sensing ----\n")
if i < j:
    p.write_text(s[:i] + NEW + s[j:], encoding="utf-8")
    print("ok      simulation/environment.py: config_collision_free replaced")
else:
    print("PROBLEM simulation/environment.py: could not locate config_collision_free")

patch("main.py", [
    ("def run_sim(n_episodes, viewer, seed):\n", "def run_sim(n_episodes, viewer, seed, margin=0.003):\n"),
    ('    env = SimEnvironment(ROOT / "simulation/scene.xml", ROBOT, WORKSPACE, CAMERA, chain, seed)\n',
     '    env = SimEnvironment(ROOT / "simulation/scene.xml", ROBOT, WORKSPACE, CAMERA, chain, seed, plan_margin=margin)\n'),
    ("    env.reset(sol.q)\n",
     "    env.reset(sol.q)\n"
     "    print('clearance margin %.1f mm; observe pose passes the check: %s' % (margin * 1000, env.config_collision_free(sol.q)))\n"),
    ('    ap.add_argument("--seed", type=int, default=0)\n',
     '    ap.add_argument("--seed", type=int, default=0)\n    ap.add_argument("--margin", type=float, default=0.003, help="planning clearance in metres")\n'),
    ('run_sim(a.episodes, a.viewer, a.seed) if', 'run_sim(a.episodes, a.viewer, a.seed, a.margin) if'),
])
