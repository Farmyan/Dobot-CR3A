from pathlib import Path
import shutil

ROOT = Path.cwd()
NEWLINES = {}
changes = {}

def read(rel):
    p = ROOT / rel
    if not p.is_file():
        raise FileNotFoundError(f"Required project file not found: {p}")
    raw = p.read_bytes()
    NEWLINES[p] = "\r\n" if b"\r\n" in raw else "\n"
    return p, raw.decode("utf-8-sig").replace("\r\n", "\n")

def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one occurrence for {label}, found {count}. "
            "No files changed."
        )
    return text.replace(old, new, 1)

p, s = read("config.py")
s = replace_once(
    s,
    "    tcp_xyz: tuple = (0.0, 0.0, 0.05)\n"
    "    tcp_rpy: tuple = (0.0, 0.0, 0.0)\n",
    "    # Link6-to-physical-AG95-TCP transform measured from simulation/robot.xml.\n"
    "    tcp_xyz: tuple = (0.00213861, -0.00708698, 0.15982866)\n"
    "    tcp_rpy: tuple = (-0.04513534, -0.01024624, -3.07159361)\n",
    "physical AG95 TCP transform"
)
s = replace_once(
    s,
    "    observe_xyz: tuple = (0.05, 0.45, 0.34)\n",
    "    observe_xyz: tuple = (0.05, 0.45, 0.22)\n"
    "    # Physical TCP is 30 mm above the cube centre for the top-down grasp.\n"
    "    tcp_to_cube_center_z: float = 0.030\n",
    "observe height and TCP-to-cube offset"
)
changes["config.py"] = (p, s)

p, s = read("control/pick_and_place.py")
s = replace_once(
    s,
    "        Tp = offset_z(place_pose(target_xy, self.ws.table_z, self.ws.cube_size, yaw=0.0), 0.002)\n",
    "        Tp = offset_z(\n"
    "            place_pose(target_xy, self.ws.table_z, self.ws.cube_size, yaw=0.0),\n"
    "            self.ws.tcp_to_cube_center_z + 0.002,\n"
    "        )\n",
    "place TCP height"
)
s = replace_once(
    s,
    "        for Tg in grasp_candidates(det.position, det.yaw):\n",
    "        # Detection reports the cube centre; IK targets the physical AG95 TCP.\n"
    "        grasp_pos = np.asarray(det.position, dtype=float).copy()\n"
    "        grasp_pos[2] += self.ws.tcp_to_cube_center_z\n"
    "        for Tg in grasp_candidates(grasp_pos, det.yaw):\n",
    "pick TCP height"
)
changes["control/pick_and_place.py"] = (p, s)

p, s = read("simulation/environment.py")
s = replace_once(
    s,
    "    def _is_reachable_xy(self, p):\n",
    '    def _is_reachable_xy(self, p, purpose="pick"):\n',
    "reachability purpose parameter"
)
s = replace_once(
    s,
    "        key = (round(float(p[0]), 3), round(float(p[1]), 3))\n"
    "        if key not in self._reachability_cache:\n"
    "            self._reachability_cache[key] = bool(self.validator_fn(np.asarray(p, dtype=float)))\n"
    "        return self._reachability_cache[key]\n",
    "        key = (purpose, round(float(p[0]), 3), round(float(p[1]), 3))\n"
    "        if key not in self._reachability_cache:\n"
    "            self._reachability_cache[key] = bool(\n"
    "                self.validator_fn(np.asarray(p, dtype=float), purpose=purpose)\n"
    "            )\n"
    "        return self._reachability_cache[key]\n",
    "purpose-aware reachability cache"
)
s = replace_once(
    s,
    "    def _sample_xy(self, forbidden_pts, min_dists, max_attempts=None):\n",
    '    def _sample_xy(self, forbidden_pts, min_dists, max_attempts=None, purpose="pick"):\n',
    "sampling purpose parameter"
)
s = replace_once(
    s,
    "            if self._is_reachable_xy(cand):\n",
    "            if self._is_reachable_xy(cand, purpose=purpose):\n",
    "sampling purpose dispatch"
)
s = replace_once(
    s,
    "            p = self._sample_xy(placed_zones, [self.ws.min_zone_spacing] * len(placed_zones))\n",
    "            p = self._sample_xy(\n"
    '                placed_zones, [self.ws.min_zone_spacing] * len(placed_zones), purpose="place"\n'
    "            )\n",
    "zone placement reachability"
)
s = replace_once(
    s,
    "                p = self._sample_xy(placed_zones, [0.11] * len(placed_zones), max_attempts=80)\n",
    "                p = self._sample_xy(\n"
    '                    placed_zones, [0.11] * len(placed_zones), max_attempts=80, purpose="place"\n'
    "                )\n",
    "relaxed zone placement reachability"
)
s = replace_once(
    s,
    "                p = self._sample_xy(forbidden, min_dists, max_attempts=120)\n",
    '                p = self._sample_xy(forbidden, min_dists, max_attempts=120, purpose="pick")\n',
    "cube pick reachability"
)
changes["simulation/environment.py"] = (p, s)

p, s = read("main.py")
s = replace_once(
    s,
    "from kinematics.transformations import rpy_to_R, make_T, Rx\n",
    "from kinematics.transformations import rpy_to_R, make_T, Rx, interpolate_pose\n",
    "pose interpolation import"
)

old = (
    "    # Pre-validate spawn points against top-down grasp IK so no unreachable cube is ever spawned\n"
    "    def can_reach_xy(xy):\n"
    "        Tg = make_T(\n"
    "            Rx(np.pi),\n"
    "            [\n"
    "                xy[0],\n"
    "                xy[1],\n"
    "                WORKSPACE.table_z + WORKSPACE.cube_size / 2\n"
    "            ]\n"
    "        )\n"
    "\n"
    "        # Fast reachability test: one IK attempt, no random multi-start.\n"
    "        target_sol = ik.solve_from(Tg, sol.q)\n"
    "\n"
    "        if not target_sol.converged:\n"
    "            return False\n"
    "\n"
    "        if target_sol.sigma_min < ik.sigma_thresh:\n"
    "            return False\n"
    "\n"
    "        return env.config_collision_free(target_sol.q)\n"
)

new = (
    "    # Validate physical TCP approach/descent for cube picks and pad placements.\n"
    "    # A multi-start solver prevents one poor seed from rejecting a reachable XY.\n"
    "    spawn_ik = IKSolver(chain, n_random_starts=12, seed=17)\n"
    "    spawn_ik.is_valid = env.config_collision_free\n"
    "\n"
    '    def can_reach_xy(xy, purpose="pick"):\n'
    "        z_tcp = (WORKSPACE.table_z + WORKSPACE.cube_size / 2\n"
    "                 + WORKSPACE.tcp_to_cube_center_z)\n"
    "        Tg = make_T(Rx(np.pi), [xy[0], xy[1], z_tcp])\n"
    "        Ta = make_T(Rx(np.pi), [xy[0], xy[1], z_tcp + WORKSPACE.approach_height])\n"
    "\n"
    "        # Require a collision-free pre-grasp and a collision-free vertical descent.\n"
    "        for approach_sol in spawn_ik.solve_all(Ta, sol.q):\n"
    "            if not env.config_collision_free(approach_sol.q):\n"
    "                continue\n"
    "            q_prev = approach_sol.q\n"
    "            path_ok = True\n"
    "            for step in range(1, 9):\n"
    "                T_step = interpolate_pose(Ta, Tg, step / 8.0)\n"
    "                step_sol = spawn_ik.solve_from(T_step, q_prev)\n"
    "                if (not step_sol.converged\n"
    "                        or step_sol.sigma_min < spawn_ik.sigma_thresh\n"
    "                        or not env.config_collision_free(step_sol.q)):\n"
    "                    path_ok = False\n"
    "                    break\n"
    "                q_prev = step_sol.q\n"
    "            if path_ok:\n"
    "                return True\n"
    "        return False\n"
)

s = replace_once(s, old, new, "spawn reachability validator")
changes["main.py"] = (p, s)

# Check all backup names before changing any file.
for rel, (p, _) in changes.items():
    backup = p.with_name(p.name + ".backup_before_tcp_fix")
    if backup.exists():
        raise RuntimeError(
            f"Backup already exists: {backup}. "
            "No files changed. Rename the backup before retrying."
        )

# Write backups and patched files.
for rel, (p, content) in changes.items():
    shutil.copy2(p, p.with_name(p.name + ".backup_before_tcp_fix"))
    p.write_bytes(content.replace("\n", NEWLINES[p]).encode("utf-8-sig"))
    print(f"Updated: {rel}")
    print(f"Backup:  {p.name}.backup_before_tcp_fix")

print("\nPatch applied.")
