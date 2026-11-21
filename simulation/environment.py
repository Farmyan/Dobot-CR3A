"""MuJoCo environment: random per-color cube counts + random colored floor zones."""
import numpy as np
import mujoco
from config import CUBE_COLORS
from perception.camera import Camera
from perception.calibration import Intrinsics
from kinematics.transformations import make_T, inv_T


AG95_WIDTH_CALIBRATION = np.array([
    (0.000000, 0.095978),
    (0.020387, 0.093609),
    (0.040775, 0.091169),
    (0.061162, 0.088660),
    (0.081550, 0.086084),
    (0.101938, 0.083445),
    (0.122325, 0.080743),
    (0.142712, 0.077983),
    (0.163100, 0.075166),
    (0.183487, 0.072294),
    (0.203875, 0.069372),
    (0.224263, 0.066401),
    (0.244650, 0.063384),
    (0.265037, 0.060323),
    (0.285425, 0.057223),
    (0.305812, 0.054085),
    (0.326200, 0.050913),
    (0.346587, 0.047709),
    (0.366975, 0.044477),
    (0.387362, 0.041219),
    (0.407750, 0.037939),
    (0.428138, 0.034639),
    (0.448525, 0.031323),
    (0.468913, 0.027994),
    (0.489300, 0.024654),
    (0.509687, 0.021307),
    (0.530075, 0.017957),
    (0.550462, 0.014605),
    (0.570850, 0.011255),
    (0.591237, 0.007911),
    (0.611625, 0.004575),
    (0.632012, 0.001251),
    (0.652400, -0.002058),
], dtype=float)

AG95_MIMICS = {
    "ag95_finger2_joint": 1.0,
    "ag95_finger1_finger_joint": 0.4563942,
    "ag95_finger2_finger_joint": 0.4563942,
    "ag95_finger1_inner_joint": 1.49462955,
    "ag95_finger2_inner_joint": 1.49462955,
    "ag95_finger1_tip_joint": 1.49462955,
    "ag95_finger2_tip_joint": 1.49462955,
}



class SimEnvironment:
    def __init__(self, scene_path, robot_cfg, ws, cam_cfg, chain, seed=0, plan_margin=0.0025):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.cfg, self.ws, self.cam_cfg, self.chain = robot_cfg, ws, cam_cfg, chain
        self.rng = np.random.default_rng(seed)
        m = self.model
        self.jadr = [m.joint(n).qposadr[0] for n in chain.names]
        self.arm_act = [m.actuator(f"act_{n}").id for n in chain.names]
        self.grip_act = m.actuator("gripper").id
        self.grip_q = [m.joint("ag95_finger1_joint").qposadr[0]]
        self.grip_geom_ids = {}
        for finger_body in ("ag95_finger1_tip", "ag95_finger2_tip"):
            body_id = m.body(finger_body).id
            geom_ids = [
                i for i in range(m.ngeom)
                if int(m.geom_bodyid[i]) == body_id
                and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
            ]
            if len(geom_ids) != 1:
                raise RuntimeError(
                    f"Expected one distal mesh on {finger_body}, found {geom_ids}"
                )
            self.grip_geom_ids[finger_body] = geom_ids[0]

        self.cubes = sorted(
            (n for n in (m.body(i).name for i in range(m.nbody)) if n.startswith("cube_")),
            key=lambda s: int(s.split("_")[1])
        )
        self.cube_adr = [m.joint(f"{c}_free").qposadr[0] for c in self.cubes]
        self.cube_bodies = [m.body(c).id for c in self.cubes]
        self.cube_geoms = [m.geom(f"{c}_geom").id for c in self.cubes]
        self.cube_colors = ["red"] * len(self.cubes)

        self.zones = sorted(n for n in (m.body(i).name for i in range(m.nbody)) if n.startswith("zone_"))
        self.zone_mocap = [int(m.body_mocapid[m.body(z).id]) for z in self.zones]
        self.zone_color_names = [z.replace("zone_", "") for z in self.zones]

        self.base_body = m.body(robot_cfg.base_link).id
        self.cam_id = m.camera(cam_cfg.name).id
        self.table_geom = m.geom("table_geom").id
        self.robot_bodies = {i for i in range(m.nbody) if m.body_rootid[i] == self.base_body}
        self.gripper_bodies = {
            i for i in self.robot_bodies
            if m.body(i).name.startswith(
                ("gripper", "bracket", "picker", "arm_", "arm2_", "jaw_", "ag95")
            )
        }
        self.renderer = mujoco.Renderer(m, cam_cfg.height, cam_cfg.width)
        self.dt = m.opt.timestep
        self._cm = mujoco.MjModel.from_xml_path(str(scene_path))
        self._cm.geom_margin[:] = plan_margin
        self._scratch = mujoco.MjData(self._cm)
        fl = m.body(robot_cfg.flange_link).id
        self.wrist_bodies = {fl, int(m.body_parentid[fl])}
        self.unsafe_contacts = 0
        self.joint_violations = 0
        self.unsafe_pairs = {}
        self.phase = ""
        self.validator_fn = None   # optional callable(xy) -> bool to verify IK reachability
        self._reachability_cache = {}
        self.sample_attempts = 160
        self.last_spawn_counts = {}

    def _is_reachable_xy(self, p, purpose="pick"):
        r = float(np.linalg.norm(p))
        if not (self.ws.min_r <= r <= self.ws.max_r):
            return False
        if self.validator_fn is None:
            return True

        # Sampling is quantized to 5 mm, so cache expensive IK checks for repeated cells.
        # The current pick/place validator uses identical geometry.
        # Share results across purposes to avoid duplicate IK checks.
        key = (round(float(p[0]), 3), round(float(p[1]), 3))
        if key not in self._reachability_cache:
            self._reachability_cache[key] = bool(
                self.validator_fn(np.asarray(p, dtype=float), purpose=purpose)
            )
        return self._reachability_cache[key]

    def _sample_xy(self, forbidden_pts, min_dists, max_attempts=None, purpose="pick"):
        """Bounded rejection sampler.

        Check cheap spacing constraints before invoking IK. Candidates are snapped to
        a 5 mm grid to avoid repeatedly testing almost-identical random positions.
        Returns None when the requested clearances cannot be satisfied.
        """
        attempts = self.sample_attempts if max_attempts is None else int(max_attempts)
        for _ in range(max(1, attempts)):
            cand = np.array([
                self.rng.uniform(*self.ws.spawn_x),
                self.rng.uniform(*self.ws.spawn_y),
            ])
            cand = np.round(cand / 0.005) * 0.005

            if any(np.linalg.norm(cand - o) < md for o, md in zip(forbidden_pts, min_dists)):
                continue
            if self._is_reachable_xy(cand, purpose=purpose):
                return cand
        return None

    def _set_cube_color(self, idx, color_name):
        self.cube_colors[idx] = color_name
        rgba = CUBE_COLORS[color_name]
        self.model.geom_rgba[self.cube_geoms[idx]] = rgba
        self._cm.geom_rgba[self.cube_geoms[idx]] = rgba

    def randomize_zones(self, active_colors):
        """Place active zones using individually validated positions."""
        zone_map = {}
        selected = []

        for color_name in active_colors:
            point = self._sample_xy(
                selected,
                [self.ws.min_zone_spacing] * len(selected),
                max_attempts=self.sample_attempts,
                purpose="place",
            )
            if point is None:
                raise RuntimeError(
                    f"[Spawner] Could not find a reachable position for zone {color_name}."
                )
            selected.append(point)
            zone_map[color_name] = point

        for color_name, mid in zip(self.zone_color_names, self.zone_mocap):
            if color_name in zone_map:
                point = zone_map[color_name]
                self.data.mocap_pos[mid] = [point[0], point[1], 0.0006]
                self.data.mocap_quat[mid] = [1.0, 0.0, 0.0, 0.0]
            else:
                self.data.mocap_pos[mid] = [6.0, 6.0, -0.05]

        mujoco.mj_forward(self.model, self.data)
        return zone_map

    def get_zone_positions(self):
        out = {}
        for color_name, mid in zip(self.zone_color_names, self.zone_mocap):
            pos = self.data.mocap_pos[mid]
            if pos[0] < 2.0:
                out[color_name] = pos[:2].copy()
        return out

    def clear_cubes(self):
        d = self.data
        for k, adr in enumerate(self.cube_adr):
            d.qpos[adr:adr + 7] = [
                5.0 + k, 5.0, self.ws.cube_size / 2, 1, 0, 0, 0
            ]
            j_dof = self.model.joint(f"{self.cubes[k]}_free").dofadr[0]
            d.qvel[j_dof:j_dof + 6] = 0.0

        for mid in self.zone_mocap:
            d.mocap_pos[mid] = [6.0, 6.0, -0.05]

        mujoco.mj_forward(self.model, d)

    def spawn_random_batch(self):
        """Generate exactly 15 cubes and four reachable color zones."""
        self._reachability_cache.clear()

        palette = ["red", "yellow", "green", "blue"]
        counts = {"red": 4, "yellow": 4, "green": 3, "blue": 4}

        assert set(palette) == set(CUBE_COLORS)
        assert sum(counts.values()) == 15
        assert len(self.cubes) >= 15

        self.clear_cubes()

        print("[Spawner] Finding four reachable zone positions...", flush=True)

        zone_points = []
        for color_name in palette:
            point = self._sample_xy(
                zone_points,
                [self.ws.min_zone_spacing] * len(zone_points),
                max_attempts=self.sample_attempts,
                purpose="place",
            )
            if point is None:
                raise RuntimeError(
                    f"[Spawner] Could not place zone {color_name} after "
                    f"{self.sample_attempts} candidate attempts."
                )
            zone_points.append(point)
            print(
                f"[Spawner] Zone {color_name}: "
                f"({point[0]:.3f}, {point[1]:.3f})",
                flush=True,
            )

        print("[Spawner] Finding 15 reachable cube positions...", flush=True)

        print("[Spawner] Building randomized geometric candidate pool...", flush=True)

        # Generate the same 5 mm grid used by the original sampler.
        xs = np.round(
            np.arange(self.ws.spawn_x[0], self.ws.spawn_x[1] + 0.0025, 0.005) / 0.005
        ) * 0.005
        ys = np.round(
            np.arange(self.ws.spawn_y[0], self.ws.spawn_y[1] + 0.0025, 0.005) / 0.005
        ) * 0.005

        candidates = np.array([
            (x, y)
            for x in xs
            for y in ys
            if self.ws.min_r <= np.hypot(x, y) <= self.ws.max_r
            and all(
                np.hypot(x - zone[0], y - zone[1]) >= 0.105
                for zone in zone_points
            )
        ], dtype=float)

        if len(candidates) < 15:
            raise RuntimeError(
                f"[Spawner] Only {len(candidates)} geometric cube positions "
                "satisfy the workspace and zone-clearance constraints."
            )

        print(
            f"[Spawner] Geometric candidate pool: {len(candidates)} positions.",
            flush=True,
        )

        # Try randomized complete geometric layouts. Expensive IK results
        # remain cached across attempts.
        max_layout_attempts = 100
        cube_points = []
        last_failed_cube = 0

        for layout_try in range(1, max_layout_attempts + 1):
            selected = []

            for candidate_index in self.rng.permutation(len(candidates)):
                point = candidates[candidate_index]

                if any(
                    np.linalg.norm(point - other) < self.ws.min_cube_spacing
                    for other in selected
                ):
                    continue

                selected.append(point.copy())

                if len(selected) == 15:
                    break

            if len(selected) != 15:
                continue

            print(
                f"[Spawner] Validating randomized layout "
                f"{layout_try}/{max_layout_attempts}...",
                flush=True,
            )

            layout_ok = True

            for cube_index, point in enumerate(selected):
                if not self._is_reachable_xy(point, purpose="pick"):
                    last_failed_cube = cube_index + 1
                    layout_ok = False
                    print(
                        f"[Spawner] Candidate failed production validation "
                        f"at cube {last_failed_cube}/15. Trying another layout.",
                        flush=True,
                    )
                    break

            if layout_ok:
                cube_points = selected
                break

        if len(cube_points) != 15:
            raise RuntimeError(
                f"[Spawner] Could not generate 15 fully validated cubes "
                f"after {max_layout_attempts} randomized layouts. "
                f"Last failed cube: {last_failed_cube}/15. "
                "No partial layout was committed."
            )

        print(
            "[Spawner] All 15 cube positions passed production validation.",
            flush=True,
        )

        # Commit the complete layout only after all 19 positions pass validation.
        zone_map = {}
        zone_iter = iter(zone_points)

        for color_name, mid in zip(self.zone_color_names, self.zone_mocap):
            if color_name in palette:
                point = next(zone_iter)
                zone_map[color_name] = point
                self.data.mocap_pos[mid] = [point[0], point[1], 0.0006]
                self.data.mocap_quat[mid] = [1.0, 0.0, 0.0, 0.0]
            else:
                self.data.mocap_pos[mid] = [6.0, 6.0, -0.05]

        d = self.data
        idx = 0
        actual_counts = {color: 0 for color in palette}

        for color_name, count in counts.items():
            for _ in range(count):
                point = cube_points[idx]
                self._set_cube_color(idx, color_name)

                yaw = float(self.rng.uniform(-np.pi, np.pi))
                adr = self.cube_adr[idx]
                d.qpos[adr:adr + 7] = [
                    point[0],
                    point[1],
                    self.ws.cube_size / 2 + 0.002,
                    np.cos(yaw / 2),
                    0,
                    0,
                    np.sin(yaw / 2),
                ]

                idx += 1
                actual_counts[color_name] += 1

        mujoco.mj_forward(self.model, d)

        self.last_spawn_counts = actual_counts.copy()
        print(
            f"[Spawner] Successfully generated {idx} cubes: {actual_counts}",
            flush=True,
        )
        return actual_counts

    # ---- scene ----
    def reset(self, q_home):
        d = self.data
        mujoco.mj_resetData(self.model, d)
        for a, v in zip(self.jadr, q_home):
            d.qpos[a] = v
        d.ctrl[self.arm_act] = q_home
        open_angle = self.gripper_angle_for_width(0.055)
        d.ctrl[self.grip_act] = open_angle
        d.qpos[self.grip_q[0]] = open_angle

        for joint_name, multiplier in AG95_MIMICS.items():
            joint = self.model.joint(joint_name)
            d.qpos[joint.qposadr[0]] = open_angle * multiplier

        mujoco.mj_forward(self.model, d)
        # Return the generated counts to the browser-control loop.  Without this,
        # clicking Generate makes main.py call counts.items() on None and exits.
        counts = self.spawn_random_batch()
        self.unsafe_contacts = self.joint_violations = 0
        self.unsafe_pairs = {}
        return counts

    def settle(self, seconds):
        for _ in range(max(1, int(seconds / self.dt))):
            mujoco.mj_step(self.model, self.data)

    # ---- stepping ----
    def step_to(self, q_cmd, duration):
        d = self.data
        d.ctrl[self.arm_act] = q_cmd
        for _ in range(max(1, int(round(duration / self.dt)))):
            mujoco.mj_step(self.model, d)
            self._monitor()

    def _monitor(self):
        d, m = self.data, self.model
        q = d.qpos[self.jadr]
        if np.any(q < self.chain.lower - 1e-3) or np.any(q > self.chain.upper + 1e-3):
            self.joint_violations += 1
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 in self.robot_bodies and b2 in self.robot_bodies:
                self.unsafe_contacts += 1
                self._note("self", c)
                continue
            for ba, bb, gb in ((b1, b2, c.geom2), (b2, b1, c.geom1)):
                if ba in self.robot_bodies:
                    other_cube = bb in self.cube_bodies
                    other_table = gb == self.table_geom or (bb == 0 and gb == self.table_geom)
                    non_gripper = ba not in self.gripper_bodies
                    if other_table or (other_cube and non_gripper):
                        self.unsafe_contacts += 1
                        self._note("table" if other_table else "cube", c)

    def _note(self, kind, c):
        m = self.model
        g = lambda i: f"{m.body(m.geom_bodyid[i]).name}/{m.geom(i).name or 'mesh'}"
        key = f"[{self.phase}] {kind}: {g(c.geom1)} <-> {g(c.geom2)}"
        self.unsafe_pairs[key] = self.unsafe_pairs.get(key, 0) + 1

    def config_collision_free(self, q):
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
                continue
            g1, g2 = b1 in self.gripper_bodies, b2 in self.gripper_bodies
            if b1 in self.cube_bodies or b2 in self.cube_bodies:
                if g1 or g2 or b1 in self._held_cubes() or b2 in self._held_cubes():
                    continue
                return False
            if g1 and g2:
                continue
            if (g1 and b2 in self.wrist_bodies) or (g2 and b1 in self.wrist_bodies):
                continue
            if (b1 == self.base_body and m.body_parentid[b2] == b1) or (b2 == self.base_body and m.body_parentid[b1] == b2):
                continue
            if r1 and r2:
                return False
            if self.table_geom in (c.geom1, c.geom2) and self.base_body not in (b1, b2):
                # The gripper may approach the table within the planning
                # margin, but actual geometric penetration remains forbidden.
                if (g1 or g2) and c.dist >= -1e-6:
                    continue
                return False
        return True

    def _held_cubes(self):
        d, m = self.data, self.model
        out = set()
        for i in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
            if b1 in self.cube_bodies and b2 in self.gripper_bodies:
                out.add(b1)
            elif b2 in self.cube_bodies and b1 in self.gripper_bodies:
                out.add(b2)
        return out

    def explain_collision(self, q):
        s, m = self._scratch, self._cm
        s.qpos[:] = self.data.qpos
        for a, v in zip(self.jadr, q):
            s.qpos[a] = v
        mujoco.mj_fwdPosition(m, s)
        g = lambda i: f"{m.body(m.geom_bodyid[i]).name}/{m.geom(i).name or 'mesh'}"
        return sorted({f"{g(s.contact[i].geom1)} <-> {g(s.contact[i].geom2)} dist={s.contact[i].dist*1000:.1f}mm"
                       for i in range(s.ncon)})

    # ---- sensing ----
    def joint_positions(self):
        return self.data.qpos[self.jadr].copy()

    def gripper_angle_for_width(self, width):
        """Convert requested physical gap (meters) to AG95 drive angle."""
        angles = AG95_WIDTH_CALIBRATION[:, 0]
        gaps = AG95_WIDTH_CALIBRATION[:, 1]

        # Reverse the descending gap column for np.interp.
        # Zero is the minimum requested physical gap; avoid mesh overlap.
        target = float(np.clip(width, 0.0, gaps[0]))
        return float(np.interp(target, gaps[::-1], angles[::-1]))

    def gripper_width(self):
        """Return current minimum distance between the two AG95 distal meshes."""
        mujoco.mj_forward(self.model, self.data)
        fromto = np.zeros(6, dtype=float)
        distance = mujoco.mj_geomDistance(
            self.model,
            self.data,
            self.grip_geom_ids["ag95_finger1_tip"],
            self.grip_geom_ids["ag95_finger2_tip"],
            1.0,
            fromto,
        )
        return float(max(0.0, distance))

    def T_world_base(self):
        d = self.data
        return make_T(d.xmat[self.base_body].reshape(3, 3), d.xpos[self.base_body])

    def tcp_pose_mujoco(self):
        d, sid = self.data, self.model.site("tcp").id
        return inv_T(self.T_world_base()) @ make_T(d.site_xmat[sid].reshape(3, 3), d.site_xpos[sid])

    def ground_truth_cubes(self):
        out = []
        for idx, (c, b) in enumerate(zip(self.cubes, self.cube_bodies)):
            T = inv_T(self.T_world_base()) @ make_T(self.data.xmat[b].reshape(3, 3), self.data.xpos[b])
            out.append((c, T[:3, 3].copy(), self.cube_colors[idx]))
        return out


class SimCamera(Camera):
    def __init__(self, env: SimEnvironment):
        self.env, c = env, env.cam_cfg
        self._intr = Intrinsics.from_fovy(float(env.model.cam_fovy[env.cam_id]), c.width, c.height)

    def capture(self):
        r = self.env.renderer
        r.update_scene(self.env.data, camera=self.env.cam_cfg.name)
        return r.render().copy()

    @property
    def intrinsics(self):
        return self._intr

    def T_base_cam(self):
        e = self.env
        T_world_cam_mj = make_T(e.data.cam_xmat[e.cam_id].reshape(3, 3), e.data.cam_xpos[e.cam_id])
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        return inv_T(e.T_world_base()) @ T_world_cam_mj @ flip
