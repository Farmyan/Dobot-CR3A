"""High-level autonomous pick-and-place with spaced 2x2 sub-slots per colored floor pad."""
from dataclasses import dataclass
import numpy as np
from control.trajectory_planning import plan_joint, plan_cartesian_line, plan_joint_rrt, PlanningError
from control.grasp_planning import grasp_candidates, offset_z, place_pose
from kinematics.transformations import Rx, make_T

# +/-32 mm sub-slots on a 120x120 mm floor pad => 64 mm center-to-center (39 mm clear gap between 25 mm cubes)
PAD_SUB_OFFSETS = [
    (-0.032, -0.032),
    ( 0.032, -0.032),
    (-0.032,  0.032),
    ( 0.032,  0.032),
]


@dataclass
class PickResult:
    success: bool
    stage: str = "done"
    detail: str = ""
    cube_estimate: np.ndarray | None = None
    color: str = ""


class PickAndPlace:
    def __init__(self, robot, camera, detector, chain, ik, safety, ws, grip, vmax, amax):
        self.robot, self.camera, self.det = robot, camera, detector
        self.chain, self.ik, self.safety = chain, ik, safety
        self.ws, self.grip, self.vmax, self.amax = ws, grip, vmax, amax
        self.q_observe = None
        self.max_candidates = 12

    def _validate(self, traj, stage, allow_low=False):
        rep = self.safety.validate(traj, allow_low_tcp=allow_low)
        if not rep.ok:
            raise PlanningError(stage, "; ".join(rep.reasons))

    def _run(self, traj, stage, allow_low=False):
        self.robot.stage = stage

        # Temporary diagnostics: measure commanded joint movement without changing it.
        q_actual = np.asarray(self.robot.get_joint_positions(), dtype=float)
        q_path = np.asarray(traj.q, dtype=float)
        if q_path.ndim == 2 and len(q_path) > 0:
            full_path = np.vstack((q_actual, q_path))
            increments = np.diff(full_path, axis=0)
            net_deg = np.degrees(q_path[-1] - q_actual)
            total_deg = np.degrees(np.sum(np.abs(increments), axis=0))
            max_step_deg = np.degrees(np.max(np.abs(increments), axis=0))
            fmt = lambda values: np.array2string(values, precision=1, suppress_small=True)
            print(
                f"[MOTION DIAG] {stage} | net={fmt(net_deg)} deg | "
                f"path_total={fmt(total_deg)} deg | max_step={fmt(max_step_deg)} deg",
                flush=True,
            )

        self._validate(traj, stage, allow_low)
        if not self.robot.execute_trajectory(traj):
            raise PlanningError(stage, "robot reported error: " + str(self.robot.get_errors()))

    def _plan_movement_cost(self, plan, q_start):
        """Measure complete planned joint movement, including segment transitions."""
        stages = ("app", "des", "lift", "tr", "pd", "rt")
        if any(plan.get(stage) is None for stage in stages):
            return None

        total = np.zeros(6, dtype=float)
        previous = np.asarray(q_start, dtype=float)

        for stage in stages:
            q = np.asarray(plan[stage].q, dtype=float)
            if q.ndim != 2 or len(q) == 0:
                return None

            full_path = np.vstack((previous, q))
            total += np.sum(np.abs(np.diff(full_path, axis=0)), axis=0)
            previous = q[-1]

        return np.degrees(total)
    def _holding(self):
        w = self.robot.get_gripper_width()
        return self.grip.min_grasp_width <= w <= self.grip.max_grasp_width

    def _line(self, T0, T1, q_seed):
        return plan_cartesian_line(self.ik, T0, T1, q_seed, self.vmax, self.amax)

    def _joint_path(self, q0, q1, stage):
        delta_deg = np.degrees(q1 - q0)
        print(
            f"[PATH DIAG] {stage} | start={np.round(np.degrees(q0), 1)} deg | "
            f"end={np.round(np.degrees(q1), 1)} deg | delta={np.round(delta_deg, 1)} deg",
            flush=True,
        )
        direct = plan_joint(q0, q1, self.vmax, self.amax)
        try:
            self._validate(direct, stage)
            print(f"[PATH DIAG] {stage} | planner=DIRECT", flush=True)
            return direct
        except PlanningError as exc:
            print(f"[PATH DIAG] {stage} | DIRECT rejected: {exc}; trying RRT", flush=True)
        rrt_lower = self.chain.lower.copy()
        rrt_upper = self.chain.upper.copy()

        if getattr(self.ik, "position_only", False):
            fixed = self.ik.fixed_q
            active = self.ik.active_dofs
            rrt_lower[active:] = fixed[active:]
            rrt_upper[active:] = fixed[active:]

        tr = plan_joint_rrt(
            q0, q1, self.safety.config_ok,
            rrt_lower, rrt_upper,
            self.vmax, self.amax,
        )
        self._validate(tr, stage)
        print(f"[PATH DIAG] {stage} | planner=RRT | waypoints={len(tr.q)}", flush=True)
        return tr

    def _nearest_equivalent(self, q, q_ref):
        q = np.asarray(q, dtype=float).copy()
        q_ref = np.asarray(q_ref, dtype=float)
        two_pi = 2.0 * np.pi

        for i, joint in enumerate(self.chain.actuated):
            if joint.type not in ("revolute", "continuous"):
                continue

            lo = float(self.chain.lower[i])
            hi = float(self.chain.upper[i])
            base = float(q[i])

            k_min = int(np.ceil((lo - base) / two_pi - 1e-10))
            k_max = int(np.floor((hi - base) / two_pi + 1e-10))

            if k_min <= k_max:
                equivalents = (
                    base + two_pi * k
                    for k in range(k_min, k_max + 1)
                )
                q[i] = min(equivalents, key=lambda value: abs(value - q_ref[i]))

        return q

    def compute_observe_pose(self, q_seed):
        x, y, z = self.ws.observe_xyz
        sol = self.ik.solve(make_T(Rx(np.pi), [x, y, z]), q_seed)
        if sol is None:
            raise PlanningError("observe", "observe pose unreachable; edit WORKSPACE.observe_xyz")
        self.q_observe = self._nearest_equivalent(sol.q, q_seed)
        return self.q_observe

    def go_observe(self):
        q0 = self.robot.get_joint_positions()
        self._run(self._joint_path(q0, self.q_observe, "go_observe"), "go_observe")

    def detect_scene(self):
        rgb = self.camera.capture()
        return self.det.detect_all(rgb, self.camera.intrinsics, self.camera.T_base_cam())

    def detect(self):
        cubes, _ = self.detect_scene()
        return cubes

    def _ik_candidates(self, T, q_ref):
        ok = self.ik.is_valid or (lambda q: True)
        yielded = []

        # Try the seed-near solution first.
        fast = self.ik.solve_from(T, q_ref)
        if fast.converged and fast.sigma_min >= self.ik.sigma_thresh:
            q = self._nearest_equivalent(fast.q, q_ref)
            if ok(q):
                yielded.append(q)
                yield q

        # Defer multi-start IK until another candidate is requested.
        candidates = []
        for sol in self.ik.solve_all(T, q_ref):
            q = self._nearest_equivalent(sol.q, q_ref)
            if not ok(q):
                continue
            if any(np.max(np.abs(q - existing)) <= 1e-2
                   for existing in yielded + candidates):
                continue
            candidates.append(q)

        candidates.sort(key=lambda q: np.linalg.norm(q - q_ref))
        for q in candidates:
            yield q

    def _plan_place(self, target_xy, q_start, yaw=0.0):
        Tp = offset_z(
            place_pose(target_xy, self.ws.table_z, self.ws.cube_size, yaw=yaw),
            self.ws.tcp_to_cube_center_z + 0.002,
        )
        Tpa = offset_z(Tp, self.ws.approach_height)
        why = "no IK solution for the place pose"
        place_candidates = list(self._ik_candidates(Tpa, q_start))
        for index, candidate in enumerate(place_candidates[:6]):
            delta = np.degrees(candidate - q_start)
            print(
                f"[PLACE IK DIAG] candidate={index} "
                f"delta={np.round(delta, 1)} deg "
                f"distance={np.linalg.norm(delta):.1f} deg",
                flush=True,
            )
        selected_plan = None
        selected_cost = np.inf
        for k, qpa in enumerate(place_candidates):
            if k >= self.max_candidates:
                break
            try:
                t_tr = self._joint_path(q_start, qpa, "transport")
                t_pd = self._line(Tpa, Tp, qpa)
                self._validate(t_pd, "place_descend", True)
                t_rt = self._line(Tp, Tpa, t_pd.q[-1])
                self._validate(t_rt, "retreat", True)

                paths = (
                    ("transport", np.vstack((np.asarray(q_start, dtype=float), t_tr.q))),
                    ("descend", np.vstack((t_tr.q[-1], t_pd.q))),
                    ("retreat", np.vstack((t_pd.q[-1], t_rt.q))),
                )
                costs = {}
                for name, path in paths:
                    costs[name] = np.degrees(
                        np.sum(np.abs(np.diff(path, axis=0)), axis=0)
                    )

                total_cost = sum(costs.values())
                print(
                    f"[PLACE FEASIBILITY] candidate={k} "
                    f"transport={costs['transport'].sum():.1f} deg "
                    f"descend={costs['descend'].sum():.1f} deg "
                    f"retreat={costs['retreat'].sum():.1f} deg "
                    f"total={total_cost.sum():.1f} deg "
                    f"joints={np.round(total_cost, 1)}",
                    flush=True,
                )

                candidate_cost = float(total_cost.sum())
                if candidate_cost < selected_cost:
                    selected_cost = candidate_cost
                    selected_plan = (t_tr, t_pd, t_rt)

            except PlanningError as e:
                why = str(e)
                print(
                    f"[PLACE FEASIBILITY] candidate={k} REJECTED: {why}",
                    flush=True,
                )

        if selected_plan is not None:
            return selected_plan
        raise PlanningError("place_plan", why)

    def _plan_pick(self, det, target_xy):
        target_body_fn = getattr(self, "target_body_fn", None)
        target_collision_fn = getattr(self, "target_collision_fn", None)

        # Real-robot controllers have no MuJoCo target callbacks.
        if not callable(target_body_fn) or not callable(target_collision_fn):
            return self._plan_pick_impl(det, target_xy)

        try:
            target_body = target_body_fn(det.position)
        except ValueError as exc:
            raise PlanningError("target_identification", str(exc)) from exc

        original_ik_valid = self.ik.is_valid
        original_collision_fn = self.safety.collision_fn

        target_check = lambda q: target_collision_fn(q, target_body)

        try:
            self.ik.is_valid = target_check
            self.safety.collision_fn = target_check

            return self._plan_pick_impl(
                det,
                target_xy,
                original_ik_valid,
                original_collision_fn,
                target_check,
            )
        finally:
            self.ik.is_valid = original_ik_valid
            self.safety.collision_fn = original_collision_fn

    def _plan_pick_impl(
        self,
        det,
        target_xy,
        original_ik_valid=None,
        original_collision_fn=None,
        target_check=None,
    ):
        q_now = self.robot.get_joint_positions()
        why, tried, fails = "no collision-free IK solution for any grasp orientation", 0, {}
        # Detection reports the cube centre; IK targets the physical AG95 TCP.
        grasp_pos = np.asarray(det.position, dtype=float).copy()
        grasp_pos[2] += self.ws.tcp_to_cube_center_z
        for candidate_index, Tg in enumerate(grasp_candidates(grasp_pos, det.yaw)):
            candidate_yaw = np.degrees(np.arctan2(Tg[1, 0], Tg[0, 0]))

            Ta = offset_z(Tg, self.ws.approach_height)
            Tl = offset_z(Tg, self.ws.lift_height)
            for qg in self._ik_candidates(Tg, q_now):
                tried += 1
                if tried > self.max_candidates:
                    break
                try:
                    qa = next(iter(self._ik_candidates(Ta, qg)), None)
                    if qa is None:
                        why = "no approach IK near the grasp configuration"
                        continue
                    t_app = self._joint_path(q_now, qa, "approach")
                    t_des = self._line(Ta, Tg, qa)
                    self._validate(t_des, "descend", True)
                    t_lift = self._line(Tg, Tl, t_des.q[-1])
                    self._validate(t_lift, "lift", True)
                    # Placement planning uses the ordinary collision checker.
                    if target_check is not None:
                        self.ik.is_valid = original_ik_valid
                        self.safety.collision_fn = original_collision_fn

                    try:
                        grasp_yaw = float(np.arctan2(Tg[1, 0], Tg[0, 0]))
                        t_tr, t_pd, t_rt = self._plan_place(
                            target_xy, t_lift.q[-1], yaw=grasp_yaw
                        )
                    finally:
                        # Restore target-aware checking for the remaining pick planning.
                        if target_check is not None:
                            self.ik.is_valid = target_check
                            self.safety.collision_fn = target_check
                    t_back = self._line(Tg, Ta, t_des.q[-1])
                    print(
                        f"[GRASP ORIENTATION] candidate={candidate_index} "
                        f"yaw={candidate_yaw:.2f} deg "
                        f"detected_yaw={np.degrees(det.yaw):.2f} deg",
                        flush=True,
                    )
                    plan = dict(app=t_app, des=t_des, lift=t_lift, tr=t_tr, pd=t_pd, rt=t_rt, back=t_back)
                    movement = self._plan_movement_cost(plan, q_now)
                    if movement is not None:
                        fmt = lambda values: np.array2string(values, precision=1, suppress_small=True)
                        print(
                            f"[PLAN COST] joints={fmt(movement)} deg | "
                            f"total={movement.sum():.1f} deg",
                            flush=True,
                        )
                    return plan
                except PlanningError as e:
                    why = str(e)
                    fails[e.stage] = fails.get(e.stage, 0) + 1
            if tried > self.max_candidates:
                break
        raise PlanningError("grasp_plan", f"{tried} candidates tried {fails}; last: {why}")

    def pick_and_place_one(self, det, target_xy):
        try:
            try:
                plan = self._plan_pick(det, target_xy)
            except PlanningError:
                self.go_observe()
                plan = self._plan_pick(det, target_xy)
        except PlanningError as e:
            return PickResult(False, e.stage, str(e), det.position, det.color)

        # Keep target-aware collision checking active during physical picking.
        target_body_fn = getattr(self, "target_body_fn", None)
        target_collision_fn = getattr(self, "target_collision_fn", None)

        original_ik_valid = self.ik.is_valid
        original_collision_fn = self.safety.collision_fn
        target_check = None

        if callable(target_body_fn) and callable(target_collision_fn):
            try:
                target_body = target_body_fn(det.position)
            except ValueError as exc:
                return PickResult(
                    False, "target_identification", str(exc),
                    det.position, det.color
                )

            target_check = lambda q: target_collision_fn(q, target_body)
            self.ik.is_valid = target_check
            self.safety.collision_fn = target_check

        stage = "approach"
        try:
            self.robot.set_gripper(self.grip.open_width)
            self._run(plan["app"], stage)

            stage = "descend"
            self._run(plan["des"], stage, True)

            stage = "grasp"
            self.robot.set_gripper(self.grip.close_width)
            w = self.robot.get_gripper_width()

            if not (self.grip.min_grasp_width <= w <= self.grip.max_grasp_width):
                self.robot.set_gripper(self.grip.open_width)
                self._run(plan["back"], "retreat_after_miss", True)
                return PickResult(
                    False, "grasp",
                    f"gripper width {w*1000:.1f} mm (miss)",
                    det.position, det.color
                )

            stage = "lift"
            self._run(plan["lift"], stage, True)

            if not self._holding():
                return PickResult(
                    False, "lift", "cube lost during lift",
                    det.position, det.color
                )

            # The target cube is now held. Use ordinary collision checking
            # for transport and placement.
            if target_check is not None:
                self.ik.is_valid = original_ik_valid
                self.safety.collision_fn = original_collision_fn

            stage = "transport"
            self._run(plan["tr"], stage)

            if not self._holding():
                return PickResult(
                    False, "transport", "cube dropped during transport",
                    det.position, det.color
                )

            stage = "place_descend"
            self._run(plan["pd"], stage, True)

            stage = "release"
            self.robot.set_gripper(self.grip.open_width)

            stage = "retreat"
            self._run(plan["rt"], stage, True)

            return PickResult(True, "done", "", det.position, det.color)

        except PlanningError as e:
            self.robot.stop()
            return PickResult(False, e.stage, str(e), det.position, det.color)

        finally:
            self.ik.is_valid = original_ik_valid
            self.safety.collision_fn = original_collision_fn

    def run(self):
        """Detect cubes and matching floor zones -> place each cube onto a spaced sub-slot of its same-colored zone."""
        self.go_observe()
        dets, zones = self.detect_scene()
        results = []
        slot_cursor = {c: 0 for c in zones}

        for d in dets:
            if d.color not in zones:
                continue
            z_xy = zones[d.color].position[:2]
            if np.linalg.norm(d.position[:2] - z_xy) < 0.062:
                continue
            sub = PAD_SUB_OFFSETS[slot_cursor[d.color] % len(PAD_SUB_OFFSETS)]
            target_xy = np.array([z_xy[0] + sub[0], z_xy[1] + sub[1]])
            res = self.pick_and_place_one(d, target_xy)
            if res.success:
                slot_cursor[d.color] += 1
            results.append(res)

        try:
            self.go_observe()
        except PlanningError:
            pass
        return dets, results

