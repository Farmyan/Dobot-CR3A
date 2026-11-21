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
        self._validate(traj, stage, allow_low)
        if not self.robot.execute_trajectory(traj):
            raise PlanningError(stage, "robot reported error: " + str(self.robot.get_errors()))

    def _holding(self):
        w = self.robot.get_gripper_width()
        return self.grip.min_grasp_width <= w <= self.grip.max_grasp_width

    def _line(self, T0, T1, q_seed):
        return plan_cartesian_line(self.ik, T0, T1, q_seed, self.vmax, self.amax)

    def _joint_path(self, q0, q1, stage):
        direct = plan_joint(q0, q1, self.vmax, self.amax)
        try:
            self._validate(direct, stage)
            return direct
        except PlanningError:
            pass
        tr = plan_joint_rrt(q0, q1, self.safety.config_ok, self.chain.lower, self.chain.upper,
                            self.vmax, self.amax)
        self._validate(tr, stage)
        return tr

    def compute_observe_pose(self, q_seed):
        x, y, z = self.ws.observe_xyz
        sol = self.ik.solve(make_T(Rx(np.pi), [x, y, z]), q_seed)
        if sol is None:
            raise PlanningError("observe", "observe pose unreachable; edit WORKSPACE.observe_xyz")
        self.q_observe = sol.q
        return sol.q

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
        seen = []
        fast = self.ik.solve_from(T, q_ref)
        if fast.converged and fast.sigma_min >= self.ik.sigma_thresh and ok(fast.q):
            seen.append(fast.q)
            yield fast.q
        for s in self.ik.solve_all(T, q_ref):
            if ok(s.q) and all(np.max(np.abs(s.q - o)) > 1e-2 for o in seen):
                seen.append(s.q)
                yield s.q

    def _plan_place(self, target_xy, q_start):
        Tp = offset_z(
            place_pose(target_xy, self.ws.table_z, self.ws.cube_size, yaw=0.0),
            self.ws.tcp_to_cube_center_z + 0.002,
        )
        Tpa = offset_z(Tp, self.ws.approach_height)
        why = "no IK solution for the place pose"
        for k, qpa in enumerate(self._ik_candidates(Tpa, q_start)):
            if k >= self.max_candidates:
                break
            try:
                t_tr = self._joint_path(q_start, qpa, "transport")
                t_pd = self._line(Tpa, Tp, qpa)
                self._validate(t_pd, "place_descend", True)
                t_rt = self._line(Tp, Tpa, t_pd.q[-1])
                self._validate(t_rt, "retreat", True)
                return t_tr, t_pd, t_rt
            except PlanningError as e:
                why = str(e)
        raise PlanningError("place_plan", why)

    def _plan_pick(self, det, target_xy):
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
                    t_tr, t_pd, t_rt = self._plan_place(target_xy, t_lift.q[-1])
                    t_back = self._line(Tg, Ta, t_des.q[-1])
                    print(
                        f"[GRASP ORIENTATION] candidate={candidate_index} "
                        f"yaw={candidate_yaw:.2f} deg "
                        f"detected_yaw={np.degrees(det.yaw):.2f} deg",
                        flush=True,
                    )
                    return dict(app=t_app, des=t_des, lift=t_lift, tr=t_tr, pd=t_pd, rt=t_rt, back=t_back)
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
                return PickResult(False, "grasp", f"gripper width {w*1000:.1f} mm (miss)", det.position, det.color)
            stage = "lift"
            self._run(plan["lift"], stage, True)
            if not self._holding():
                return PickResult(False, "lift", "cube lost during lift", det.position, det.color)
            stage = "transport"
            self._run(plan["tr"], stage)
            if not self._holding():
                return PickResult(False, "transport", "cube dropped during transport", det.position, det.color)
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
