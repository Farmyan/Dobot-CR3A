from pathlib import Path

RRT = r'''# ---------------- collision-aware joint-space planning (RRT-Connect) ----------------
def _edge_ok(a, b, valid, res=0.05):
    n = max(1, int(np.ceil(np.max(np.abs(b - a)) / res)))
    return all(valid(a + (b - a) * s) for s in np.linspace(0, 1, n + 1)[1:])


def rrt_connect(q0, q1, valid, lower, upper, rng, max_iter=800, step=0.4, res=0.05):
    """Bidirectional RRT. `valid(q)` must be True for every configuration on the path
    (the start itself is not checked). Returns a list of waypoints or None."""
    ta, pa = [np.asarray(q0, float)], [-1]
    tb, pb = [np.asarray(q1, float)], [-1]
    swapped = False

    def extend(tree, par, q):
        i = int(np.argmin(np.linalg.norm(np.array(tree) - q, axis=1)))
        d = q - tree[i]
        dist = float(np.linalg.norm(d))
        if dist < 1e-9:
            return i
        qn = q if dist <= step else tree[i] + d / dist * step
        if not _edge_ok(tree[i], qn, valid, res):
            return None
        tree.append(qn)
        par.append(i)
        return len(tree) - 1

    def trace(tree, par, i):
        out = []
        while i != -1:
            out.append(tree[i])
            i = par[i]
        return out[::-1]

    for _ in range(max_iter):
        qr = rng.uniform(lower, upper) if rng.random() > 0.1 else tb[0]
        ia = extend(ta, pa, qr)
        if ia is not None:
            qn = ta[ia]
            while True:
                ib = extend(tb, pb, qn)
                if ib is None:
                    break
                if np.max(np.abs(tb[ib] - qn)) < 1e-9:
                    full = trace(ta, pa, ia) + trace(tb, pb, ib)[::-1][1:]
                    return full[::-1] if swapped else full
        ta, pa, tb, pb = tb, pb, ta, pa
        swapped = not swapped
    return None


def shortcut(path, valid, rng, iters=80, res=0.05):
    path = list(path)
    for _ in range(iters):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i >= 2 and _edge_ok(path[i], path[j], valid, res):
            path = path[: i + 1] + path[j:]
    return path


def plan_waypoints(wps, vmax, amax, dt=0.01):
    """Chain of rest-to-rest quintic segments through the waypoints."""
    ts, qs, t0 = [], [], 0.0
    for a, b in zip(wps[:-1], wps[1:]):
        tr = plan_joint(a, b, vmax, amax, dt)
        k = 0 if not qs else 1
        ts.append(tr.t[k:] + t0)
        qs.append(tr.q[k:])
        t0 += tr.t[-1]
    return Trajectory(np.concatenate(ts), np.vstack(qs), dt)


def plan_joint_rrt(q0, q1, valid, lower, upper, vmax, amax, dt=0.01, seed=0, max_iter=800):
    if not valid(np.asarray(q1, float)):
        raise PlanningError("rrt", "goal configuration is not valid")
    rng = np.random.default_rng(seed)
    path = rrt_connect(q0, q1, valid, lower, upper, rng, max_iter=max_iter)
    if path is None:
        raise PlanningError("rrt", "no collision-free joint-space path found")
    return plan_waypoints(shortcut(path, valid, rng), vmax, amax, dt)
'''

CFG = r'''    def config_ok(self, q, allow_low=False):
        """Single-configuration version of the checks in validate()."""
        if np.any(q < self.chain.lower + self.margin) or np.any(q > self.chain.upper - self.margin):
            return False
        pts = self.chain.link_points(q)
        tcp = self.chain.fk(q)[:3, 3]
        x0, x1, y0, y1, z0, z1 = self.ws.tcp_box
        zmin = self.ws.table_z - 1e-3 if allow_low else z0 + self.ws.table_z
        if not (x0 <= tcp[0] <= x1 and y0 <= tcp[1] <= y1 and zmin <= tcp[2] <= z1 + self.ws.table_z):
            return False
        if any(p[2] < self.ws.table_z + self.ws.link_min_z for p in pts[2:-1]):
            return False
        return self.collision_fn is None or bool(self.collision_fn(q))

'''

p = Path("control/trajectory_planning.py")
s = p.read_text(encoding="utf-8")
if "def rrt_connect" in s:
    print("skip    control/trajectory_planning.py: RRT already present")
else:
    p.write_text(s.rstrip("\n") + "\n" + RRT, encoding="utf-8")
    print("ok      control/trajectory_planning.py: RRT planner appended")

p = Path("deployment/safety.py")
s = p.read_text(encoding="utf-8")
a = "    def validate(self, traj, allow_low_tcp=False):\n"
if "def config_ok" in s:
    print("skip    deployment/safety.py: config_ok already present")
elif s.count(a) == 1:
    p.write_text(s.replace(a, CFG + a), encoding="utf-8")
    print("ok      deployment/safety.py: config_ok added")
else:
    print("PROBLEM deployment/safety.py: anchor not found")

def patch(path, pairs):
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        print(("ok      " if s.count(a) == 1 else "PROBLEM(%d) " % s.count(a)) + path + ": " + a.strip()[:60])
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8")

patch("control/pick_and_place.py", [
    ('from control.trajectory_planning import plan_joint, plan_cartesian_line, PlanningError\n',
     'from control.trajectory_planning import plan_joint, plan_cartesian_line, plan_joint_rrt, PlanningError\n'),
    ('        self._run(plan_joint(q0, self.q_observe, self.vmax, self.amax), "go_observe")\n',
     '        self._run(self._joint_path(q0, self.q_observe, "go_observe"), "go_observe")\n'),
    ('    def compute_observe_pose(self, q_seed):\n',
     '    def _joint_path(self, q0, q1, stage):\n'
     '        """Direct joint-space move if it is valid, otherwise a collision-free RRT path."""\n'
     '        direct = plan_joint(q0, q1, self.vmax, self.amax)\n'
     '        try:\n'
     '            self._validate(direct, stage)\n'
     '            return direct\n'
     '        except PlanningError:\n'
     '            pass\n'
     '        tr = plan_joint_rrt(q0, q1, self.safety.config_ok, self.chain.lower, self.chain.upper,\n'
     '                            self.vmax, self.amax)\n'
     '        self._validate(tr, stage)\n'
     '        return tr\n\n'
     '    def compute_observe_pose(self, q_seed):\n'),
    ('                t_tr = plan_joint(q_start, qpa, self.vmax, self.amax)\n                self._validate(t_tr, "transport")\n',
     '                t_tr = self._joint_path(q_start, qpa, "transport")\n'),
    ('                    t_app = plan_joint(q_now, qa, self.vmax, self.amax)\n                    self._validate(t_app, "approach")\n',
     '                    t_app = self._joint_path(q_now, qa, "approach")\n'),
    ('        why, tried = "no IK solution for any grasp orientation", 0\n',
     '        why, tried, fails = "no collision-free IK solution for any grasp orientation", 0, {}\n'),
    ('                except PlanningError as e:\n                    why = str(e)\n            if tried > self.max_candidates:\n                break\n        raise PlanningError("grasp_plan", why)\n',
     '                except PlanningError as e:\n                    why = str(e)\n                    fails[e.stage] = fails.get(e.stage, 0) + 1\n'
     '            if tried > self.max_candidates:\n                break\n'
     '        raise PlanningError("grasp_plan", f"{tried} candidates tried {fails}; last: {why}")\n'),
])
patch("main.py", [
    ('failures=[(r.stage, r.detail) for r in results if not r.success],',
     'failures=[(r.stage, r.detail, [round(float(v) * 1000) for v in r.cube_estimate[:2]]) for r in results if not r.success],'),
])
