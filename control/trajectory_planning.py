"""Joint-space quintic trajectories q(t)=q0+s(t)(qf-q0), s(t)=10t^3-15t^4+6t^5
(s(0)=0, s(1)=1, s'=s''=0 at both ends). Duration is chosen so velocity and
acceleration limits hold: peak |s'|=1.875/T, peak |s''|=5.7735/T^2."""
from dataclasses import dataclass
import numpy as np
from kinematics.transformations import interpolate_pose


@dataclass
class Trajectory:
    t: np.ndarray       # (N,)
    q: np.ndarray       # (N, n)
    dt: float


class PlanningError(Exception):
    def __init__(self, stage, msg):
        super().__init__(f"[{stage}] {msg}")
        self.stage = stage


def quintic(tau):
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def min_duration(dq, vmax, amax, t_min=0.2):
    dq = np.abs(dq)
    tv = 1.875 * dq / vmax
    ta = np.sqrt(5.7735 * dq / amax)
    return float(max(t_min, tv.max(), ta.max()))


def plan_joint(q0, qf, vmax, amax, dt=0.01):
    q0, qf = np.asarray(q0, float), np.asarray(qf, float)
    T = min_duration(qf - q0, vmax, amax)
    t = np.arange(0, T + dt / 2, dt)
    t[-1] = T
    s = quintic(t / T)[:, None]
    return Trajectory(t, q0 + s * (qf - q0), dt)


def plan_cartesian_line(ik, T0, T1, q_seed, vmax, amax, step=0.005, dt=0.01, max_jump=0.35):
    """Straight-line TCP motion (used for approach/lift): IK along the line, seeded by the
    previous waypoint for branch continuity, then time-scaled in joint space."""
    dist = np.linalg.norm(T1[:3, 3] - T0[:3, 3])
    n = max(2, int(np.ceil(dist / step)) + 1)
    qs, q_prev = [], np.asarray(q_seed, float)
    for s in np.linspace(0, 1, n):
        sol = ik.solve(interpolate_pose(T0, T1, s), q_prev)
        if sol is None:
            raise PlanningError("ik_line", f"no IK at s={s:.2f} of Cartesian segment")
        if np.max(np.abs(sol.q - q_prev)) > max_jump:
            raise PlanningError("ik_line", "joint branch jump along Cartesian segment")
        qs.append(sol.q)
        q_prev = sol.q
    qs = np.array(qs)
    seg = np.abs(np.diff(qs, axis=0)).sum(axis=0)
    T = min_duration(seg, vmax, amax)
    for _ in range(10):                      # stretch the duration until v and a limits really hold
        tw = np.linspace(0, T, n)
        t = np.arange(0, T + dt / 2, dt)
        t[-1] = T
        s = quintic(t / T)
        q = np.stack([np.interp(s * T, tw, qs[:, i]) for i in range(qs.shape[1])], axis=1)
        v = np.diff(q, axis=0) / dt
        a = np.diff(v, axis=0) / dt
        f = max(1.0,
                float(np.max(np.abs(v) / vmax)) if len(v) else 1.0,
                float(np.max(np.abs(a) / amax)) if len(a) else 1.0)
        if f <= 1.0:
            break
        T *= 1.05 * f
    if T > 8.0:
        raise PlanningError("ik_line", "Cartesian segment needs an unreasonably jerky joint path")
    return Trajectory(t, q, dt)
# ---------------- collision-aware joint-space planning (RRT-Connect) ----------------
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
