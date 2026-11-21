"""Safety validation that sits between the planner and ANY robot (sim or real)."""
from dataclasses import dataclass
import numpy as np


@dataclass
class SafetyReport:
    ok: bool
    reasons: list


class SafetyValidator:
    def __init__(self, chain, workspace, vmax, amax, speed_scale=1.0, joint_margin=0.01):
        self.chain, self.ws = chain, workspace
        self.vmax, self.amax = vmax * speed_scale, amax * speed_scale
        self.margin = joint_margin
        self.collision_fn = None   # optional callback q -> True if collision-free

    def config_ok(self, q, allow_low=False):
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

    def validate(self, traj, allow_low_tcp=False):
        r = []
        q, dt = traj.q, traj.dt
        if not np.all(np.isfinite(q)):
            return SafetyReport(False, ["non-finite joint values"])
        lo, hi = self.chain.lower + self.margin, self.chain.upper - self.margin
        if np.any(q < lo) or np.any(q > hi):
            r.append("joint limit violation")
        v = np.diff(q, axis=0) / dt
        if len(v) and np.any(np.abs(v) > self.vmax * 1.02):
            r.append(f"velocity limit exceeded ({np.abs(v).max():.2f} rad/s)")
        a = np.diff(v, axis=0) / dt
        if len(a) and np.any(np.abs(a) > self.amax * 1.5):   # finite-difference noise margin
            r.append(f"acceleration limit exceeded ({np.abs(a).max():.2f} rad/s^2)")
        if self.collision_fn is not None:
            dq = np.abs(np.diff(q, axis=0)).max(axis=1) if len(q) > 1 else np.zeros(0)
            cum = np.concatenate([[0.0], np.cumsum(dq)])
            marks = np.arange(0.0, cum[-1] + 1e-9, 0.02)          # a collision check every ~0.02 rad
            idx = np.unique(np.clip(np.searchsorted(cum, marks), 0, len(q) - 1))
            for qi in list(q[idx]) + [q[-1]]:
                if not self.collision_fn(qi):
                    r.append('configuration in collision (self/table) along trajectory')
                    break
        x0, x1, y0, y1, z0, z1 = self.ws.tcp_box
        for qi in q[:: max(1, len(q) // 60)]:                # sampled collision/workspace check
            pts = self.chain.link_points(qi)
            tcp = self.chain.fk(qi)[:3, 3]
            zmin = self.ws.table_z - 1e-3 if allow_low_tcp else z0 + self.ws.table_z
            if not (x0 <= tcp[0] <= x1 and y0 <= tcp[1] <= y1 and zmin <= tcp[2] <= z1 + self.ws.table_z):
                r.append(f"TCP outside safe workspace at {np.round(tcp, 3)}")
                break
            if any(p[2] < self.ws.table_z + self.ws.link_min_z for p in pts[2:-1]):
                r.append("arm link below table clearance")
                break
        return SafetyReport(not r, r)
