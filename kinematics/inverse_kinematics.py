"""Numerical IK: damped least squares (Levenberg-Marquardt style) with joint-limit
clamping, multi-start to enumerate several configurations, singularity screening
and reachability reporting."""
from dataclasses import dataclass
import numpy as np
from .transformations import pose_error


@dataclass
class IKSolution:
    q: np.ndarray
    pos_err: float
    rot_err: float
    sigma_min: float          # smallest singular value of J (singularity measure)
    converged: bool


class IKSolver:
    def __init__(self, chain, pos_tol=1e-4, rot_tol=1e-3, max_iter=200,
                 sigma_min_thresh=2e-3, n_random_starts=24, seed=0, margin=1e-3):
        self.chain = chain
        self.pos_tol, self.rot_tol, self.max_iter = pos_tol, rot_tol, max_iter
        self.sigma_thresh = sigma_min_thresh
        self.n_starts = n_random_starts
        self.rng = np.random.default_rng(seed)
        self.margin = margin
        self.is_valid = None       # optional callback q -> bool (e.g. collision check)

    def _clamp(self, q):
        return np.clip(q, self.chain.lower + self.margin, self.chain.upper - self.margin)

    def solve_from(self, T_target, q0):
        ch = self.chain
        q = self._clamp(np.array(q0, float))
        lam = 0.05
        for _ in range(self.max_iter):
            T = ch.fk(q)
            e = pose_error(T, T_target)
            if np.linalg.norm(e[:3]) < self.pos_tol and np.linalg.norm(e[3:]) < self.rot_tol:
                break
            J = ch.jacobian(q)
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + (lam ** 2) * np.eye(6), e)
            step = np.max(np.abs(dq))
            if step > 0.3:                      # limit step size (rad)
                dq *= 0.3 / step
            q_new = self._clamp(q + dq)
            e_new = np.linalg.norm(pose_error(ch.fk(q_new), T_target))
            if e_new < np.linalg.norm(e):
                q, lam = q_new, max(lam * 0.7, 1e-4)
            else:
                lam = min(lam * 2.0, 1.0)       # increase damping when stuck
        T = ch.fk(q)
        e = pose_error(T, T_target)
        sig = np.linalg.svd(ch.jacobian(q), compute_uv=False)[-1]
        pe, re = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
        return IKSolution(q, pe, re, float(sig), pe < self.pos_tol * 2 and re < self.rot_tol * 2)

    def solve_all(self, T_target, q_seed):
        """All distinct valid configurations found, closest to q_seed first.
        Singular and non-converged results are discarded (=> unreachable pose)."""
        q_seed = np.asarray(q_seed, float)
        seeds = [q_seed]
        for _ in range(self.n_starts):
            seeds.append(self.rng.uniform(self.chain.lower, self.chain.upper))
        sols = []
        for s in seeds:
            sol = self.solve_from(T_target, s)
            if not sol.converged or sol.sigma_min < self.sigma_thresh:
                continue
            if any(np.max(np.abs(sol.q - o.q)) < 1e-2 for o in sols):
                continue
            sols.append(sol)
        sols.sort(key=lambda s: np.linalg.norm(s.q - q_seed))
        return sols

    def solve(self, T_target, q_seed):
        """Best solution (closest to q_seed, which keeps motion continuous) or None."""
        ok = (lambda q: True) if self.is_valid is None else self.is_valid
        first = self.solve_from(T_target, q_seed)
        if first.converged and first.sigma_min >= self.sigma_thresh and ok(first.q):
            return first
        for s in self.solve_all(T_target, q_seed):
            if ok(s.q):
                return s
        return None
