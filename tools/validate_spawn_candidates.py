from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import numpy as np

from config import ROBOT, WORKSPACE, CAMERA
from main import ROOT, build_chain
from simulation.environment import SimEnvironment
from kinematics.inverse_kinematics import IKSolver
from kinematics.transformations import make_T, Rx
from kinematics.transformations import interpolate_pose


CANDIDATES = np.array([
    (0.395,  0.065),
    (0.250,  0.350),
    (0.420,  0.165),
    (0.285, -0.300),
    (0.330,  0.320),
    (0.435, -0.030),
    (0.220, -0.255),
    (0.470,  0.070),
    (0.290, -0.185),
    (0.385,  0.230),
    (0.310,  0.085),
    (0.360, -0.310),
    (0.350,  0.155),
    (0.230,  0.420),
    (0.455, -0.095),
], dtype=float)


def main():
    chain = build_chain()
    env = SimEnvironment(
        ROOT / "simulation/scene.xml",
        ROBOT,
        WORKSPACE,
        CAMERA,
        chain,
        seed=0,
    )

    ik = IKSolver(chain)
    ik.is_valid = env.config_collision_free
    q_seed = (chain.lower + chain.upper) / 2
    home = ik.solve(make_T(Rx(np.pi), WORKSPACE.observe_xyz), q_seed)

    if home is None:
        raise RuntimeError("Observe pose is unreachable.")

    spawn_ik = IKSolver(chain, n_random_starts=12, seed=17)
    spawn_ik.is_valid = env.config_collision_free

    def can_reach_xy(xy, purpose="pick"):
        z_tcp = (
            WORKSPACE.table_z
            + WORKSPACE.cube_size / 2
            + WORKSPACE.tcp_to_cube_center_z
        )
        Tg = make_T(Rx(np.pi), [xy[0], xy[1], z_tcp])
        Ta = make_T(
            Rx(np.pi),
            [xy[0], xy[1], z_tcp + WORKSPACE.approach_height],
        )

        for approach_sol in spawn_ik.solve_all(Ta, home.q):
            if not env.config_collision_free(approach_sol.q):
                continue

            q_prev = approach_sol.q
            path_ok = True

            for step in range(1, 9):
                T_step = interpolate_pose(Ta, Tg, step / 8.0)
                step_sol = spawn_ik.solve_from(T_step, q_prev)

                if (
                    not step_sol.converged
                    or step_sol.sigma_min < spawn_ik.sigma_thresh
                    or not env.config_collision_free(step_sol.q)
                ):
                    path_ok = False
                    break

                q_prev = step_sol.q

            if path_ok:
                return True

        return False

    env.validator_fn = can_reach_xy

    print("\n========== EXACT CANDIDATE VALIDATION ==========", flush=True)
    print("Using production IK, approach, descent and collision checks.", flush=True)

    results = []
    started = time.perf_counter()

    for i, point in enumerate(CANDIDATES, start=1):
        t0 = time.perf_counter()
        reachable = env._is_reachable_xy(point, purpose="pick")
        elapsed = time.perf_counter() - t0
        results.append(reachable)

        print(
            f"Candidate {i:02d}/15: "
            f"({point[0]:.3f}, {point[1]:.3f}) "
            f"=> {'PASS' if reachable else 'FAIL'} "
            f"({elapsed:.2f}s)",
            flush=True,
        )

    print("\n========== SPACING VALIDATION ==========", flush=True)

    zone_points = np.array([
        (0.320, -0.075),
        (0.210, -0.375),
        (0.255,  0.220),
        (0.410, -0.190),
    ])

    cube_distances = [
        np.linalg.norm(CANDIDATES[i] - CANDIDATES[j])
        for i in range(15)
        for j in range(i + 1, 15)
    ]

    zone_distances = [
        np.linalg.norm(cube - zone)
        for cube in CANDIDATES
        for zone in zone_points
    ]

    print(f"Minimum cube separation: {min(cube_distances)*1000:.1f} mm")
    print(f"Minimum cube-zone distance: {min(zone_distances)*1000:.1f} mm")
    print(f"Required cube separation: {WORKSPACE.min_cube_spacing*1000:.1f} mm")
    print("Required cube-zone clearance: 105.0 mm")

    print("\n========== SUMMARY ==========", flush=True)
    print(f"IK reachable: {sum(results)}/15")
    print(f"Total validation time: {time.perf_counter()-started:.2f}s")
    print("Reachable candidate indices:", [
        i + 1 for i, ok in enumerate(results) if ok
    ])
    print("Unreachable candidate indices:", [
        i + 1 for i, ok in enumerate(results) if not ok
    ])

    del env


if __name__ == "__main__":
    main()


