"""Reachability map: can the arm grasp a cube tool-down (two yaws) with a collision-free approach?"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import ROBOT, WORKSPACE, CAMERA
from main import build_chain
from simulation.environment import SimEnvironment
from kinematics.inverse_kinematics import IKSolver
from kinematics.transformations import make_T, Rx
from control.grasp_planning import grasp_pose, offset_z

margin = float(sys.argv[1]) if len(sys.argv) > 1 else 0.003
chain = build_chain()
env = SimEnvironment(ROOT / "simulation/scene.xml", ROBOT, WORKSPACE, CAMERA, chain, 0, plan_margin=margin)
ik = IKSolver(chain, n_random_starts=12)
q_obs = ik.solve(make_T(Rx(np.pi), WORKSPACE.observe_xyz), (chain.lower + chain.upper) / 2)
env.reset(q_obs.q, n_cubes=0)
ik.is_valid = env.config_collision_free
xs = np.arange(0.16, 0.53, 0.03)
ys = np.arange(-0.30, 0.301, 0.075)
z = WORKSPACE.table_z + WORKSPACE.cube_size / 2
print(f"margin {margin*1000:.1f} mm.  '#' = both yaws reachable, '+' = one yaw, '.' = unreachable")
print("        y(mm): " + " ".join(f"{int(y*1000):5d}" for y in ys))
for x in xs:
    row = []
    for y in ys:
        n = 0
        for yaw in (0.0, 0.5):
            Tg = grasp_pose(np.array([x, y, z]), yaw)
            sg = ik.solve(Tg, q_obs.q)
            if sg is not None and ik.solve(offset_z(Tg, WORKSPACE.approach_height), sg.q) is not None:
                n += 1
        row.append("#" if n == 2 else "+" if n == 1 else ".")
    print(f"x={int(x*1000):4d} mm      " + " ".join(f"{c:>5s}" for c in row))
