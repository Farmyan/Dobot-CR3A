"""Task 2 validation: analytic FK (from URDF chain) vs MuJoCo TCP site, 200 random poses.
Needs MuJoCo + generated simulation/robot.xml. Run: python tests/validate_fk_vs_mujoco.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, mujoco
from config import ROBOT, WORKSPACE, CAMERA, ROOT
from main import build_chain
from simulation.environment import SimEnvironment
from kinematics.transformations import pose_error

chain = build_chain()
env = SimEnvironment(ROOT / "simulation/scene.xml", ROBOT, WORKSPACE, CAMERA, chain)
rng = np.random.default_rng(0)
pe, re = [], []
for _ in range(200):
    q = rng.uniform(chain.lower * 0.8, chain.upper * 0.8)
    for a, v in zip(env.jadr, q):
        env.data.qpos[a] = v
    mujoco.mj_forward(env.model, env.data)
    e = pose_error(chain.fk(q), env.tcp_pose_mujoco())
    pe.append(np.linalg.norm(e[:3]) * 1000); re.append(np.degrees(np.linalg.norm(e[3:])))
print(f"FK vs MuJoCo: max position error {max(pe):.4f} mm, max orientation error {max(re):.4f} deg")
