import numpy as np
import mujoco
from config import ROBOT, ROOT
from main import build_chain
from kinematics.transformations import make_T, inv_T

chain = build_chain()
model = mujoco.MjModel.from_xml_path(str(ROOT / "simulation/scene.xml"))
data = mujoco.MjData(model)

base_id = model.body(ROBOT.base_link).id
site_id = model.site("tcp").id
joint_ids = [model.joint(name).id for name in chain.names]
qpos_adr = [model.jnt_qposadr[jid] for jid in joint_ids]

rng = np.random.default_rng(42)
max_pos = 0.0
max_rot = 0.0

for k in range(12):
    q = rng.uniform(chain.lower * 0.65, chain.upper * 0.65)

    for adr, value in zip(qpos_adr, q):
        data.qpos[adr] = value

    mujoco.mj_forward(model, data)

    T_world_base = make_T(
        data.xmat[base_id].reshape(3, 3),
        data.xpos[base_id]
    )
    T_world_tcp = make_T(
        data.site_xmat[site_id].reshape(3, 3),
        data.site_xpos[site_id]
    )

    T_base_tcp_mj = inv_T(T_world_base) @ T_world_tcp
    T_base_tcp_urdf = chain.fk(q)

    dp = np.linalg.norm(
        T_base_tcp_mj[:3, 3] - T_base_tcp_urdf[:3, 3]
    )
    dR = T_base_tcp_mj[:3, :3] @ T_base_tcp_urdf[:3, :3].T
    angle = np.arccos(
        np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0)
    )

    max_pos = max(max_pos, dp)
    max_rot = max(max_rot, angle)

    print(
        f"case {k+1:02d}: "
        f"position error={dp*1000:.4f} mm, "
        f"orientation error={np.degrees(angle):.4f} deg"
    )

print(f"\nMAX position error: {max_pos*1000:.4f} mm")
print(f"MAX orientation error: {np.degrees(max_rot):.4f} deg")

if max_pos > 0.5e-3 or max_rot > np.deg2rad(0.2):
    raise SystemExit(
        "FAIL: URDF TCP and MuJoCo tcp site do not agree within tolerance."
    )

print("PASS: URDF TCP matches the physical MuJoCo TCP site.")
