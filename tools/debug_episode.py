"""Single-cube diagnostic v3 (compact): per-move tracking/stall report, grip-loss report."""
import sys
from pathlib import Path
import numpy as np
import mujoco
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import ROBOT, WORKSPACE, CAMERA
from main import build_chain, build_controller
from simulation.environment import SimEnvironment, SimCamera
from deployment.simulation_interface import SimulationInterface
from kinematics.inverse_kinematics import IKSolver
from kinematics.transformations import make_T, Rx, inv_T

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
chain = build_chain()
env = SimEnvironment(ROOT / "simulation/scene.xml", ROBOT, WORKSPACE, CAMERA, chain, seed)
sol = IKSolver(chain).solve(make_T(Rx(np.pi), WORKSPACE.observe_xyz), (chain.lower + chain.upper) / 2)
env.reset(sol.q)

# Keep only the first generated cube for this diagnostic.
first_cube = env.ground_truth_cubes()[0]
first_idx = env.cubes.index(first_cube[0])
first_qpos = env.data.qpos[env.cube_adr[first_idx]:env.cube_adr[first_idx] + 7].copy()
first_qvel_adr = env.model.joint(f"{first_cube[0]}_free").dofadr[0]
first_qvel = env.data.qvel[first_qvel_adr:first_qvel_adr + 6].copy()

env.clear_cubes()
env.data.qpos[env.cube_adr[first_idx]:env.cube_adr[first_idx] + 7] = first_qpos
env.data.qvel[first_qvel_adr:first_qvel_adr + 6] = first_qvel
env.cube_colors[first_idx] = first_cube[2]
mujoco.mj_forward(env.model, env.data)
NAMES = {1: "go_observe", 2: "approach", 3: "descend", 4: "lift", 5: "transport",
         6: "place_descend", 7: "retreat", 8: "return_home"}
m, d = env.model, env.data


def gname(g):
    b = m.geom_bodyid[g]
    return f"{m.body(b).name}/{m.geom(g).name or 'mesh'}"


def cube_grip_contacts_by_finger():
    cid = env.cube_bodies[0]
    finger_bodies = {
        "finger1": m.body("ag95_finger1_tip").id,
        "finger2": m.body("ag95_finger2_tip").id,
    }
    result = {
        name: {"contacts": 0, "force": 0.0}
        for name in finger_bodies
    }

    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]

        if cid not in (b1, b2):
            continue

        other_body = b2 if b1 == cid else b1

        for name, body_id in finger_bodies.items():
            if other_body == body_id:
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, f)
                result[name]["contacts"] += 1
                result[name]["force"] += float(f[0])

    return result


def cube_contact_details():
    cube_geom = env.cube_geoms[first_idx]
    finger_geoms = {
        "finger1": env.grip_geom_ids["ag95_finger1_tip"],
        "finger2": env.grip_geom_ids["ag95_finger2_tip"],
    }
    cube_center = env.data.geom_xpos[cube_geom].copy()
    result = []

    for i in range(d.ncon):
        c = d.contact[i]
        pair = {int(c.geom1), int(c.geom2)}

        for name, finger_geom in finger_geoms.items():
            if pair != {int(cube_geom), int(finger_geom)}:
                continue

            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force)

            # MuJoCo contact position and frame are in world coordinates.
            point = np.asarray(c.pos).copy()
            normal = np.asarray(c.frame[:3]).copy()

            result.append({
                "finger": name,
                "point_cube_mm": np.round((point - cube_center) * 1000, 2),
                "normal_world": np.round(normal, 4),
                "distance_mm": round(float(c.dist) * 1000, 3),
                "normal_force_N": round(float(force[0]), 4),
            })

    return result


def cube_grip_contacts():
    result = cube_grip_contacts_by_finger()
    n = sum(v["contacts"] for v in result.values())
    fn = sum(v["force"] for v in result.values())
    return n, fn


def other_contacts():
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if not (b1 in env.robot_bodies or b2 in env.robot_bodies):
            continue
        if (b1 in env.cube_bodies or b2 in env.cube_bodies) and ((b1 in env.gripper_bodies) or (b2 in env.gripper_bodies)):
            continue
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f)
        out.append(f"{gname(c.geom1)} <-> {gname(c.geom2)} dist={c.dist*1000:.1f}mm F={f[0]:.1f}N")
    return out


def cube_in_tool():
    tcp = env.tcp_pose_mujoco()
    return np.round((inv_T(tcp) @ np.append(env.ground_truth_cubes()[0][1], 1))[:3] * 1000, 1)


class Logged(SimulationInterface):
    def __init__(self, env):
        super().__init__(env)
        self.n_traj, self.holding = 0, False

    def execute_trajectory(self, traj):
        self.n_traj += 1
        tag = NAMES.get(self.n_traj, f"move{self.n_traj}")
        trk, lost = 0.0, False
        for i, q in enumerate(traj.q):
            env.step_to(q, traj.dt)
            trk = max(trk, float(np.abs(env.joint_positions() - q).max()))
            if self.holding and not lost and cube_grip_contacts()[0] == 0:
                lost = True
                fing = [round(d.qpos[a] * 1000, 1) for a in env.grip_q]
                print(f"  !! GRIP LOST in [{tag}] at {traj.t[i]:.2f}s of {traj.t[-1]:.2f}s; fingers(mm)={fing} "
                      f"width={env.gripper_width()*1000:.1f}mm cube_in_tool={cube_in_tool()}")
        self._settle(traj.q[-1])
        qerr = float(np.degrees(np.abs(env.joint_positions() - traj.q[-1]).max()))
        tcp_err = float(np.linalg.norm(env.tcp_pose_mujoco()[:3, 3] - chain.fk(traj.q[-1])[:3, 3]) * 1000)
        print(f"[{tag}] T={traj.t[-1]:.2f}s  lag_max={np.degrees(trk):.1f}deg  settled_err={qerr:.2f}deg  "
              f"tcp_err={tcp_err:.1f}mm  tcp_z={env.tcp_pose_mujoco()[2, 3]*1000:.0f}mm")
        if tcp_err > 3.0:
            print("  STALLED. contacts involving the robot (excluding gripper-cube):")
            for s in other_contacts()[:8]:
                print("   ", s)
            print("  q_cmd (deg):   ", np.round(np.degrees(traj.q[-1]), 1))
            print("  q_actual (deg):", np.round(np.degrees(env.joint_positions()), 1))
        return True

    def set_gripper(self, width, wait=True):
        # TEMPORARY DIAGNOSTIC: test stronger position control.
        env.model.actuator_gainprm[env.grip_act, 0] = 100.0
        env.model.actuator_biasprm[env.grip_act, 1] = -100.0
        env.model.actuator_biasprm[env.grip_act, 2] = -2.0

        angle = env.gripper_angle_for_width(width)
        env.data.ctrl[env.grip_act] = angle
        q_hold = env.joint_positions()

        print(f"\n-> gripper command: width={width*1000:.1f}mm "
              f"target_angle={angle:.5f}rad "
              f"start_angle={env.data.qpos[env.grip_q[0]]:.5f}rad")

        for i in range(200 if wait else 0):
            if self.abort_requested:
                return

            env.step_to(q_hold, 0.03)
            self._pump()

            if (i + 1) % 10 == 0:
                q = env.data.qpos[env.grip_q[0]]
                qv = env.data.qvel[
                    env.model.joint("ag95_finger1_joint").dofadr[0]
                ]
                force = env.data.actuator_force[env.grip_act]

                cube_geom = env.cube_geoms[first_idx]
                f1_geom = env.grip_geom_ids["ag95_finger1_tip"]
                f2_geom = env.grip_geom_ids["ag95_finger2_tip"]

                d1 = mujoco.mj_geomDistance(
                    m, d, cube_geom, f1_geom, 1.0, np.zeros(6)
                )
                d2 = mujoco.mj_geomDistance(
                    m, d, cube_geom, f2_geom, 1.0, np.zeros(6)
                )

                print(
                    f"   t={(i+1)*0.03:.2f}s "
                    f"angle={q:.5f} "
                    f"vel={qv:.5f} "
                    f"force={force:.3f}N "
                    f"width={env.gripper_width()*1000:.2f}mm "
                    f"finger1_dist={d1*1000:.2f}mm "
                    f"finger2_dist={d2*1000:.2f}mm"
                )

        n, fn = cube_grip_contacts()
        per_finger = cube_grip_contacts_by_finger()
        f1 = per_finger["finger1"]
        f2 = per_finger["finger2"]
        print(f"   final width={env.gripper_width()*1000:.1f}mm "
              f"contacts={n} Fn={fn:.2f}N "
              f"finger1={f1['contacts']} contacts/{f1['force']:.3f}N "
              f"finger2={f2['contacts']} contacts/{f2['force']:.3f}N "
              f"cube_in_tool={cube_in_tool()}")
        self.holding = width < 0.03 and n > 0
        for contact in cube_contact_details():
            print("   CONTACT", contact)
        # TEMPORARY DIAGNOSTIC: measure each finger's clearance to the cube.
        cube_geom = env.cube_geoms[first_idx]
        cube_center = d.geom_xpos[cube_geom].copy()
        cube_R = d.geom_xmat[cube_geom].reshape(3, 3).copy()

        # Measure the cube pose relative to the physical TCP frame.
        tcp_T = env.tcp_pose_mujoco()
        tcp_R = tcp_T[:3, :3]
        cube_pos_tcp = tcp_R.T @ (cube_center - tcp_T[:3, 3])
        cube_R_tcp = tcp_R.T @ cube_R
        relative_angle = np.degrees(np.arccos(np.clip(
            (np.trace(cube_R_tcp) - 1.0) / 2.0, -1.0, 1.0
        )))

        print("   CUBE POSE RELATIVE TO TCP:")
        print(f"      center TCP (mm): {np.round(cube_pos_tcp * 1000, 2)}")
        print(f"      relative rotation angle: {relative_angle:.3f} deg")
        print("      relative rotation matrix:")
        print(np.round(cube_R_tcp, 5))

        print("   FINGER CLEARANCE DIAGNOSTIC:")
        for name in ("ag95_finger1_tip", "ag95_finger2_tip"):
            finger_geom = env.grip_geom_ids[name]
            finger_center = d.geom_xpos[finger_geom].copy()

            # Express finger position in the cube's local coordinate frame.
            local_pos = cube_R.T @ (finger_center - cube_center)

            fromto = np.zeros(6)
            distance = mujoco.mj_geomDistance(
                m, d, cube_geom, finger_geom, 1.0, fromto
            )

            print(
                f"      {name}: "
                f"distance={distance*1000:.2f}mm "
                f"local_center={np.round(local_pos*1000, 2)}mm"
            )

        print("   ACTUAL SOLVER-BASED FINGER DISTANCES:")
        print("      (Distances at the end of the real closing motion.)")
        for name in ("ag95_finger1_tip", "ag95_finger2_tip"):
            distance = mujoco.mj_geomDistance(
                m, d, cube_geom,
                env.grip_geom_ids[name],
                1.0, np.zeros(6)
            )
            print(f"      {name}: {distance*1000:.3f} mm")

        print("   FINGER TRANSFORMS AT GRASP:")
        for name in ("ag95_finger1_tip", "ag95_finger2_tip"):
            bid = m.body(name).id
            gid = env.grip_geom_ids[name]

            print(f"      {name}")
            print("         body pos mm:",
                  np.round(d.xpos[bid] * 1000, 2))
            print("         geom pos mm:",
                  np.round(d.geom_xpos[gid] * 1000, 2))
            print("         geom rotation:")
            print(np.round(d.geom_xmat[gid].reshape(3, 3), 4))

robot = Logged(env)
pnp = build_controller(robot, SimCamera(env), chain)
pnp.q_observe = sol.q
# TEMPORARY: report the first rejected collision configurations.
_original_collision_check = env.config_collision_free
_collision_reject_count = [0]

def diagnostic_collision_check(q):
    ok = _original_collision_check(q)
    if not ok and _collision_reject_count[0] < 12:
        _collision_reject_count[0] += 1
        print(
            f"\\n[COLLISION REJECT {_collision_reject_count[0]}] "
            f"joints_deg={np.round(np.rad2deg(q), 1)}",
            flush=True
        )
        for contact in env.explain_collision(q):
            print("   " + contact, flush=True)
    return ok

pnp.ik.is_valid = diagnostic_collision_check
pnp.safety.collision_fn = diagnostic_collision_check
pnp.go_observe()
dets = pnp.detect()
gt = env.ground_truth_cubes()[0]

def fold_yaw(a):
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4

# Exact cube yaw from its saved MuJoCo free-joint quaternion (w, x, y, z).
actual_yaw = 2.0 * np.arctan2(first_qpos[6], first_qpos[3])

print(f"\n========== YAW VALIDATION (seed {seed}) ==========")
print(f"Cube quaternion: {np.round(first_qpos[3:7], 6)}")
print(f"Actual cube yaw:   {np.degrees(fold_yaw(actual_yaw)):.2f} deg")

if not dets:
    raise RuntimeError("Detector returned no cubes; cannot run yaw validation.")

det = dets[0]
yaw_error = fold_yaw(det.yaw - actual_yaw)

print(f"Detected cube yaw: {np.degrees(det.yaw):.2f} deg")
print(f"Folded yaw error:  {np.degrees(yaw_error):.2f} deg")
print(f"Position error:    {np.linalg.norm(det.position - gt[1])*1000:.2f} mm")
print(f"Cube XY:           {np.round(gt[1][:2]*1000, 1)} mm")

print("Grasp candidate yaws:")
for k in (0, 1, -1, 2):
    candidate_yaw = det.yaw + k * np.pi / 2
    print(f"  k={k:2d}: {np.degrees(candidate_yaw):8.2f} deg")

res = pnp.pick_and_place_one(
    det,
    np.array(WORKSPACE.place_xy, dtype=float)
)
print("RESULT:", res.success, res.stage, res.detail)
fin = env.ground_truth_cubes()[0][1]
print(f"final cube xy(mm)={np.round(fin[:2]*1000, 1)}  distance to slot={np.linalg.norm(fin[:2]-np.array(WORKSPACE.place_xy))*1000:.1f} mm  "
      f"unsafe_contacts={env.unsafe_contacts} joint_violations={env.joint_violations}")





