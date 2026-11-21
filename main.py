"""Entry point.
  python main.py --mode sim --episodes 100
  python main.py --mode sim --viewer          (opens browser control panel + MuJoCo 3D viewer)
  python main.py --mode real                  (supervised; see README deployment checklist)"""
import argparse, json, time, threading, webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
from config import ROBOT, WORKSPACE, CAMERA, GRIPPER, ROOT
from kinematics.forward_kinematics import SerialChain
from kinematics.inverse_kinematics import IKSolver
from kinematics.transformations import rpy_to_R, make_T, Rx, interpolate_pose
from perception.object_detection import CubeDetector
from control.pick_and_place import PickAndPlace, PAD_SUB_OFFSETS
from deployment.safety import SafetyValidator


def build_chain():
    tcp = make_T(rpy_to_R(*ROBOT.tcp_rpy), ROBOT.tcp_xyz)
    return SerialChain.from_urdf(ROBOT.urdf_path, ROBOT.base_link, ROBOT.flange_link, tcp)


def build_controller(robot, camera, chain, speed_scale=1.0):
    vmax = chain.vel_limits(ROBOT.default_vmax) * ROBOT.vel_scale * speed_scale
    amax = np.full(chain.n, ROBOT.default_amax) * ROBOT.acc_scale * speed_scale
    ik = IKSolver(chain)
    safety = SafetyValidator(chain, WORKSPACE, vmax, amax)
    det = CubeDetector(WORKSPACE.cube_size, WORKSPACE.table_z)
    return PickAndPlace(robot, camera, det, chain, ik, safety, WORKSPACE, GRIPPER, vmax, amax)


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>DOBOT CR3A Cube Spawner</title>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; background: #16181d; color: #f1f3f5;
         display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
  .card { background: #21252b; padding: 28px 34px; border-radius: 14px; box-shadow: 0 12px 32px rgba(0,0,0,0.45);
          width: 380px; text-align: center; }
  h2 { margin: 0 0 8px; font-size: 20px; }
  p { color: #adb5bd; font-size: 13px; margin: 0 0 22px; }
  button { width: 100%; padding: 14px; margin: 8px 0; border: none; border-radius: 8px; font-size: 15px;
           font-weight: 600; cursor: pointer; transition: transform 0.06s, opacity 0.15s; }
  button:active { transform: scale(0.98); }
  .btn-gen { background: #2d7ff9; color: #fff; }
  .btn-gen:hover { background: #1e6ce0; }
  .btn-clr { background: #e03131; color: #fff; }
  .btn-clr:hover { background: #c92a2a; }
  .status { margin-top: 18px; padding: 12px; background: #16181d; border-radius: 8px;
            font-size: 13px; color: #69db7c; min-height: 20px; }
</style>
</head>
<body>
  <div class="card">
    <h2>DOBOT CR3A Color Sorter</h2>
    <p>Spawns 4+ colored 25 mm sponge cubes (random counts per color) &amp; matching floor pads.</p>
    <button class="btn-gen" onclick="sendCmd('generate')">Generate Random Cubes</button>
    <button class="btn-clr" onclick="sendCmd('clear')">Clear Table</button>
    <div class="status" id="status">Ready</div>
  </div>
<script>
async function sendCmd(cmd) {
  document.getElementById('status').innerText = 'Sending: ' + cmd + '...';
  try {
    const r = await fetch('/cmd?action=' + cmd, { method: 'POST' });
    const data = await r.json();
    document.getElementById('status').innerText = data.status;
  } catch (e) {
    document.getElementById('status').innerText = 'Connection error';
  }
}
setInterval(async () => {
  try {
    const r = await fetch('/status');
    const data = await r.json();
    document.getElementById('status').innerText = data.status;
  } catch (e) {}
}, 800);
</script>
</body>
</html>
"""


def _start_browser_ui(state, robot):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            if self.path == "/status":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": state["status"]}).encode())
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML_PAGE.encode())

        def do_POST(self):
            if "generate" in self.path:
                state["regenerate"] = True
                state["status"] = "Generating new random cubes & floor zones..."
                robot.abort_requested = True
            elif "clear" in self.path:
                state["clear"] = True
                state["status"] = "Clearing table..."
                robot.abort_requested = True
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": state["status"]}).encode())

    server = HTTPServer(("127.0.0.1", 8765), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:8765"
    print(f"[Browser UI] Control panel open at {url}")
    webbrowser.open(url)
    return server


def run_interactive_viewer(env, pnp, sol_q):
    import mujoco.viewer

    state = {"regenerate": True, "clear": False, "status": "Initializing..."}
    server = _start_browser_ui(state, pnp.robot)

    def key_callback(keycode):
        if keycode == ord("R"):
            state["regenerate"] = True
            pnp.robot.abort_requested = True
        elif keycode == ord("C"):
            state["clear"] = True
            pnp.robot.abort_requested = True

    vw = mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback)
    pnp.robot.viewer = vw

    # Track exact occupied sub-slots per color so cubes NEVER stack on top of each other
    placed_slots_by_color = {}
    failed_keys = set()

    try:
        while vw.is_running():
            if state["clear"]:
                state["clear"] = False
                state["regenerate"] = False
                pnp.robot.abort_requested = False
                placed_slots_by_color.clear()
                failed_keys.clear()
                env.clear_cubes()
                try:
                    pnp.go_observe()
                except Exception:
                    pass
                state["status"] = "Table cleared. Click 'Generate Random Cubes'."

            if state["regenerate"]:
                state["regenerate"] = False
                pnp.robot.abort_requested = False
                placed_slots_by_color.clear()
                failed_keys.clear()
                counts = env.reset(sol_q)
                if counts:
                    desc = ", ".join(f"{v} {k}" for k, v in counts.items())
                    state["status"] = f"Spawned: {desc}"

            vw.sync()
            dets, zones = pnp.detect_scene()

            pending = []
            for d in dets:
                if d.color not in zones:
                    continue
                z_xy = zones[d.color].position[:2]
                # Already placed inside this colored pad's 120x120 mm boundary
                if np.linalg.norm(d.position[:2] - z_xy) < 0.062:
                    continue
                key = (d.color, round(float(d.position[0]), 2), round(float(d.position[1]), 2))
                if key not in failed_keys:
                    pending.append((d, z_xy, key))

            if pending:
                pending.sort(key=lambda item: np.linalg.norm(item[0].position[:2] - item[1]))
                det_cube, z_xy, key = pending[0]
                slot_idx = placed_slots_by_color.get(det_cube.color, 0)
                sub = PAD_SUB_OFFSETS[slot_idx % len(PAD_SUB_OFFSETS)]
                target_xy = np.array([z_xy[0] + sub[0], z_xy[1] + sub[1]])

                state["status"] = f"Sorting {det_cube.color.upper()} cube -> slot {slot_idx + 1}..."
                print(f"[Sort] Moving {det_cube.color.upper()} cube -> {det_cube.color.upper()} pad slot {slot_idx + 1}...")
                res = pnp.pick_and_place_one(det_cube, target_xy)
                if pnp.robot.abort_requested:
                    continue
                if res.success:
                    placed_slots_by_color[det_cube.color] = slot_idx + 1
                    state["status"] = f"Placed {det_cube.color.upper()} cube (slot {slot_idx + 1})."
                else:
                    print(f"[Sort] Skipping {det_cube.color.upper()} cube ({res.stage}: {res.detail})")
                    failed_keys.add(key)
                try:
                    pnp.go_observe()
                except Exception:
                    pass
            else:
                if not state["status"].startswith("Table cleared"):
                    state["status"] = "All reachable cubes sorted! Click 'Generate Random Cubes' for a new batch."
                env.settle(0.02)
                time.sleep(0.01)
    finally:
        server.shutdown()
        vw.close()


def run_sim(n_episodes, viewer, seed, margin=0.0025):
    from simulation.environment import SimEnvironment, SimCamera
    from deployment.simulation_interface import SimulationInterface
    chain = build_chain()
    env = SimEnvironment(ROOT / "simulation/scene.xml", ROBOT, WORKSPACE, CAMERA, chain, seed, plan_margin=margin)
    ik = IKSolver(chain)
    ik.is_valid = env.config_collision_free
    q_seed = (chain.lower + chain.upper) / 2
    sol = ik.solve(make_T(Rx(np.pi), WORKSPACE.observe_xyz), q_seed)
    assert sol is not None, "observe pose unreachable - edit WORKSPACE.observe_xyz"

    # Validate physical TCP approach/descent for cube picks and pad placements.
    # A multi-start solver prevents one poor seed from rejecting a reachable XY.
    spawn_ik = IKSolver(chain, n_random_starts=12, seed=17)
    spawn_ik.is_valid = env.config_collision_free

    def can_reach_xy(xy, purpose="pick"):
        z_tcp = (WORKSPACE.table_z + WORKSPACE.cube_size / 2
                 + WORKSPACE.tcp_to_cube_center_z)
        Tg = make_T(Rx(np.pi), [xy[0], xy[1], z_tcp])
        Ta = make_T(Rx(np.pi), [xy[0], xy[1], z_tcp + WORKSPACE.approach_height])

        # Require a collision-free pre-grasp and a collision-free vertical descent.
        for approach_sol in spawn_ik.solve_all(Ta, sol.q):
            if not env.config_collision_free(approach_sol.q):
                continue
            q_prev = approach_sol.q
            path_ok = True
            for step in range(1, 9):
                T_step = interpolate_pose(Ta, Tg, step / 8.0)
                step_sol = spawn_ik.solve_from(T_step, q_prev)
                if (not step_sol.converged
                        or step_sol.sigma_min < spawn_ik.sigma_thresh
                        or not env.config_collision_free(step_sol.q)):
                    path_ok = False
                    break
                q_prev = step_sol.q
            if path_ok:
                return True
        return False

    env.validator_fn = can_reach_xy
    import time

    print("[6/6] Starting environment reset...", flush=True)
    t0 = time.perf_counter()

    env.reset(sol.q)

    print(
        f"[6/6] Environment reset completed in {time.perf_counter() - t0:.2f}s",
        flush=True
    )
    ok_obs = env.config_collision_free(sol.q)
    print('clearance margin %.1f mm; observe pose passes the check: %s' % (margin * 1000, ok_obs))
    if not ok_obs:
        raise SystemExit('Lower the margin (--margin 0.0015).')

    robot = SimulationInterface(env, None)
    pnp = build_controller(robot, SimCamera(env), chain)
    pnp.q_observe = sol.q
    pnp.ik.is_valid = env.config_collision_free
    pnp.safety.collision_fn = env.config_collision_free

    if viewer:
        run_interactive_viewer(env, pnp, sol.q)
        return

    log = []
    for ep in range(n_episodes):
        import time

        print("[6/6] Starting environment reset...", flush=True)
        t0 = time.perf_counter()

        env.reset(sol.q)

        print(
            f"[6/6] Environment reset completed in {time.perf_counter() - t0:.2f}s",
            flush=True
        )
        gt_zones = env.get_zone_positions()
        t0 = time.time()
        dets, results = pnp.run()
        placed = 0
        for _, p, col in env.ground_truth_cubes():
            if p[0] < 4 and col in gt_zones and np.linalg.norm(p[:2] - gt_zones[col]) < 0.065:
                placed += 1
        from collections import Counter

        print("\\n========== PICK RESULTS ==========", flush=True)
        for i, res in enumerate(results, 1):
            status = "SUCCESS" if res.success else "FAILED"
            pos = np.asarray(res.cube_estimate).round(4).tolist() if res.cube_estimate is not None else None
            print(
                f"Cube {i}: {status} | color={res.color} | "
                f"stage={res.stage} | position={pos} | detail={res.detail}",
                flush=True
            )

        failures = Counter(res.stage for res in results if not res.success)
        print(f"Failure stages: {dict(failures)}", flush=True)

        log.append(dict(episode=ep, n_detected=len(dets), picks_attempted=len(results), placed=placed,
                        failures=dict(failures), seconds=time.time() - t0))
        print(f"ep {ep}: det={len(dets)} placed={placed}/{len(results)}", flush=True)
    (ROOT / "evaluation_results.json").write_text(json.dumps(log, indent=1))


def run_real(ip, speed):
    raise SystemExit("Real mode: configure RealVX500Camera and DobotInterface first.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sim", "real"], default="sim")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.0025, help="planning clearance in metres")
    ap.add_argument("--ip", default="192.168.5.1")
    a = ap.parse_args()
    run_sim(a.episodes, a.viewer, a.seed, a.margin) if a.mode == "sim" else run_real(a.ip, 0.1)

