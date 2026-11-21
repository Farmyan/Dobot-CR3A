# DOBOT CR3A Autonomous Pick-and-Place (MuJoCo → real robot)

Mathematical (non-learning) manipulation: URDF-based FK, numerical IK, camera ray-plane
localisation, top-down grasp planning, quintic trajectories, safety validation, and a
hardware-agnostic robot interface.

## Status — read first
| Part | Status |
|---|---|
| Kinematics, IK, perception geometry, detector, trajectories, safety | **Unit-tested** on a mock 6-DOF arm (`tests/test_all.py`, all pass) |
| MuJoCo scene/environment, asset conversion, evaluation loop | **Written but NOT run** (no MuJoCo in the authoring sandbox). Expect small fixes on first run |
| `DobotInterface` (real robot) | **Unverified.** Command strings must be checked against your controller's TCP/IP V4 protocol PDF |
| CR3A URDF/meshes, real gripper, VX500 intrinsics, hand-eye calibration | **Not included — you must supply** |

## Setup
1. Get the CR3A URDF + STL meshes (Dobot-Arm/DOBOT_6Axis_ROS2_V4 → `cra_description`; look for a
   CR3A/`cr3a` model; if only `cr3` exists, it is the older robot — do not use it). If the file is
   `.xacro`: `xacro cr3a.urdf.xacro > cr3a.urdf`.
2. Put it in `assets/urdf/` (meshes beside it or under `assets/`).
3. Edit `config.py`: `base_link`, `flange_link` (check names in the URDF), TCP offset, workspace.
4. `pip install mujoco numpy opencv-python`
5. `python tools/prepare_assets.py --urdf assets/urdf/cr3a.urdf --base <base> --flange <flange>`
   (prints joints + limits — compare with the CR3A datasheet; writes `simulation/robot.xml`).
6. `python tests/validate_fk_vs_mujoco.py` (expect ≪ 0.1 mm).
7. `python main.py --mode sim --episodes 100 --viewer` → writes `evaluation_results.json`.

## Frames and conventions
- `base`: robot base frame (= MuJoCo world; table top at `z = WORKSPACE.table_z`).
- `cam`: OpenCV convention (x right, y down, z forward). `P_base = T_base_cam · P_cam`.
  MuJoCo cameras look along −z with y up; `SimCamera` converts with `diag(1,−1,−1)`.
- TCP: flange frame + `tcp_xyz/rpy`. Grasp convention: TCP z points **down**, `R = Rz(yaw)·Rx(π)`.
  If your real TCP frame differs, change `grasp_pose()` only.

## Methods
- **FK**: `T = Π (origin_i · Rot(axis_i, q_i)) · T_tcp`, parsed from the URDF — the same kinematics MuJoCo uses.
- **IK**: damped-least-squares on the 6D pose error (position + rotation-vector), step clamping,
  joint-limit clamping, adaptive damping. Multi-start (`solve_all`) enumerates distinct configurations;
  solutions with σ_min(J) below a threshold (singular) or residual above tolerance (unreachable) are rejected;
  the solution closest to the current joints is chosen for continuity.
- **Localisation**: cube top face centroid (colour mask → `minAreaRect`) → ray from camera centre through the
  pixel → intersection with plane `z = table_z + cube_size` → cube centre. Yaw from the rectangle edge mapped to
  the plane, folded by the cube's 90° symmetry. Ground truth is used only in the evaluation.
- **Trajectories**: `q(t)=q0+s(t)(qf−q0)`, `s=10τ³−15τ⁴+6τ⁵` (zero velocity/acceleration at ends); duration
  from `T ≥ 1.875|Δq|/v_max`, `T ≥ √(5.77|Δq|/a_max)`. Approach/lift/descend are straight Cartesian lines solved
  by IK waypoint-by-waypoint with a branch-jump check.
- **Safety** (`deployment/safety.py`): joint limits (+margin), velocity, acceleration, TCP workspace box, link
  clearance above the table; runs before every motion in both modes.
- Collision in sim is *monitored* (robot vs table, non-gripper vs cube) and counted; planning-time collision
  checking is the conservative box/clearance test above. For cluttered scenes add a proper checker (e.g. MuJoCo
  contact queries on candidate waypoints).

## Architecture
```
Sim : SimCamera → CubeDetector → grasp_planning → IKSolver → trajectory_planning → SafetyValidator → SimulationInterface(MuJoCo)
Real: VX500 camera → (same modules)                                             → SafetyValidator → DobotInterface(TCP/IP)
```
`control/pick_and_place.py` only uses `RobotInterface` and `Camera` abstractions.

## Placeholders you must replace
- Gripper (`assets/gripper/gripper.xml`, `GripperConfig`, `tcp_xyz`): real stroke, finger geometry, TCP offset.
- Camera (`CameraConfig`, `scene.xml` camera): VX500 resolution/intrinsics (calibrate with a checkerboard),
  mounting pose; real hand-eye calibration `T_base_cam`.
- `default_amax`, joint sign/offset (`RobotConfig`), actuator gains in `prepare_assets.py`.
- Cube/place/spawn positions must lie inside the CR3A's 620 mm reach and the camera view.

## Deployment checklist (do in order, robot at ≤10 % speed, E-stop in hand)
1. Compare URDF joint names/limits/zero positions with the real robot; set `joint_sign`/`joint_offset_deg`.
2. Verify `GetAngle` readback matches simulated `fk` TCP: jog the robot, compare reported pose vs `chain.fk`.
3. Calibrate camera intrinsics and hand-eye; check by projecting a known marker to the base frame (< 5 mm).
4. Verify every `DobotInterface.COMMANDS` entry against the protocol PDF; test `stop()` and error handling.
5. Dry-run trajectories with no objects, reduced speed (`speed_scale`), check collision-free.
6. Test E-stop and protective stop; test gripper open/close.
7. Supervised single-cube grasps, then autonomous runs.
