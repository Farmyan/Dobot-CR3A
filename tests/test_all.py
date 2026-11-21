"""Run: python -m pytest tests -q   (numpy/opencv only; no MuJoCo needed)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import cv2
from kinematics.forward_kinematics import SerialChain
from kinematics.inverse_kinematics import IKSolver
from kinematics.transformations import *
from perception.calibration import Intrinsics, project_base_to_pixel, pixel_to_base_on_plane
from perception.object_detection import CubeDetector, fold_yaw
from control.trajectory_planning import plan_joint, plan_cartesian_line
from control.grasp_planning import grasp_pose
from deployment.safety import SafetyValidator
from config import WORKSPACE

URDF = pathlib.Path(__file__).parent / "mock_arm.urdf"
chain = SerialChain.from_urdf(URDF, "base_link", "flange", make_T(None, [0, 0, 0.05]))


def test_fk_known_pose():
    T = chain.fk(np.zeros(6))      # straight up: 0.14+0.28+0.25+0.06+0.05+0.06+0.05
    assert np.allclose(T[:3, 3], [0, 0, 0.89], atol=1e-9)
    T = chain.fk(np.array([np.pi / 2, 0, 0, 0, 0, 0]))
    assert np.allclose(T[:3, 3], [0, 0, 0.89], atol=1e-9)


def test_rotation_log_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        w = rng.normal(size=3)
        w *= rng.uniform(0, 3.1) / np.linalg.norm(w)
        assert np.allclose(R_to_rotvec(rotvec_to_R(w)), w, atol=1e-7)


def test_jacobian_matches_finite_difference():
    q = np.array([0.3, 0.4, -0.5, 0.2, 0.7, -0.3])
    J = chain.jacobian(q)
    for i in range(6):
        dq = np.zeros(6); dq[i] = 1e-6
        d = pose_error(chain.fk(q), chain.fk(q + dq)) / 1e-6
        assert np.allclose(J[:, i], d, atol=1e-4)


def test_ik_roundtrip_and_limits():
    ik = IKSolver(chain)
    rng = np.random.default_rng(3)
    ok = 0
    for _ in range(30):
        q = rng.uniform(chain.lower * 0.6, chain.upper * 0.6)
        T = chain.fk(q)
        sol = ik.solve(T, q + rng.normal(scale=0.3, size=6))
        if sol is not None:
            ok += 1
            assert np.all(sol.q >= chain.lower) and np.all(sol.q <= chain.upper)
            assert np.linalg.norm(pose_error(chain.fk(sol.q), T)[:3]) < 3e-4
    assert ok >= 27


def test_ik_unreachable_returns_none():
    assert IKSolver(chain).solve(make_T(Rx(np.pi), [2.0, 0, 0.1]), np.zeros(6)) is None


def test_ik_multiple_solutions():
    q = np.array([0.2, 0.5, 0.8, 0.3, 0.4, 0.5])
    sols = IKSolver(chain, n_random_starts=40).solve_all(chain.fk(q), q)
    assert len(sols) >= 2


def test_topdown_grasp_ik():
    ik = IKSolver(chain)
    T = grasp_pose(np.array([0.30, 0.10, 0.0125]), 0.3)
    assert ik.solve(T, (chain.lower + chain.upper) / 2) is not None


def test_ray_plane_localisation_sub_mm():
    intr = Intrinsics.from_fovy(45, 1280, 960)
    T = np.eye(4); T[:3, :3] = np.diag([1, -1, -1]); T[:3, 3] = [0.35, 0, 0.8]
    p = np.array([0.31, 0.07, 0.025])
    uv = project_base_to_pixel(p, intr, T)
    assert np.linalg.norm(pixel_to_base_on_plane(uv, intr, T, 0.025) - p) < 1e-9


def test_detector_on_synthetic_image():
    intr = Intrinsics.from_fovy(45, 1280, 960)
    T = np.eye(4); T[:3, :3] = np.diag([1, -1, -1]); T[:3, 3] = [0.35, 0, 0.8]
    img = np.full((960, 1280, 3), 190, np.uint8)
    c, yaw, h = np.array([0.33, 0.06]), 0.4, 0.0125
    corners = [c + Rz(yaw)[:2, :2] @ np.array(s) * h for s in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    px = np.array([project_base_to_pixel([*k, 0.025], intr, T) for k in corners], np.int32)
    cv2.fillPoly(img, [px], (230, 20, 20))
    dets = CubeDetector(0.025, 0.0).detect(img, intr, T)
    assert len(dets) == 1
    assert np.linalg.norm(dets[0].position[:2] - c) < 0.002
    assert abs(fold_yaw(dets[0].yaw - yaw)) < np.radians(3) or abs(abs(fold_yaw(dets[0].yaw - yaw)) - np.pi/2) < np.radians(3)


def test_trajectory_boundaries_and_limits():
    v, a = np.full(6, 1.0), np.full(6, 2.0)
    tr = plan_joint(np.zeros(6), np.array([1, -1, .5, .2, 0, 1.5]), v, a, dt=0.002)
    assert np.allclose(tr.q[0], 0) and np.allclose(tr.q[-1], [1, -1, .5, .2, 0, 1.5])
    vel = np.diff(tr.q, axis=0) / tr.dt
    acc = np.diff(vel, axis=0) / tr.dt
    assert np.abs(vel).max() <= 1.0 * 1.001 and np.abs(acc).max() <= 2.0 * 1.01
    assert np.allclose(vel[0], 0, atol=1e-3) and np.allclose(vel[-1], 0, atol=1e-3)


def test_cartesian_line_and_safety():
    ik = IKSolver(chain)
    q0 = ik.solve(grasp_pose(np.array([0.3, 0.1, 0.09]), 0.0), (chain.lower + chain.upper) / 2).q
    T0, T1 = chain.fk(q0), grasp_pose(np.array([0.3, 0.1, 0.0125]), 0.0)
    v, a = np.full(6, 1.0), np.full(6, 2.0)
    tr = plan_cartesian_line(ik, T0, T1, q0, v, a)
    sv = SafetyValidator(chain, WORKSPACE, v, a)
    assert sv.validate(tr, allow_low_tcp=True).ok
    assert sv.validate(tr).ok                               # grasp height is above the z floor
    T_low = grasp_pose(np.array([0.3, 0.1, 0.001]), 0.0)
    tr_low = plan_cartesian_line(ik, T0, T_low, q0, v, a)
    assert not sv.validate(tr_low, allow_low_tcp=False).ok  # below normal TCP floor => rejected
    assert sv.validate(tr_low, allow_low_tcp=True).ok
    bad = plan_joint(q0, q0 + 10, v, a)
    assert not sv.validate(bad).ok
