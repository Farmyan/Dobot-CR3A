from pathlib import Path

p = Path("control/trajectory_planning.py")
s = p.read_text(encoding="utf-8")
old = '''    qs = np.array(qs)
    seg = np.abs(np.diff(qs, axis=0)).sum(axis=0)
    T = min_duration(seg, vmax, amax)
    tw = np.linspace(0, T, n)
    t = np.arange(0, T + dt / 2, dt)
    t[-1] = T
    s = quintic(t / T)
    q = np.stack([np.interp(s * T, tw, qs[:, i]) for i in range(qs.shape[1])], axis=1)
    return Trajectory(t, q, dt)
'''
new = '''    qs = np.array(qs)
    seg = np.abs(np.diff(qs, axis=0)).sum(axis=0)
    T = min_duration(seg, vmax, amax)
    for _ in range(10):                      # stretch the duration until v and a limits really hold
        tw = np.linspace(0, T, n)
        t = np.arange(0, T + dt / 2, dt)
        t[-1] = T
        s = quintic(t / T)
        q = np.stack([np.interp(s * T, tw, qs[:, i]) for i in range(qs.shape[1])], axis=1)
        v = np.diff(q, axis=0) / dt
        a = np.diff(v, axis=0) / dt
        f = max(1.0,
                float(np.max(np.abs(v) / vmax)) if len(v) else 1.0,
                float(np.max(np.abs(a) / amax)) if len(a) else 1.0)
        if f <= 1.0:
            break
        T *= 1.05 * f
    if T > 8.0:
        raise PlanningError("ik_line", "Cartesian segment needs an unreasonably jerky joint path")
    return Trajectory(t, q, dt)
'''
print(("ok      " if s.count(old) == 1 else "PROBLEM(%d) " % s.count(old)) + "control/trajectory_planning.py: line time scaling")
p.write_text(s.replace(old, new), encoding="utf-8")

p = Path("main.py")
s = p.read_text(encoding="utf-8")
old = '''        placed = 0
        for k, r in enumerate(results):
            slot = np.array([WORKSPACE.place_xy[0] + k * WORKSPACE.place_spacing, WORKSPACE.place_xy[1]])
            for n, p, _ in env.ground_truth_cubes():
                if np.linalg.norm(p[:2] - slot) < 0.015 and p[2] < WORKSPACE.cube_size:
                    placed += 1
                    break
'''
new = '''        placed, notes = 0, []
        for k, r in enumerate(results):
            slot = np.array([WORKSPACE.place_xy[0] + k * WORKSPACE.place_spacing, WORKSPACE.place_xy[1]])
            cubes = [p for n, p, _ in env.ground_truth_cubes() if p[0] < 4]
            ok = any(np.linalg.norm(p[:2] - slot) < 0.015 and p[2] < WORKSPACE.cube_size for p in cubes)
            placed += int(ok)
            if r.success and not ok:
                dmin = min(float(np.linalg.norm(p[:2] - slot)) for p in cubes)
                notes.append("slot %d: pick reported success but no cube within 15 mm (nearest %.0f mm)" % (k, dmin * 1000))
'''
print(("ok      " if s.count(old) == 1 else "PROBLEM(%d) " % s.count(old)) + "main.py: placement check")
s = s.replace(old, new)
old = "placed=placed,"
print(("ok      " if s.count(old) == 1 else "PROBLEM(%d) " % s.count(old)) + "main.py: log notes")
s = s.replace(old, "placed=placed, notes=notes,")
p.write_text(s, encoding="utf-8")
