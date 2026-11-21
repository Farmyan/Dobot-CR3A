"""Dobot cra_description URDF(xacro) -> clean assets/urdf/cr3a.urdf, inspect, patch config.py, build MJCF."""
import re, shutil, subprocess, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
model = sys.argv[1] if len(sys.argv) > 1 else "cr3"
desc = ROOT / "_dobot_repo" / "cra_description"
src = desc / "urdf" / f"{model}_robot.xacro"
if not src.exists():
    sys.exit(f"missing {src}")

text = src.read_text(encoding="utf-8")
text = text.replace("file://$(find cra_description)", "package://cra_description")
text = re.sub(r'\sxmlns:xacro="[^"]*"', "", text)
root = ET.fromstring(text.encode("utf-8"))

# drop simulator-specific blocks
for tag in ("gazebo", "ros2_control"):
    for el in root.findall(tag):
        root.remove(el)
# drop world/dummy mounting links so base_link is the root (base frame origin = base mounting plane)
for j in root.findall("joint"):
    if j.get("name") in ("world_joint", "dummy_joint"):
        root.remove(j)
for l in root.findall("link"):
    if l.get("name") in ("world", "dummy_link"):
        root.remove(l)
# drop empty-named materials
for l in root.findall("link"):
    for v in l.findall("visual"):
        for m in v.findall("material"):
            v.remove(m)
# velocity limits (file has 0): use Dobot MoveIt joint_limits.yaml values
vel = {"joint1": 3.14, "joint2": 3.14, "joint3": 3.89, "joint4": 3.89, "joint5": 3.89, "joint6": 3.89}
for j in root.findall("joint"):
    lim = j.find("limit")
    if lim is not None and j.get("name") in vel:
        lim.set("velocity", str(vel[j.get("name")]))
        lim.set("effort", "100")

out = ROOT / "assets" / "urdf" / "cr3a.urdf"
ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)
shutil.copytree(desc / "meshes" / model, ROOT / "assets" / "meshes" / model, dirs_exist_ok=True)
print("wrote", out)

from kinematics.forward_kinematics import SerialChain, parse_urdf_joints
joints = parse_urdf_joints(out)
parents = {j.parent for j in joints}
children = {j.child for j in joints}
roots, leaves = sorted(parents - children), sorted(children - parents)
print("root links:", roots, " leaf links:", leaves)
if len(roots) != 1 or len(leaves) != 1:
    sys.exit("unexpected tree shape - send me this output")
base, flange = roots[0], leaves[0]

print(f"\n{'joint':10s}{'type':10s}{'lower':>8s}{'upper':>8s}{'vel':>7s}   origin xyz")
for j in joints:
    print(f"{j.name:10s}{j.type:10s}{j.lower:8.3f}{j.upper:8.3f}{(j.velocity or 0):7.2f}   {np.round(j.origin[:3, 3], 4)}")

chain = SerialChain.from_urdf(out, base, flange)
rng = np.random.default_rng(0)
shoulder = chain.fk(np.zeros(chain.n), True)[1][1][1]
reach = max(float(np.linalg.norm(chain.fk(rng.uniform(chain.lower, chain.upper))[:3, 3] - shoulder))
            for _ in range(20000))
print(f"\nmax shoulder-to-flange distance over joint limits: {reach*1000:.0f} mm  (CR3A working radius: 620 mm)")

cfg = ROOT / "config.py"
c = cfg.read_text(encoding="utf-8")
c = re.sub(r'(base_link: str = ")[^"]*(")', lambda m: m.group(1) + base + m.group(2), c)
c = re.sub(r'(flange_link: str = ")[^"]*(")', lambda m: m.group(1) + flange + m.group(2), c)
c = c.replace("default_amax: float = 6.0 ", "default_amax: float = 6.28")
cfg.write_text(c, encoding="utf-8")
print(f"config.py updated: base_link={base}, flange_link={flange}")

print("\n=== building MuJoCo robot (tools/prepare_assets.py) ===")
r = subprocess.run([sys.executable, str(ROOT / "tools" / "prepare_assets.py"), "--urdf", str(out),
                    "--base", base, "--flange", flange], cwd=ROOT)
print("prepare_assets exit code:", r.returncode)
