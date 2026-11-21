import importlib.util
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = Path.home() / "AG95_test" / "ag95_gripper.py"

spec = importlib.util.spec_from_file_location("ag95_generator", source)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

# Original AG95 mounting orientation from its URDF:
# rpy = 1.57075 0 1.57075
mount_quat = g.rpy_to_quat(1.57075, 0.0, 1.57075)

# Roll the entire AG95 by +90 degrees around its local Z axis.
roll90 = g.rpy_to_quat(1.38, 1.5, -9.57079632679)
w1, x1, y1, z1 = map(float, mount_quat.split())
w2, x2, y2, z2 = map(float, roll90.split())

mount_quat = " ".join(f"{v:.9f}" for v in (
    w1*w2 - x1*x2 - y1*y2 - z1*z2,
    w1*x2 + x1*w2 + y1*z2 - z1*y2,
    w1*y2 - x1*z2 + y1*w2 + z1*x2,
    w1*z2 + x1*y2 - y1*x2 + z1*w2,
))

body = g.ag95_body_xml(
    name="ag95",
    pos="0 0 0",
    quat=mount_quat
).strip()

# Preserve the TCP site name used by the CR3A application.
body = body.rsplit("</body>", 1)[0]
body += '<site name="tcp" pos="0 0 0.16" size="0.004" rgba="1 0 0 1"/>\n</body>'

mesh_files = {
    "base": "ag95_base_link.STL",
    "crank": "ag95_crank_Link.STL",
    "rod": "ag95_rod_Link.STL",
    "prox": "ag95_proximal_phalanx_Link.STL",
    "distal": "ag95_distal_phalanx_Link.STL",
}

assets = "\n".join(
    f'<mesh name="ag95_{key}" file="{filename}" scale="0.001 0.001 0.001"/>'
    for key, filename in mesh_files.items()
)

xml = f"""<mujoco model="ag95_gripper">
    <asset>
        {assets}
    </asset>
    <worldbody>
        {body}
    </worldbody>
    <equality>
        {g.ag95_equality_xml()}
    </equality>
</mujoco>
"""

output = root / "assets" / "gripper" / "gripper.xml"
output.write_text(xml, encoding="utf-8")

print("Generated:", output)
print("Mount quaternion:", mount_quat)
print("AG95 drive joint: ag95_finger1_joint")
print("AG95 maximum drive angle:", g.CLOSE)


