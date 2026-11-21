"""CR3A URDF -> MuJoCo MJCF with integrated AG95 gripper."""
import argparse
import copy
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def flatten_meshes(urdf_path, mesh_dir):
    text = Path(urdf_path).read_text(encoding="utf-8")
    mesh_dir.mkdir(parents=True, exist_ok=True)

    def repl(m):
        ref = m.group(1)
        rel = re.sub(r"^package://[^/]+/", "", ref)
        cands = [
            Path(urdf_path).parent / rel,
            Path(urdf_path).parent.parent / rel,
            Path(urdf_path).parent / Path(rel).name,
            ROOT / "assets" / rel,
        ]
        src = next((c for c in cands if c.exists()), None)

        if src is None:
            sys.exit(f"mesh not found: {ref}")

        flat = "_".join(Path(rel).parts[-2:])
        shutil.copy(src, mesh_dir / flat)
        return f'filename="{flat}"'

    return re.sub(r'filename="([^"]+)"', repl, text)


def get_or_create(parent, tag):
    element = parent.find(tag)
    if element is None:
        element = ET.SubElement(parent, tag)
    return element


def main():
    import mujoco

    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=str(ROOT / "assets/urdf/cr3a.urdf"))
    ap.add_argument("--base", default="base_link")
    ap.add_argument("--flange", default="Link6")
    ap.add_argument("--kp", type=float, default=1000.0)
    ap.add_argument("--kv", type=float, default=60.0)
    a = ap.parse_args()

    mesh_dir = ROOT / "assets/meshes"

    # Prepare and compile the CR3A URDF.
    text = flatten_meshes(a.urdf, mesh_dir)

    if "<mujoco>" not in text:
        inject = (
            r'\1<mujoco><compiler meshdir="%s" '
            r'balanceinertia="true" discardvisual="false" '
            r'fusestatic="false"/></mujoco>' % mesh_dir.as_posix()
        )
        text = re.sub(r"(<robot[^>]*>)", inject, text, count=1)

    tmp = ROOT / "assets/urdf/_prepared.urdf"
    tmp.write_text(text, encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(tmp))

    raw = ROOT / "simulation/_robot_raw.xml"
    mujoco.mj_saveLastXML(str(raw), model)

    tree = ET.parse(raw)
    root = tree.getroot()

    # The scene.xml file owns compiler settings.
    for comp in root.findall("compiler"):
        root.remove(comp)

    # Collect CR3A arm joints before adding the gripper.
    print(f"{'joint':14s} {'lower':>8s} {'upper':>8s}   (rad)")
    arm_joints = []

    for j in root.iter("joint"):
        if j.get("type", "hinge") == "hinge":
            lo, hi = (j.get("range") or "-6.283 6.283").split()
            arm_joints.append((j.get("name"), float(lo), float(hi)))
            j.set("armature", "0.1")
            print(f"{j.get('name'):14s} {float(lo):8.3f} {float(hi):8.3f}")

    # Gravity compensation.
    for b in root.iter("body"):
        b.set("gravcomp", "1")

    # Locate the CR3A flange.
    flange = next(
        (b for b in root.iter("body") if b.get("name") == a.flange),
        None
    )

    if flange is None:
        names = [b.get("name") for b in root.iter("body")]
        sys.exit(f"Flange '{a.flange}' not found. Bodies: {names}")

    # Load the generated AG95 model.
    ag95_path = ROOT / "assets/gripper/gripper.xml"

    if not ag95_path.exists():
        sys.exit(f"AG95 model not found: {ag95_path}")

    ag95_root = ET.parse(ag95_path).getroot()

    # Transfer AG95 mesh assets into the robot model.
    ag95_assets = ag95_root.find("asset")

    if ag95_assets is not None:
        assets = get_or_create(root, "asset")

        existing_meshes = {
            item.get("name")
            for item in assets.findall("mesh")
        }

        for item in ag95_assets.findall("mesh"):
            if item.get("name") in existing_meshes:
                sys.exit(f"Duplicate mesh asset: {item.get('name')}")

            assets.append(copy.deepcopy(item))

    # Extract the AG95 root body.
    ag95_body = ag95_root.find("./worldbody/body[@name='ag95']")

    if ag95_body is None:
        sys.exit("AG95 root body was not found in gripper.xml.")

    ag95_body = copy.deepcopy(ag95_body)

    # Preserve the AG95 mounting position and quaternion.
    # They reproduce the original URDF gripper mounting transform.
    flange.append(ag95_body)

    # Transfer AG95 mechanical mimic constraints.
    ag95_equality = ag95_root.find("equality")

    if ag95_equality is not None:
        equality = get_or_create(root, "equality")

        for item in ag95_equality:
            equality.append(copy.deepcopy(item))

    # Add CR3A arm actuators.
    actuator = get_or_create(root, "actuator")

    for n, lo, hi in arm_joints:
        ET.SubElement(
            actuator,
            "position",
            name=f"act_{n}",
            joint=n,
            kp=str(a.kp),
            kv=str(a.kv),
            ctrlrange=f"{lo} {hi}",
        )

    # Add the AG95 drive actuator.
    ET.SubElement(
        actuator,
        "position",
        name="gripper",
        joint="ag95_finger1_joint",
        kp="30",
        kv="1",
        ctrlrange="0 0.6524",
        forcerange="-1.5 1.5",
    )

    # Save the integrated robot model.
    output = ROOT / "simulation/robot.xml"
    tree.write(output, encoding="unicode")

    raw.unlink(missing_ok=True)

    # Verify that the generated MJCF actually loads.
    check = mujoco.MjModel.from_xml_path(str(ROOT / "simulation/scene.xml"))

    print()
    print("AG95 integration successful.")
    print(f"Generated: {output}")
    print(f"Robot joints: {check.njnt}")
    print(f"Robot actuators: {check.nu}")

    for name in (
        "ag95_finger1_joint",
        "ag95_finger2_joint",
    ):
        check.joint(name)
        print(f"  Joint OK: {name}")

    check.actuator("gripper")
    print("  Actuator OK: gripper")
    print("Complete CR3A + AG95 model loaded successfully.")


if __name__ == "__main__":
    main()
