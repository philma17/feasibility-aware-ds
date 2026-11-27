"""Generate the additional FR3 MJCFs from the original arm.

Run ``python -B -m tools.generate_fr3_variants`` to write, or add ``--check`` to verify.
The variants use the existing shared assets/ meshes.
"""

import argparse
from decimal import Decimal
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.fr3_variants import FIXED_VARIANTS, MASS_VARIANTS


MODEL_DIR = Path(__file__).resolve().parents[1] / "vendor" / "franka_fr3"


def preserve_self_contacts(text, locked):
    """Preserve the full FR3's non-neighbour collision pairs after welding links."""
    # Link 0 is welded to the world. Each retained hinge starts a new rigid body.
    group = np.cumsum([0] + [joint not in locked for joint in range(1, 8)])
    rules = []
    for i in range(7):
        for j in range(i + 1, 8):
            # MuJoCo skips same-group pairs and adjacent non-world groups.
            filtered = group[i] == group[j] or (group[i] > 0 and group[j] == group[i] + 1)
            if j > i + 1 and filtered:
                rules.append(f'    <pair geom1="fr3_link{i}_collision" geom2="fr3_link{j}_collision"/>')
            elif j == i + 1 and i > 0 and not filtered:
                rules.append(f'    <exclude body1="fr3_link{i}" body2="fr3_link{j}"/>')
    return text.replace("  </contact>", "\n".join([*rules, "  </contact>"]))


def fixed_xml(source, model, name, locked):
    """Bake hinge midpoint rotations about their local anchors before removal."""
    text = source.replace('model="fr3"', f'model="{name}"', 1)
    for number in locked:
        joint = f"fr3_joint{number}"
        jid = model.joint(joint).id
        angle = float(model.jnt_range[jid].mean() - model.qpos0[model.jnt_qposadr[jid]])
        if angle != 0:
            bid = int(model.jnt_bodyid[jid])
            delta, quat, matrix = np.zeros(4), np.zeros(4), np.zeros(9)
            mujoco.mju_axisAngle2Quat(delta, model.jnt_axis[jid], angle)
            # Joint rotation is local to the body: compose it after the body rotation.
            mujoco.mju_mulQuat(quat, model.body_quat[bid], delta)
            mujoco.mju_quat2Mat(matrix, model.body_quat[bid])
            original_rotation = matrix.reshape(3, 3).copy()
            mujoco.mju_quat2Mat(matrix, delta)
            anchor = model.jnt_pos[jid]
            # Rotate about the hinge anchor, which need not be the body origin.
            pos = model.body_pos[bid] + original_rotation @ (anchor - matrix.reshape(3, 3) @ anchor)

            def transform(match):
                body = ET.fromstring(match[0] + "</body>")
                body.set("pos", " ".join(format(v, ".17g") for v in pos))
                body.set("quat", " ".join(format(v, ".17g") for v in quat))
                return ET.tostring(body, encoding="unicode", short_empty_elements=False).split("</body>")[0]

            text = re.sub(rf'<body\b[^>]*name="{model.body(bid).name}"[^>]*>', transform, text, count=1)
        text = re.sub(rf'\n[ \t]*<joint\b[^>]*name="{joint}"[^>]*/>', "", text, count=1)
        text = re.sub(rf'\n[ \t]*<position\b[^>]*joint="{joint}"[^>]*/>', "", text, count=1)
    keep = [i for i in range(7) if i + 1 not in locked]

    def keyframe(match):
        return match[1] + " ".join(match[2].split()[i] for i in keep) + '"'

    return preserve_self_contacts(re.sub(r'((?:qpos|ctrl)=")([^"]*)"', keyframe, text), locked)


def mass_xml(source, name, link, factor):
    """Change exactly one inertial's density, retaining its COM frame verbatim."""
    text = source.replace('model="fr3"', f'model="{name}"', 1)

    def scale(match):
        # Preserve exact decimal products of the source MJCF's mass and inertia.
        return match[1] + " ".join(
            str(Decimal(v) * Decimal(str(factor))) for v in match[2].split()
        ) + '"'

    def inertial(match):
        return re.sub(r'((?:mass|diaginertia)=")([^"]*)"', scale, match[0])

    return re.sub(rf'<body\b[^>]*name="{link}"[^>]*>\s*<inertial\b[^>]*/>',
                  inertial, text, count=1)


def generated_models():
    source = (MODEL_DIR / "fr3.xml").read_text()
    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "fr3.xml"))
    variants = {
        filename: fixed_xml(source, model, name, locked)
        for name, (filename, locked) in FIXED_VARIANTS.items()
    }
    variants.update({
        filename: mass_xml(source, name, link, factor)
        for name, (filename, link, factor) in MASS_VARIANTS.items()
    })
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for filename, text in generated_models().items():
        path = MODEL_DIR / filename
        if args.check:
            if not path.is_file() or path.read_text() != text:
                raise ValueError(f"Generated MJCF differs: {path}")
        else:
            path.write_text(text)
        print(f"{'Verified' if args.check else 'Wrote'} {path}")


if __name__ == "__main__":
    main()
