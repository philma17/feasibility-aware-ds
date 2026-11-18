"""Compare full-DoF FR3 density variants at one identical joint configuration."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from main import load_config
from sim.fr3_variants import MASS_VARIANTS
from sim.robot import FR3, gravity_torque, resolve_plant
from sim.sim import build_case


def compare(cfg, q=None):
    """Use the runner's actual plant/nominal pair and existing gravity routine."""
    cfg = deepcopy(cfg)
    cfg["robot"] = "FR3"
    if q is not None:
        cfg["task"]["q_start_rad"] = list(q)
    plant = resolve_plant(cfg.get("plant"))
    if any(plant[key] != 1.0 for key in ("mass_scale", "inertia_scale", "gravity_scale")) or plant["com_shift"]:
        raise ValueError("Mass comparison requires unit mass/inertia/gravity scales and no COM shift")
    reference = build_case(cfg)
    q = reference.q_start.copy()
    cfg["task"]["q_start_rad"] = q.tolist()
    g0 = gravity_torque(reference.robot, q)
    total0 = reference.robot.model.body_mass.sum()
    rows = []
    for name in ("FR3", *MASS_VARIANTS):
        cfg["robot"] = name
        case = reference if name == "FR3" else build_case(cfg)
        nominal = case.robot_nominal
        matched = (
            case.robot.spec.mjcf == nominal.spec.mjcf
            and np.array_equal(case.robot.model.body_mass, nominal.model.body_mass)
            and np.array_equal(case.robot.model.body_inertia, nominal.model.body_inertia)
            and np.array_equal(case.robot.model.body_ipos, nominal.model.body_ipos)
            and np.array_equal(case.robot.model.opt.gravity, nominal.model.opt.gravity)
        )
        if not matched:
            raise ValueError(f"{name}: nominal compensation differs from the selected MJCF plant")
        g = gravity_torque(case.robot, q)
        link = MASS_VARIANTS[name][1] if name in MASS_VARIANTS else None
        bid = case.robot.model.body(link).id if link else None
        rows.append({
            "robot": name,
            "mjcf": f"vendor/{case.robot.spec.mjcf}",
            "changed_link": link,
            "link_mass_kg": float(case.robot.model.body_mass[bid]) if link else None,
            "added_mass_kg": float(case.robot.model.body_mass.sum() - total0),
            "gravity_torque_Nm": g.tolist(),
            "delta_gravity_torque_Nm": (g - g0).tolist(),
            "nominal_mjcf": f"vendor/{nominal.spec.mjcf}",
            "nominal_matches_plant": matched,
            "nominal_vs_plant_gravity_max_abs_Nm": float(np.max(np.abs(gravity_torque(nominal, q) - g))),
        })
    return {"q_rad": q.tolist(), "joint_names": list(FR3.joints), "variants": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/five_obstacle_pic.yaml")
    parser.add_argument("--q", nargs=7, type=float, help="J1..J7 [rad]; default: reference runner start")
    parser.add_argument("--output", type=Path, help="also save the complete JSON report")
    args = parser.parse_args(argv)
    report = compare(load_config(args.config), args.q)
    print("q [rad], J1..J7:", report["q_rad"])
    for row in report["variants"]:
        print(f"{row['robot']}: added mass = {row['added_mass_kg']:.9g} kg")
        print("  g [Nm]:", " ".join(f"{v:.9f}" for v in row["gravity_torque_Nm"]))
        print("  Δg [Nm]:", " ".join(f"{v:.9f}" for v in row["delta_gravity_torque_Nm"]))
        print(f"  nominal = plant: {row['nominal_matches_plant']}; MJCF: {row['mjcf']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
