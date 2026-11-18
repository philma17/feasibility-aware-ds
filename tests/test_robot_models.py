"""Check hard-fixed MJCF arms, physical references, and the public run paths."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import main
from evaluation.aggregate import aggregate
from sim.batch import plan_starts
from sim.fr3_variants import MASS_VARIANTS, generated_models, MODEL_DIR
from sim.robot import (
    FR3, MENAGERIE_DIR, ROBOTS, Robot, apply_plant, gravity_torque,
    position_servo_map, torque_limit,
)


class RobotModelTests(unittest.TestCase):
    def test_compilation_joint_names_limits_and_keyframes(self):
        counts = {"FR3": 7, "FR3_fixed_J5": 6, "FR3_fixed_J3": 6,
                  "FR3_fixed_J3_J5": 5, "FR3_fixed_J3_J7": 5,
                  "FR3_fixed_J3_J5_J7": 4, "FR3_fixed_J3_J5_J6_J7": 3,
                  **{name: 7 for name in MASS_VARIANTS}, "UR5e": 6}
        self.assertEqual(set(counts), set(ROBOTS))
        for name, count in counts.items():
            with self.subTest(robot=name):
                robot = Robot(ROBOTS[name])
                m = robot.model
                self.assertEqual((robot.n, m.nq, m.nv, m.nu), (count,) * 4)
                self.assertEqual(m.neq, 0)  # Hard removal, not a solver constraint.
                self.assertEqual(tuple(m.joint(i).name for i in range(m.njnt)), robot.spec.joints)
                self.assertEqual(m.key_qpos.shape, (1, count))
                self.assertEqual(m.key_ctrl.shape, (1, count))
                sv = position_servo_map(robot)
                np.testing.assert_array_equal(m.key_ctrl[0, sv.act], robot.q_home)
                np.testing.assert_allclose(robot.q_min, robot.spec.q_min_doc, atol=1e-4)
                np.testing.assert_allclose(robot.q_max, robot.spec.q_max_doc, atol=1e-4)
                if robot.spec.family == "FR3":
                    indices = [FR3.joints.index(j) for j in robot.spec.joints]
                    np.testing.assert_array_equal(torque_limit(name, 1), torque_limit("FR3", 1)[indices])
                for joint, angle in robot.spec.fixed_joints:
                    original = Robot(FR3)
                    index = FR3.joints.index(joint)
                    self.assertEqual(angle, (original.q_min[index] + original.q_max[index]) / 2)

    def test_fixed_variants_preserve_bodies_fk_jacobian_mass_and_gravity(self):
        original = Robot(FR3)
        base_xml = ET.parse(Path(MENAGERIE_DIR) / FR3.mjcf)
        rng = np.random.default_rng(527)
        for spec in ROBOTS.values():
            if not spec.fixed_joints:
                continue
            with self.subTest(robot=spec.name):
                reduced = Robot(spec)
                indices = [FR3.joints.index(j) for j in spec.joints]
                xml = ET.parse(Path(MENAGERIE_DIR) / spec.mjcf)
                # Only the nonzero J6 lock changes a local body transformation.
                for body in base_xml.findall(".//body"):
                    other = xml.find(f".//body[@name='{body.attrib['name']}']")
                    self.assertIsNotNone(other)
                    if not (body.get("name") == "fr3_link6" and
                            "fr3_joint6" in dict(spec.fixed_joints)):
                        self.assertEqual(body.attrib, other.attrib)
                    a, b = body.find("inertial"), other.find("inertial")
                    if a is not None:
                        self.assertEqual(a.attrib, b.attrib)
                for _ in range(25):
                    q = rng.uniform(original.q_min, original.q_max)
                    for joint, angle in spec.fixed_joints:
                        q[FR3.joints.index(joint)] = angle
                    p, R = original.fk(q)
                    pr, Rr = reduced.fk(q[indices])
                    np.testing.assert_allclose(pr, p, atol=2e-14, rtol=0)
                    np.testing.assert_allclose(Rr, R, atol=2e-14, rtol=0)
                    np.testing.assert_allclose(reduced.jac(), original.jac()[:, indices], atol=2e-14)
                    np.testing.assert_allclose(
                        gravity_torque(reduced, q[indices]), gravity_torque(original, q)[indices],
                        atol=2e-12, rtol=0,
                    )
                    matrices = []
                    for robot, state in ((original, q), (reduced, q[indices])):
                        d = mujoco.MjData(robot.model)
                        d.qpos[robot.qadr] = state
                        mujoco.mj_forward(robot.model, d)
                        M = np.zeros((robot.n, robot.n))
                        mujoco.mj_fullM(robot.model, d, M)
                        matrices.append(M)
                    np.testing.assert_allclose(matrices[1], matrices[0][np.ix_(indices, indices)],
                                               atol=2e-12, rtol=0)
                # Independently check the reduced position Jacobian with finite differences.
                qr = q[indices].copy()
                reduced.fk(qr)
                Jp = reduced.jac()[:3].copy()
                h = 1e-6
                numeric = np.column_stack([
                    (reduced.fk(qr + h * e)[0] - reduced.fk(qr - h * e)[0]) / (2*h)
                    for e in np.eye(reduced.n)
                ])
                np.testing.assert_allclose(Jp, numeric, atol=2e-10, rtol=0)

    def test_generated_mjcfs_and_isolated_link_density_changes(self):
        for filename, text in generated_models().items():
            self.assertEqual((MODEL_DIR / filename).read_text(), text)
        original = Robot(FR3)
        for name, (filename, link, factor) in MASS_VARIANTS.items():
            with self.subTest(robot=name):
                base = ET.parse(MODEL_DIR / "fr3.xml")
                variant = ET.parse(MODEL_DIR / filename)
                a = base.find(f".//body[@name='{link}']/inertial")
                b = variant.find(f".//body[@name='{link}']/inertial")
                for attr in ("mass", "diaginertia"):
                    np.testing.assert_allclose(np.fromstring(b.get(attr), sep=" "),
                                               factor * np.fromstring(a.get(attr), sep=" "),
                                               atol=0, rtol=3e-16)
                    b.set(attr, a.get(attr))
                variant.getroot().set("model", base.getroot().get("model"))
                # Every other XML attribute, joint, keyframe, geom, COM frame is identical.
                self.assertEqual(ET.tostring(base.getroot()), ET.tostring(variant.getroot()))
                robot = Robot(ROBOTS[name])
                bid = robot.model.body(link).id
                expected = (factor - 1) * original.model.body_mass[bid]
                self.assertAlmostEqual(robot.model.body_mass.sum() - original.model.body_mass.sum(),
                                       expected, places=13)
                np.testing.assert_array_equal(torque_limit(name, 1), torque_limit("FR3", 1))

    def test_mass_gravity_report_and_matched_compensation(self):
        from evaluation.compare_fr3_mass import compare
        cfg = main.load_config("configs/five_obstacle_pic.yaml")
        report = compare(cfg)
        np.testing.assert_array_equal(report["q_rad"], cfg["task"]["q_start_rad"])
        rows = report["variants"]
        for row, added in zip(rows, (0, 1.12245, 2.2449, 0.3135715, 0.627143)):
            self.assertAlmostEqual(row["added_mass_kg"], added, places=13)
            self.assertTrue(row["nominal_matches_plant"])
            self.assertEqual(row["nominal_vs_plant_gravity_max_abs_Nm"], 0)
        for i in (1, 3):
            np.testing.assert_allclose(rows[i+1]["delta_gravity_torque_Nm"],
                                       2*np.array(rows[i]["delta_gravity_torque_Nm"]), atol=3e-14)

    def test_separate_series_keep_benchmark_and_controller_parameters(self):
        for scenario, count in (("five_obstacle", 10), ("worm", 9)):
            reference = main.load_config(f"configs/{scenario}_pic.yaml")
            expected_starts = plan_starts(reference, 100)
            paths = sorted(Path("configs/mass").glob(f"{scenario}_pic_*.yaml"))
            paths += sorted(Path("configs/dof").glob(f"{scenario}_pic_*.yaml"))
            self.assertEqual(len(paths), count)
            for path in paths:
                with self.subTest(config=path):
                    cfg = main.load_config(path)
                    self.assertEqual(plan_starts(cfg, 100), expected_starts)
                    if path.parent.name == "mass":
                        self.assertFalse(ROBOTS[cfg["robot"]].fixed_joints)
                    else:
                        self.assertNotIn(cfg["robot"], MASS_VARIANTS)
                        if scenario != "worm" or cfg["robot"] != "FR3":
                            self.assertIsNone(cfg["task"]["q_start_rad"])
                            cfg["task"]["q_start_rad"] = reference["task"]["q_start_rad"]
                    cfg["robot"] = reference["robot"]
                    self.assertEqual(cfg, reference)

    def test_ur5e_uses_own_effort_limits_and_no_fr3_friction_or_acceleration_verdict(self):
        robot = Robot(ROBOTS["UR5e"])
        report = apply_plant(robot, {})
        np.testing.assert_array_equal(report["applied"]["torque_limit_Nm"], [150]*3 + [28]*3)
        np.testing.assert_array_equal(robot.model.dof_frictionloss, np.zeros(6))
        self.assertIsNone(robot.spec.acceleration_reference_rad_s2)
        with self.assertRaisesRegex(ValueError, "FR3 synthetic friction is not defined"):
            apply_plant(Robot(ROBOTS["UR5e"]), {"synthetic_friction": "relative"})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "manifest.json").write_text(json.dumps({"status": "complete", "robot": "UR5e"}))
            np.savez(path / "traj.npz", t=[0, 0.001], qacc=[[100]*6, [0]*6])
            _, rows, _ = aggregate(path)
            self.assertEqual(rows[0]["peak_abs_qacc_rad_s2"], 100)
            self.assertIn("| FR3 acceleration reference exceeded in simulation (>10 rad/s²) | NA | 0 | 1 | NA |",
                          (path / "summary.md").read_text())

    def test_tcp_plan_matches_existing_batch_for_every_robot_and_controller(self):
        cfg = main.load_config("configs/five_obstacle_pic.yaml")
        expected = plan_starts(cfg, 100)
        saved = Path("results/five_obstacle_pic_100/starts.csv")
        if saved.exists():
            with saved.open() as handle:
                rows = list(csv.DictReader(handle))
            np.testing.assert_array_equal([r["tcp_start_m"] for r in expected],
                                          [json.loads(r["tcp_start_m"]) for r in rows])
        for robot in ROBOTS:
            for controller in ("pic", "servo"):
                cfg["robot"], cfg["controller"] = robot, controller
                self.assertEqual(plan_starts(cfg, 100), expected)

    def test_public_single_and_batch_paths_all_robots_both_controllers(self):
        with tempfile.TemporaryDirectory() as folder:
            for robot, spec in ROBOTS.items():
                for controller in ("pic", "servo"):
                    with self.subTest(robot=robot, controller=controller):
                        args = ["configs/five_obstacle_pic.yaml", "--set", f"robot={robot}",
                                "--set", f"controller={controller}", "--set", "task.q_start_rad=null",
                                "--set", "sim.T_max=0.004"]
                        single = Path(folder) / f"{robot}_{controller}_single"
                        self.assertEqual(main.main([*args, "--output", str(single)]), 0)
                        batch = Path(folder) / f"{robot}_{controller}_batch"
                        self.assertEqual(main.main([*args, "--runs", "2", "--output", str(batch)]), 0)
                        with (batch / "runs.csv").open() as handle:
                            rows = list(csv.DictReader(handle))
                        self.assertEqual([r["status"] for r in rows], ["complete", "complete"])
                        for directory in (single, batch / "start_000", batch / "start_001"):
                            manifest = json.loads((directory / "manifest.json").read_text())
                            self.assertEqual(manifest["robot_model"]["joint_names"], list(spec.joints))
                            self.assertEqual(manifest["robot_model"]["fixed_joints_rad"], dict(spec.fixed_joints))
                            self.assertEqual(len(manifest["q_start_rad"]), len(spec.joints))
                            tr = np.load(directory / "traj.npz")
                            self.assertEqual(tr["Q"].shape, (5, len(spec.joints)))
                            self.assertTrue(np.isfinite(tr["tau_applied"]).all())
                            self.assertLess(np.linalg.norm(tr["P"][0] - tr["P_ideal"][0]), 1e-6)
                        # The paired controllers must select exactly the same IK branch per model.
                        if controller == "servo":
                            for start in ("start_000", "start_001"):
                                pic = Path(folder) / f"{robot}_pic_batch" / start / "manifest.json"
                                servo = batch / start / "manifest.json"
                                self.assertEqual(json.loads(pic.read_text())["q_start_rad"],
                                                 json.loads(servo.read_text())["q_start_rad"])
                        if robot in MASS_VARIANTS:
                            pairs = [(single, Path(folder) / f"FR3_{controller}_single")]
                            pairs += [(batch / start, Path(folder) / f"FR3_{controller}_batch" / start)
                                      for start in ("start_000", "start_001")]
                            for changed, reference in pairs:
                                a = json.loads((changed / "manifest.json").read_text())
                                b = json.loads((reference / "manifest.json").read_text())
                                self.assertEqual(a["q_start_rad"], b["q_start_rad"])


if __name__ == "__main__":
    unittest.main()
