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
from sim.robot import (
    FR3, MENAGERIE_DIR, ROBOTS, Robot, apply_plant, gravity_torque,
    position_servo_map, torque_limit,
)


class RobotModelTests(unittest.TestCase):
    def test_compilation_joint_names_limits_and_keyframes(self):
        counts = {"FR3": 7, "FR3_fixed_J5": 6, "FR3_fixed_J3": 6,
                  "FR3_fixed_J3_J5": 5, "UR5e": 6}
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
                    self.assertEqual(robot.spec.joints[-1], "fr3_joint7")
                    indices = [FR3.joints.index(j) for j in robot.spec.joints]
                    np.testing.assert_array_equal(torque_limit(name, 1), torque_limit("FR3", 1)[indices])
                for joint, angle in robot.spec.fixed_joints:
                    original = Robot(FR3)
                    index = FR3.joints.index(joint)
                    self.assertEqual(angle, (original.q_min[index] + original.q_max[index]) / 2)
                    self.assertEqual(angle, 0.0)

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
                # Link transforms and inertial elements are unchanged at the zero locks.
                for body in base_xml.findall(".//body"):
                    other = xml.find(f".//body[@name='{body.attrib['name']}']")
                    self.assertIsNotNone(other)
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


if __name__ == "__main__":
    unittest.main()
