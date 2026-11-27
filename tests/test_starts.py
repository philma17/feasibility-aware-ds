"""Reproducibility and task geometry at the same interfaces used by the CLI."""
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from experiment import load_config
from main import main
from sim.batch import start_config
from sim.controllers import servo
from sim.pairs import comparison_configs, plan_pair
from sim.sim import build_case, prepare_case
from sim.robot import IKParams, ROBOTS, Robot, UR5E, scaled_sigma_min, solve_qdot, task_rows
from sim.starts import NO_IK, NO_PAIR, StartUnavailable


class StartTests(unittest.TestCase):
    def test_locking_the_base_preserves_home_self_contact_filter(self):
        for name in ("FR3", "FR3_fixed_J1", "FR3_fixed_J1_J3", "FR3_fixed_J1_J5"):
            with self.subTest(name=name):
                robot = Robot(ROBOTS[name])
                robot.data.qpos[robot.qadr] = robot.reference_posture()
                mujoco.mj_forward(robot.model, robot.data)
                self.assertEqual(robot.data.ncon, 0)

    def test_real_self_contact_survives_a_wrist_lock(self):
        q = np.array([2.702039, .171347, -1.193862, -1.277630, 1.003839, .440714, 2.898543])
        for name, angles in (("FR3", None), ("FR3_fixed_J7", {"fr3_joint7": q[-1]})):
            with self.subTest(name=name):
                robot = Robot(ROBOTS[name], fixed_joints_rad=angles)
                robot.data.qpos[robot.qadr] = q[:robot.n]
                mujoco.mj_forward(robot.model, robot.data)
                pairs = {tuple(sorted(robot.model.geom(g).name for g in contact.geom))
                         for contact in robot.data.contact}
                self.assertIn(("fr3_link5_collision", "fr3_link7_collision"), pairs)

    def test_servo_centers_free_roll_in_rank_deficient_axis5(self):
        cfg = load_config("configs/worm_axis5_pic.yaml", ["robot=FR3_fixed_J1_J3"])
        case = prepare_case(cfg)
        robot = case.robot
        q = robot.q_home
        _, R = robot.fk(q)
        rows = task_rows(robot.jac(), R, case.tgt, np.zeros(3), 1.0)[0]
        span = robot.q_max - robot.q_min
        mid = robot.q_min + span / 2
        qd, _ = solve_qdot(rows, np.zeros(5), np.ones(robot.n, bool), robot.qd_max,
                           IKParams(k_null=1.0), q, mid, span)
        spin = robot.spec.joints.index("fr3_joint7")
        self.assertAlmostEqual(qd[spin], (mid[spin] - q[spin]) / (span[spin] / 2))
        np.testing.assert_allclose(rows @ qd, 0, atol=1e-12)

    def test_singular_angular_inertia_at_a_locked_start(self):
        for name in ("FR3_fixed_J5", "FR3_fixed_J3_J5_J7"):
            cfg = load_config("configs/worm_axis5_pic.yaml", [f"robot={name}", "sim.n_starts=1"])
            robot = Robot(ROBOTS[name])
            q = robot.reference_posture()
            p, R = robot.fk(q)
            cfg["task"].update(orientation="hold", center=p.tolist(), start={"x0": [0, 0]},
                               R_des=R.tolist())
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "rotational inertia"):
                build_case(cfg)
            cfg["controller"] = "servo"
            with self.subTest(name=name, controller="servo"):
                np.testing.assert_array_equal(build_case(cfg).q_start, q)

    def test_pairing_is_selected_by_yaml(self):
        args = ["configs/worm_axis5_pic.yaml", "--set", "sim.n_starts=1", "--set", "sim.T_max=0.002",
                "--set", "pairing={mode: lock, fixed_joints_rad: {fr3_joint5: 0.2}}"]
        with TemporaryDirectory() as tmp:
            batch = Path(tmp) / "batch"
            self.assertEqual(main([*args, "--output", str(batch)]), 2)
            self.assertFalse(batch.exists())
            self.assertEqual(main([*args, "--runs", "1", "--output", str(batch)]), 0)
            rows = json.loads((batch / "starts.json").read_text())
            free, locked = rows
            self.assertEqual((free["member"], locked["member"]), ("free", "locked"))
            self.assertEqual(free["q_start_full_rad"], locked["q_start_full_rad"])
            self.assertEqual(free["q_start_full_rad"][4], 0.2)

    def test_batch_and_single_cli_replay_identical_trajectory(self):
        args = ["configs/worm_axis5_pic.yaml", "--set", "sim.n_starts=1", "--set", "sim.T_max=0.002"]
        with TemporaryDirectory() as tmp:
            batch, replay = Path(tmp) / "batch", Path(tmp) / "replay"
            self.assertEqual(main([*args, "--runs", "1", "--output", str(batch)]), 0)
            with patch("sim.starts.multistart_ik", side_effect=AssertionError("replay ran IK")):
                self.assertEqual(main([*args, "--start-list", str(batch / "starts.json"),
                                       "--start-id", "start_000", "--output", str(replay)]), 0)
            with np.load(batch / "start_000/traj.npz") as a, np.load(replay / "traj.npz") as b:
                for field in ("Q", "QD", "P", "sigma_min_task"):
                    np.testing.assert_array_equal(a[field], b[field])

    def test_axis5_tilted_plane(self):
        cfg = load_config("configs/worm_axis5_pic.yaml", ["sim.n_starts=1"])
        cfg["task"]["plane"]["e2"] = [0, 1, 0.3]
        case = build_case(cfg)
        target = case.tgt
        pointing = -np.array([0, -0.3, 1]) / np.sqrt(1.09)
        p, R = case.robot.fk(case.q_start)
        np.testing.assert_allclose(p, case.p0_world, atol=1e-6)
        np.testing.assert_allclose(R @ target.axis_local, pointing, atol=1e-6)
        J = np.eye(6)
        for angle in (0, 0.3, np.pi):
            tilt = Rotation.from_rotvec(angle * np.array([1, 0, 0])).as_matrix()
            rows, _, error = task_rows(J, tilt @ R, target, np.zeros(3), 1)
            self.assertAlmostEqual(error, angle, places=6)
            np.testing.assert_allclose(rows[3:, 3:] @ (tilt @ R @ target.axis_local), 0, atol=1e-12)
        spin = Rotation.from_rotvec(1.2 * target.axis_local).as_matrix()
        self.assertLess(task_rows(J, R @ spin, target, np.zeros(3), 1)[2], 1e-6)
        # Use an exactly aligned/opposed frame to exercise the antipodal branch.
        aligned = Rotation.align_vectors([pointing], [target.axis_local])[0].as_matrix()
        for angle in (0, 0.3, np.pi):
            frame = Rotation.from_rotvec([angle, 0, 0]).as_matrix() @ aligned
            e, basis = target.tilt(frame)
            moment = case.orient.torque(frame @ target.axis_local, e, basis)
            np.testing.assert_allclose(moment, case.orient.K_r * e, atol=1e-12)
            self.assertAlmostEqual(np.linalg.norm(e), angle)
            self.assertAlmostEqual(moment @ (frame @ target.axis_local), 0)
            if angle > 0:
                corrected = Rotation.from_rotvec(e).apply(frame @ target.axis_local)
                np.testing.assert_allclose(corrected, pointing, atol=1e-12)

    def test_replay_validates_saved_state_without_ik(self):
        cfg = start_config(load_config("configs/worm_axis5_pic.yaml", ["sim.n_starts=1"]), 2, 1)
        case = build_case(cfg)
        row = case.starts.record(case.q_start)
        self.assertEqual(row["start_id"], "start_001")
        with patch("sim.starts.multistart_ik", side_effect=AssertionError("replay ran IK")):
            servo = build_case({**cfg, "controller": "servo"}, start_row=row)
            np.testing.assert_array_equal(servo.q_start, case.q_start)
            for field, value in (("q_start_rad", [0]*7), ("qd_start_rad_s", [1]*7),
                                 ("q_start_rad", case.starts.q_ref), ("status", "complete")):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    build_case(cfg, start_row={**row, field: value})
            moved = deepcopy(cfg)
            moved["task"]["center"][0] += 0.01
            with self.assertRaisesRegex(ValueError, "incompatible"):
                build_case(moved, start_row=row)

    def test_no_replacement_and_missing_pair(self):
        cfg = load_config("configs/worm_axis5_pic.yaml", ["sim.n_starts=1"])
        case = build_case(cfg, start_id="s")
        self.assertEqual(case.starts.lower_sigma(case.ik, 0.01)["status"], NO_PAIR)
        cfg["task"]["center"][0] = 10
        with self.assertRaises(StartUnavailable) as failed:
            build_case(cfg, start_id="s")
        self.assertEqual((failed.exception.row["start_id"], failed.exception.row["status"]), ("s", NO_IK))

    def test_prescribed_lock_and_task_rank(self):
        cfg = load_config("configs/worm_axis5_pic.yaml", ["sim.n_starts=1"])
        with self.assertRaisesRegex(ValueError, "lock angle"):
            comparison_configs(cfg, "lock", ("fr3_joint5", 9.0))
        pair = plan_pair(comparison_configs(cfg, "lock", ("fr3_joint5", 0.2)), "s", "lock", None)
        np.testing.assert_array_equal(pair["free"]["q_start_full_rad"], pair["locked"]["q_start_full_rad"])
        self.assertEqual(pair["free"]["q_start_rad"][4], 0.2)
        self.assertAlmostEqual(pair["locked"]["effective_fixed_joints_rad"]["fr3_joint5"], 0.2)
        self.assertEqual(scaled_sigma_min(np.eye(5, 4), 0.1), 0.0)
        self.assertAlmostEqual(scaled_sigma_min(np.eye(5), 0.1), 0.1)

    def test_original_max_sigma_margin_and_bounded_angles(self):
        cfg = start_config(load_config("configs/worm_pic.yaml", ["sim.n_starts=6"]), 2, 0)
        # Recorded before the changes on c9c1af0, MuJoCo 3.16.0.
        expected = [-0.005666887692003158, -0.24926547791684775, -0.005870037878680272,
                    -2.086161664817612, -0.0013288164914594816, 1.2186386361548023, -0.7853]
        np.testing.assert_allclose(build_case(cfg).q_start, expected, atol=1e-12, rtol=0)
        ur = Robot(UR5E)
        a, b = ur.reference_posture(), ur.reference_posture()
        a[0], b[0] = -3.1, 3.1
        self.assertAlmostEqual(ur.distance_squared(a, b), 6.2**2)

    def test_locked_servo_logs_joint_torque_after_native_clipping(self):
        cfg = load_config("configs/worm_servo.yaml", ["robot=FR3_fixed_J1_J5", "sim.dt=0.1",
                          "sim.T_max=0.1", "plant.servo_enforce_torque_limit=false"])
        case = prepare_case(cfg)
        case.q_start = case.robot.reference_posture()

        def static_command(model, data, n, *, observe, command, after_step, **kwargs):
            command(0, observe(0)).apply(data)
            mujoco.mj_forward(model, data)
            after_step(0)
            observe(1)

        with patch.object(servo, "execute_intervals", static_command), patch.object(
            servo, "ideal_rollout", return_value=np.tile(case.x0_n, (2, 1))
        ):
            result = servo.run(case)
        limit = np.array([87, 87, 87, 12, 12])  # J2, J3, J4, J6, J7.
        command = result["traj"]["tau_cmd"][0]
        saturated = np.abs(command) > limit
        self.assertTrue(saturated.any())
        np.testing.assert_allclose(result["traj"]["tau_applied"][0], np.clip(command, -limit, limit))
        np.testing.assert_array_equal(result["force_sat_steps_per_joint"], saturated)
