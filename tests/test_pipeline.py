"""Check interval timing, model selection, metrics, and failure accounting."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

import main
from evaluation.metrics import dtw_mean, tracking_metrics
from sim.controllers import pic, servo
from sim.ds import ROOT, SpatialEmbedding
from sim.robot import multistart_ik, position_servo_map
from sim.sim import build_case


def short_case(controller="pic"):
    cfg = main.load_config(ROOT / f"configs/worm_{controller}.yaml")
    cfg["sim"]["T_max"] = 0.004
    return cfg, build_case(cfg)


class TestIntervals(unittest.TestCase):
    def test_pic_interval_torque_and_cleanup(self):
        _, case = short_case()
        flags = int(case.robot.model.opt.disableflags)
        before, after = [], []
        original_step = mujoco.mj_step

        def step(model, data):
            self.assertTrue(model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION))
            before.append(data.qfrc_applied[case.robot.dadr].copy())
            original_step(model, data)
            after.append(data.qacc[case.robot.dadr].copy())

        with patch.object(mujoco, "mj_step", side_effect=step):
            result = pic.run(case)
        tr = result["traj"]
        self.assertEqual(len(before), 4)
        self.assertEqual(len(tr["t"]), 5)
        np.testing.assert_array_equal(before, tr["tau_applied"][:-1])
        np.testing.assert_array_equal(after, tr["qacc"][:-1])
        self.assertEqual(int(case.robot.model.opt.disableflags), flags)
        with patch.object(mujoco, "mj_step", side_effect=RuntimeError("injected step error")):
            with self.assertRaisesRegex(RuntimeError, "injected step error"):
                pic.run(case)
        self.assertEqual(int(case.robot.model.opt.disableflags), flags)

    def test_servo_applies_new_target_and_keeps_old_target_at_state(self):
        _, case = short_case("servo")
        mapping = position_servo_map(case.robot)
        before, after = [], []
        original_step = mujoco.mj_step

        def step(model, data):
            before.append(
                (
                    data.qpos[case.robot.qadr].copy(),
                    data.qvel[case.robot.dadr].copy(),
                    data.ctrl[mapping.act].copy(),
                )
            )
            original_step(model, data)
            after.append(data.actuator_force[mapping.act].copy())

        with patch.object(mujoco, "mj_step", side_effect=step):
            tr = servo.run(case)["traj"]
        self.assertEqual(len(before), 4)
        for k, (q, qd, target) in enumerate(before):
            np.testing.assert_array_equal(q, tr["Q"][k])
            np.testing.assert_array_equal(target, tr["Q_des"][k + 1])
            np.testing.assert_array_equal(
                tr["tau_cmd"][k], mapping.kp * (target - q) - mapping.kv * qd
            )
        np.testing.assert_array_equal(after, tr["tau_applied"])
        self.assertTrue(np.any(tr["Q_des"][0] != tr["Q_des"][1]))


class TestInputs(unittest.TestCase):
    def test_unsupported_benchmark_is_not_silently_aliased(self):
        cfg = main.load_config(ROOT / "configs/worm_pic.yaml")
        cfg["model"] = "LASA/Worm_benchmark"
        with self.assertRaisesRegex(ValueError, "no demonstration data mapping"):
            build_case(cfg)

    def test_unknown_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown configuration key"):
            main.load_config(ROOT / "configs/worm_pic.yaml", ["sim.timstep=0.001"])

    def test_invalid_timestep_and_boolean_are_rejected(self):
        for setting in ("sim.dt=0", "sim.T_max=-1", "pic.use_f=1"):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                main.load_config(ROOT / "configs/worm_pic.yaml", [setting])

    def test_spatial_embedding_roundtrip(self):
        emb = SpatialEmbedding([0.5, 0, 0.5], np.eye(3), 0.1, [0, 0, 0])
        x = np.array([0.2, -0.3, 0.4])
        recovered, offset = emb.to_plane(emb.to_world(x))
        np.testing.assert_allclose(recovered, x, atol=1e-15)
        self.assertEqual(offset, 0.0)

    def test_multistart_ik_is_reproducible(self):
        _, case = short_case()
        a = multistart_ik(
            case.robot, case.p0_world, case.tgt, case.robot.q_home, n_starts=2, seed=73
        )
        b = multistart_ik(
            case.robot, case.p0_world, case.tgt, case.robot.q_home, n_starts=2, seed=73
        )
        self.assertTrue(a.converged)
        np.testing.assert_array_equal(a.q, b.q)
        self.assertLess(a.pos_err, 1e-6)


class TestEvaluationAndSaving(unittest.TestCase):
    def test_mse_and_dtw_constant_offset(self):
        reference = np.zeros((3, 3))
        measured = np.tile([3.0, 4.0, 0.0], (3, 1))
        metrics, _, _ = tracking_metrics(measured, reference, [0, 0, 0], 0.1, 0.001, 0.01)
        self.assertEqual(metrics["mse_sync_m2"], 25.0)
        self.assertEqual(dtw_mean(measured, reference), 5.0)

    def test_failure_has_manifest_and_existing_output_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            with patch.object(pic, "run", side_effect=RuntimeError("injected run error")):
                status = main.main([str(ROOT / "configs/worm_pic.yaml"), "--output", str(output)])
            self.assertEqual(status, 1)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "error")
            self.assertIn("injected run error", (output / "error.txt").read_text())
            self.assertFalse((output / "traj.npz").exists())
            self.assertEqual(
                main.main([str(ROOT / "configs/worm_pic.yaml"), "--output", str(output)]), 2
            )


if __name__ == "__main__":
    unittest.main()
