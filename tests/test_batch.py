"""Check paired starts, single-run reuse, and complete accounting of failed batches."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import experiment
import main
from sim.batch import plan_starts, run_batch, start_config
from sim.ds import load_scenario, sample_starts


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "batch"
        self.cfg = main.load_config("configs/worm_pic.yaml")

    def test_source_sampling_prefix_and_controller_pair_inputs(self):
        for model in ("worm", "five_obstacle"):
            with self.subTest(model=model):
                pic = main.load_config(f"configs/{model}_pic.yaml")
                servo = main.load_config(f"configs/{model}_servo.yaml")
                a, b = plan_starts(pic, 3), plan_starts(servo, 3)
                self.assertEqual(a, b)
                self.assertEqual(pic["sim"]["seed"], servo["sim"]["seed"])
                self.assertEqual(pic["sim"]["n_starts"], servo["sim"]["n_starts"])
                one = start_config(pic, 3, 1)
                self.assertIsNone(one["task"]["q_start_rad"])
                scn = load_scenario(one)
                legacy = np.random.default_rng(pic["multistart"]["seed"]).uniform(
                    scn["init_box"][:, 0], scn["init_box"][:, 1], size=(3, 2)
                )
                np.testing.assert_array_equal(scn["x0"], legacy[1])
                np.testing.assert_array_equal([r["x0_n"] for r in a], legacy)
                np.testing.assert_array_equal(sample_starts(scn["init_box"], 0, 100)[:3], legacy)

    def fake_single(self, cfg, directory, *, hashes, command_line, start_id):
        self.assertIsNone(cfg["task"]["q_start_rad"])
        directory.mkdir()
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "model": cfg["model"],
                    "robot": cfg["robot"],
                    "controller": cfg["controller"],
                    "start_id": start_id,
                    "q_start_rad": [0] * 7,
                }
            )
        )
        (directory / "metrics.json").write_text(json.dumps({"success": True, "finite": True}))
        return 0

    def read_rows(self, name="runs.csv"):
        with (self.output / name).open() as handle:
            return list(csv.DictReader(handle))

    def test_each_start_calls_single_run_and_automatic_aggregation(self):
        calls = []

        def single(cfg, directory, **kwargs):
            calls.append(cfg["task"]["start"]["index"])
            return self.fake_single(cfg, directory, **kwargs)

        code = run_batch(self.cfg, self.output, 3, single, hashes={}, command_line=["test"])
        self.assertEqual(code, 0)
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(len(self.read_rows()), 3)
        starts = self.read_rows("starts.csv")
        self.assertEqual([r["run_dir"] for r in starts], [f"start_{i:03d}" for i in range(3)])
        self.assertEqual(starts[0]["q_start_rad"], "[0, 0, 0, 0, 0, 0, 0]")
        self.assertFalse((self.output / "manifest.json").exists())
        self.assertEqual(self.read_rows()[0]["barrier_exit"], "NA")
        with self.assertRaisesRegex(ValueError, "already exists"):
            run_batch(self.cfg, self.output, 3, single, hashes={}, command_line=["test"])

    def test_ik_failure_keeps_row_and_later_start_runs(self):
        original = experiment.build_case

        def build(cfg):
            if cfg["task"]["start"]["index"] == 0:
                raise RuntimeError("start IK did not converge (test injection)")
            return original(cfg)

        cfg = main.load_config("configs/worm_pic.yaml", ["sim.T_max=0.002"])
        with patch.object(experiment, "build_case", side_effect=build):
            code = run_batch(
                cfg,
                self.output,
                2,
                experiment.run_single,
                hashes=main.input_hashes(cfg),
                command_line=["test"],
            )
        self.assertEqual(code, 1)
        rows = self.read_rows()
        self.assertEqual([r["status"] for r in rows], ["error", "complete"])
        self.assertEqual(rows[0]["goal_failure"], "NA")
        self.assertIn("start_ik", rows[0]["notes"])
        self.assertIn("did not converge", rows[0]["notes"])

    def test_interrupt_preserves_all_planned_starts(self):
        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            run_batch(self.cfg, self.output, 2, interrupted, hashes={}, command_line=["test"])
        self.assertEqual([r["status"] for r in self.read_rows()], ["missing", "missing"])
        self.assertEqual(
            json.loads((self.output / "batch.json").read_text())["status"], "incomplete"
        )

    def test_setup_exception_is_recorded_and_continues(self):
        def broken(*args, **kwargs):
            raise ValueError("test setup failure")

        code = run_batch(self.cfg, self.output, 2, broken, hashes={}, command_line=["test"])
        self.assertEqual(code, 1)
        self.assertEqual([r["status"] for r in self.read_rows()], ["error", "error"])

    def test_cli_rejects_invalid_count_before_creating_output(self):
        for args in (("--runs", "0"), ("--runs", "2", "--check")):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                main.main(["configs/worm_pic.yaml", "--output", str(self.output), *args])
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
