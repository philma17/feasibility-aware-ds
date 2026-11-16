"""Check planned-start accounting and evaluation decisions using small saved artifacts."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from evaluation.aggregate import FIELDS, aggregate, load_evaluation, metric_statistics


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_dir(self, name="run", status="complete", **updates):
        directory = self.root / name
        directory.mkdir()
        manifest = {
            "status": status,
            "model": "test/model",
            "robot": "FR3",
            "controller": "pic",
            "embedding": {
                "origin": [0, 0, 0],
                "axes": [[1, 0], [0, 1], [0, 0]],
                "anchor_n": [0, 0],
                "L_m": 0.1,
            },
        }
        metrics = {
            "final_distance_m": 0.0001,
            "t_reach_s": 0.1,
            "success": True,
            "mse_sync_m2": 0.0003,
            "dtw_mean_m": 0.004,
            "max_err_sync_m": 0.02,
            "geom_dev_max_m": 0.003,
            "min_clearance_m": 0.01,
            "domain_exit": False,
            "peak_applied_torque_ratio": 0.8,
            "torque_saturation_steps": 2,
            "finite": True,
        }
        metrics.update(updates)
        (directory / "manifest.json").write_text(json.dumps(manifest))
        (directory / "metrics.json").write_text(json.dumps(metrics))
        (directory / "config_resolved.yaml").write_text(
            yaml.safe_dump({"sim": {"success_frac": 0.01}})
        )
        return directory

    def evaluation(self, **cfg):
        path = self.root / "evaluation.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path

    def test_saved_metrics_and_acceleration_exclude_terminal_padding(self):
        directory = self.run_dir()
        np.savez(directory / "traj.npz", t=[0, 0.1, 0.2], qacc=[[1, -6], [3, 4], [9999, 9999]])
        path, rows, planned = aggregate(self.root)
        self.assertFalse(planned)
        row = rows[0]
        self.assertEqual(row["mse_sync_m2"], 0.0003)
        self.assertEqual(row["goal_threshold_m"], 0.001)
        self.assertFalse(row["goal_failure"])
        self.assertFalse(row["obstacle_entry"])
        self.assertIsNone(row["tracking_failure"])
        self.assertIsNone(row["barrier_exit"])
        self.assertEqual(row["peak_abs_qacc_rad_s2"], 6)
        self.assertEqual(row["qacc_intervals"], 2)
        with path.open() as handle:
            saved = list(csv.DictReader(handle))[0]
        self.assertEqual(saved["tracking_failure"], "NA")
        self.assertEqual(saved["goal_failure"], "false")

    def test_single_run_takes_precedence_and_preserves_batch_csv(self):
        directory = self.run_dir()
        self.run_dir("run/nested")
        (directory / "starts.csv").write_text("start_id,run_dir\nother,missing\n")
        existing_csv = self.root / "runs.csv"
        existing_csv.write_text("existing batch table\n")
        path, rows, planned = aggregate(directory)
        self.assertEqual(path, directory / "runs.csv")
        self.assertEqual(len(rows), 1)
        self.assertFalse(planned)
        self.assertEqual(rows[0]["start_id"], "run")
        self.assertEqual(rows[0]["run_dir"], str(directory))
        self.assertEqual(rows[0]["mse_sync_m2"], 0.0003)
        self.assertEqual(existing_csv.read_text(), "existing batch table\n")

    def test_batch_writes_one_shared_csv(self):
        self.run_dir("one")
        self.run_dir("two")
        path, rows, _ = aggregate(self.root)
        self.assertEqual(path, self.root / "runs.csv")
        self.assertEqual([row["start_id"] for row in rows], ["one", "two"])
        self.assertFalse((self.root / "one/runs.csv").exists())
        self.assertFalse((self.root / "two/runs.csv").exists())

    def test_plan_keeps_error_incomplete_and_missing_cases(self):
        for name, status in (("a", "complete"), ("b", "error"), ("c", "incomplete")):
            self.run_dir(name, status)
        self.run_dir("unplanned")
        (self.root / "starts.csv").write_text(
            "start_id,run_dir\n001,a\n002,b\n003,c\n004,missing\n"
        )
        _, rows, planned = aggregate(self.root)
        self.assertTrue(planned)
        self.assertEqual([r["start_id"] for r in rows], ["001", "002", "003", "004"])
        self.assertEqual(
            [r["status"] for r in rows], ["complete", "error", "incomplete", "missing"]
        )
        self.assertFalse(rows[0]["goal_failure"])
        for row in rows[1:]:
            self.assertIsNone(row["goal_failure"])
            self.assertIsNone(row["tracking_failure"])
            self.assertIsNone(row["obstacle_entry"])

    def test_discovery_keeps_invalid_or_missing_manifest(self):
        bad = self.run_dir("bad")
        (bad / "manifest.json").write_text("{")
        orphan = self.run_dir("orphan")
        (orphan / "manifest.json").unlink()
        _, rows, _ = aggregate(self.root)
        self.assertEqual([r["status"] for r in rows], ["invalid_manifest", "missing_manifest"])
        self.assertIsNone(rows[0]["goal_failure"])

    def test_goal_failure_uses_saved_success_not_tracking(self):
        self.run_dir(success=False, final_distance_m=0.01, max_err_sync_m=0.001, min_clearance_m=0)
        _, rows, _ = aggregate(
            self.root, evaluation=self.evaluation(tracking_thresholds={"max_err_sync_m": 0.001})
        )
        self.assertTrue(rows[0]["goal_failure"])
        self.assertFalse(rows[0]["tracking_failure"])
        self.assertTrue(rows[0]["obstacle_entry"])

    def test_tracking_exceedance_and_missing_threshold_metric(self):
        self.run_dir(dtw_mean_m=None)
        _, rows, _ = aggregate(
            self.root,
            evaluation=self.evaluation(
                tracking_thresholds={"max_err_sync_m": 0.01, "dtw_mean_m": 0.001}
            ),
        )
        self.assertTrue(rows[0]["tracking_failure"])
        _, rows, _ = aggregate(
            self.root,
            evaluation=self.evaluation(
                tracking_thresholds={"max_err_sync_m": 0.03, "dtw_mean_m": 0.001}
            ),
        )
        self.assertIsNone(rows[0]["tracking_failure"])

    def test_goal_override_updates_entry_time_from_world_positions(self):
        directory = self.run_dir(success=False, final_distance_m=0.1, t_reach_s=None)
        np.savez(
            directory / "traj.npz", t=[0, 0.1, 0.2], P=[[0.3, 0, 0], [0.15, 0, 0], [0.1, 0, 0]]
        )
        _, rows, _ = aggregate(self.root, evaluation=self.evaluation(goal_threshold_m=0.2))
        self.assertFalse(rows[0]["goal_failure"])
        self.assertEqual(rows[0]["t_reach_s"], 0.1)
        self.assertEqual(rows[0]["goal_entry_status"], "entered")

    def test_goal_override_without_raw_samples_does_not_reuse_saved_entry(self):
        self.run_dir(t_reach_s=0.1)
        _, rows, _ = aggregate(self.root, evaluation=self.evaluation(goal_threshold_m=0.01))
        self.assertIsNone(rows[0]["t_reach_s"])
        self.assertEqual(rows[0]["goal_entry_status"], "unavailable")

    def test_unavailable_and_corrupt_metrics_do_not_become_false(self):
        directory = self.run_dir(min_clearance_m=None, success=None)
        _, rows, _ = aggregate(self.root)
        self.assertIsNone(rows[0]["obstacle_entry"])
        self.assertIsNone(rows[0]["goal_failure"])
        (directory / "metrics.json").write_text("[]")
        _, rows, _ = aggregate(self.root)
        self.assertEqual(rows[0]["status"], "complete")
        self.assertEqual(rows[0]["metrics_status"], "invalid")

    def test_corrupt_trajectory_is_reported_without_dropping_run(self):
        directory = self.run_dir()
        (directory / "traj.npz").write_bytes(b"invalid npz")
        _, rows, _ = aggregate(self.root)
        self.assertEqual(rows[0]["qacc_status"], "invalid")
        self.assertIn("Invalid trajectory", rows[0]["notes"])
        self.assertEqual(rows[0]["mse_sync_m2"], 0.0003)

    def test_nonfinite_complete_run_has_no_task_verdict(self):
        self.run_dir(finite=False)
        _, rows, _ = aggregate(self.root)
        self.assertEqual(rows[0]["status"], "complete")
        self.assertIsNone(rows[0]["goal_failure"])

    def test_malformed_saved_yaml_does_not_discard_saved_success(self):
        directory = self.run_dir()
        (directory / "config_resolved.yaml").write_text("sim: [")
        _, rows, _ = aggregate(self.root)
        self.assertFalse(rows[0]["goal_failure"])
        self.assertIsNone(rows[0]["goal_threshold_m"])
        self.assertIn("Cannot read saved goal radius", rows[0]["notes"])

    def test_invalid_evaluation_and_duplicate_plan_are_rejected(self):
        for cfg in (
            {"goal_threshold_m": 0},
            {"tracking_thresholds": {"unknown": 0.1}},
            {"tracking_thresholds": {"mse_sync_m2": float("nan")}},
        ):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                load_evaluation(self.evaluation(**cfg))
        (self.root / "starts.csv").write_text("start_id,run_dir\na,one\na,two\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate(self.root)

    def test_statistics_use_sample_sd_linear_quartiles_and_all_tied_max_ids(self):
        rows = [{"start_id": str(i), "value": value} for i, value in enumerate(
            (1.0, 2.0, 3.0, 4.0, None, float("nan"), True)
        )]
        stats = metric_statistics(rows, "value")
        self.assertEqual(stats["n"], 4)
        self.assertEqual(stats["mean"], 2.5)
        self.assertAlmostEqual(stats["sd"], (5 / 3) ** 0.5)
        self.assertEqual((stats["q1"], stats["median"], stats["q3"]), (1.75, 2.5, 3.25))
        self.assertEqual(stats["max_ids"], ["3"])
        rows.append({"start_id": "tie", "value": 4.0})
        self.assertEqual(metric_statistics(rows, "value")["max_ids"], ["3", "tie"])
        self.assertIsNone(metric_statistics(rows[:1], "value")["sd"])
        self.assertEqual(metric_statistics([], "value")["n"], 0)
        self.assertIsNone(metric_statistics([], "value")["maximum"])

    def test_summary_keeps_unknown_events_and_technical_status_separate(self):
        self.run_dir("a", domain_exit=True, min_clearance_m=-0.001, success=False)
        self.run_dir("b", "incomplete", domain_exit=False, torque_saturation_steps=None)
        (self.root / "starts.csv").write_text("start_id,run_dir\na,a\nb,b\nc,missing\n")
        path, _, _ = aggregate(self.root)
        summary = path.with_name("summary.md").read_text()
        self.assertIn("| `domain_exit` | 1 | 2 | 1 | `a` |", summary)
        self.assertIn("| `goal_failure` | 1 | 1 | 2 | `a` |", summary)
        self.assertIn("| `obstacle_entry` | 1 | 1 | 2 | `a` |", summary)
        self.assertIn("| TCP | B>0 inside X | NA | 0 | 3 | NA | NA | NA |", summary)
        self.assertIn("| `tracking_failure` | NA | 0 | 3 | NA |", summary)
        self.assertIn("status other than complete: `b`, `c`", summary)
        self.assertFalse(list(self.root.rglob("*.png")))
        with path.open() as handle:
            self.assertEqual(csv.DictReader(handle).fieldnames, list(FIELDS))

    def test_summary_reports_raw_peaks_excluding_padding_and_explicit_thresholds(self):
        directory = self.run_dir()
        np.savez(directory / "traj.npz", t=[0, 0.1, 0.2],
                 tau_applied=[[1, -8], [2, 3], [9999, 9999]],
                 qacc=[[1, -6], [2, 3], [9999, 9999]])
        path, _, _ = aggregate(self.root, evaluation=self.evaluation(
            tracking_thresholds={"max_err_sync_m": 0.01}
        ))
        summary = path.with_name("summary.md").read_text()
        self.assertIn("| `peak_abs_applied_torque_nm` | N·m | 1 | 8 | `run` |", summary)
        self.assertIn("| `peak_abs_qacc_rad_s2` | rad/s² | 1 | 6 | `run` |", summary)
        self.assertIn('Explicit tracking thresholds: `{"max_err_sync_m": 0.01}`', summary)
        self.assertIn("| `tracking_failure` | 1 | 1 | 0 | `run` |", summary)

    def test_summary_follows_explicit_csv_output_and_handles_empty_batch(self):
        path, rows, _ = aggregate(self.root, output=self.root / "tables/custom.csv")
        self.assertEqual(rows, [])
        self.assertTrue(path.with_name("summary.md").is_file())
        summary = path.with_name("summary.md").read_text()
        self.assertIn("| TCP | B>0 inside X | NA | 0 | 0 | NA | NA | NA |", summary)
        self.assertFalse((self.root / "summary.md").exists())

    def test_compact_summary_converts_units_and_uses_common_valid_population(self):
        self.run_dir("a", mse_sync_m2=0.0001, dtw_mean_m=0.001, geom_dev_max_m=0.001)
        self.run_dir("b", mse_sync_m2=0.0003, dtw_mean_m=0.003, geom_dev_max_m=0.003)
        self.run_dir("c", "incomplete", mse_sync_m2=100, dtw_mean_m=100, geom_dev_max_m=100)
        self.run_dir("d", dtw_mean_m=None)
        path, _, _ = aggregate(self.root)
        summary = path.with_name("summary.md").read_text()
        self.assertIn("| test/model | FR3 | pic | 2 | 200.000 ± 141.421 | 2.000 ± 1.414 | 2.000 ± 1.414 |", summary)
        self.assertIn("| DS–TCP MSE | mm² | 100 | `a` | 300 | `b` |", summary)
        self.assertIn("| DTW | mm | 1 | `a` | 3 | `b` |", summary)
        self.assertIn("| Max geometric deviation | mm | 1 | `a` | 3 | `b` |", summary)

    def test_acceleration_reference_is_strict_and_unknown_is_not_none(self):
        for name, peak in (("at", 10), ("above", 10.1)):
            directory = self.run_dir(name)
            np.savez(directory / "traj.npz", t=[0, 0.1], qacc=[[-peak], [9999]])
        self.run_dir("unknown")
        path, _, _ = aggregate(self.root)
        summary = path.with_name("summary.md").read_text()
        self.assertIn("| FR3 acceleration reference exceeded in simulation (>10 rad/s²) | 1 | 2 | 1 | `above` |", summary)
        self.assertIn("| `goal_failure` | 0 | 3 | 0 | keine |", summary)

    def test_sampled_barrier_checks_stay_separate_and_preserve_raw_metrics(self):
        self.run_dir("a")
        model = self.root / "model_b.pth"
        model.touch()
        report = {
            "start_id": "a", "status": "evaluated",
            "paths": {
                "ideal": {"positive_samples_inside": 0, "first_positive_inside_s": None,
                          "box_exit_samples": 2, "first_box_exit_s": 0.1},
                "tcp": {"positive_samples_inside": 1, "first_positive_inside_s": 0.4,
                        "box_exit_samples": 0, "first_box_exit_s": None},
            },
        }
        before = (self.root / "a/metrics.json").read_bytes()
        with patch("evaluation.barrier.evaluate_run", return_value=report):
            path, rows, _ = aggregate(self.root, barrier_model=model)
        summary = path.with_name("summary.md").read_text()
        self.assertIn("| Ideal DS | B>0 inside X | 0 | 1 | 0 | 0 | keine | keine |", summary)
        self.assertIn("| TCP | B>0 inside X | 1 | 1 | 0 | 1 | `a` | `a` at 0.4 s |", summary)
        self.assertIn("| Ideal DS | Box exit | 1 | 1 | 0 | 2 | `a` | `a` at 0.1 s |", summary)
        self.assertEqual(json.loads(path.with_name("barrier.json").read_text())["runs"], [report])
        self.assertIsNone(rows[0]["barrier_exit"])
        self.assertEqual((self.root / "a/metrics.json").read_bytes(), before)

    def test_existing_barrier_flag_is_retained_without_inventing_sampled_checks(self):
        self.run_dir(barrier_exit=True)
        path, rows, _ = aggregate(self.root)
        self.assertTrue(rows[0]["barrier_exit"])
        summary = path.with_name("summary.md").read_text()
        self.assertIn("| `barrier_exit` | 1 | 1 | 0 | `run` |", summary)
        self.assertIn("| TCP | B>0 inside X | NA | 0 | 1 | NA | NA | NA |", summary)


if __name__ == "__main__":
    unittest.main()
