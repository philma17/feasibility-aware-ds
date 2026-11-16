"""Check task-plane projection independently of world XY and simulation state."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evaluation.plots import draw_trajectories, plt, select_batch_runs, task_plane_mm


class TaskPlaneTests(unittest.TestCase):
    def test_rotated_plane_uses_saved_axes_and_origin(self):
        # Task axes are world +Z and -X. The world Y offset is out of plane.
        embedding = {"origin": [1, 2, 3], "axes": [[0, -1], [0, 0], [1, 0]]}
        points = [[0.98, 2.7, 3.01], [1.03, 2.0, 2.96]]
        np.testing.assert_allclose(task_plane_mm(points, embedding), [[10, 20], [-40, -30]])

    def test_spatial_embedding_is_not_silently_projected_to_world_xy(self):
        embedding = {"origin": [0, 0, 0], "axes": np.eye(3)}
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            task_plane_mm([[0, 0, 0]], embedding)


class BatchPlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        with (self.root / "runs.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "start_id", "run_dir", "domain_exit", "barrier_exit", "torque_saturation_steps"
            ])
            writer.writeheader()
            for i, value in enumerate(("true", "false", "NA", "true")):
                writer.writerow({"start_id": f"s{i}", "run_dir": f"run{i}",
                                 "domain_exit": value, "barrier_exit": "NA",
                                 "torque_saturation_steps": "2" if i == 0 else "NA"})

    def test_event_filter_excludes_false_and_na_and_resolves_relative_paths(self):
        self.assertEqual(select_batch_runs(self.root, event="domain_exit"), [
            ("s0", self.root / "run0"), ("s3", self.root / "run3")
        ])
        self.assertEqual(select_batch_runs(self.root, event="torque_saturation"), [
            ("s0", self.root / "run0")
        ])
        with self.assertRaisesRegex(ValueError, "evaluated: 0/4"):
            select_batch_runs(self.root, event="barrier_exit")

    def test_id_selection_is_explicit_and_unknown_ids_are_rejected(self):
        self.assertEqual(select_batch_runs(self.root, ["s2", "s0", "s0"]), [
            ("s0", self.root / "run0"), ("s2", self.root / "run2")
        ])
        for kwargs in ({}, {"start_ids": ["missing"]}, {"start_ids": ["s0"], "event": "domain_exit"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_batch_runs(self.root, **kwargs)

    def plot_run(self, start_id):
        return {
            "start_id": start_id, "run_dir": str(self.root / start_id),
            "manifest": {"model": "test", "robot": "FR3"},
            "embedding": {"origin": [1, 2, 3], "axes": [[1, 0], [0, 1], [0, 0]],
                          "L_m": 0.1, "anchor_n": [0, 0]},
            "displayed_axes": [0, 1], "package": {"shape": "Test"},
            "cfg": {}, "spec": {"domain": {"range": [[-1, 1], [-1, 1]]}},
            "ideal_world": np.array([[1, 2, 3], [1.01, 2.02, 3]]),
            "measured_world": np.array([[1, 2, 3], [1.02, 2.01, 3]]),
            "barrier_model": None, "plot_inputs": {},
        }

    def test_overlay_preserves_world_coordinates_goal_and_path_styles(self):
        from PIL import Image

        runs = [self.plot_run("s0"), self.plot_run("s3")]
        fig, ax = plt.subplots()
        path = self.root / "plots/selected.png"
        with patch("evaluation.plots.experiment_setting", return_value="Nominal"), patch(
            "evaluation.plots.plt.subplots", return_value=(fig, ax)
        ):
            draw_trajectories(runs, path)
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1000, 1010])
        np.testing.assert_array_equal(ax.lines[-1].get_data(), [[1000], [2000]])
        self.assertEqual(ax.lines[0].get_linestyle(), "--")
        self.assertEqual(ax.lines[1].get_linestyle(), "-")
        self.assertEqual(ax.lines[0].get_color(), ax.lines[1].get_color())
        self.assertNotEqual(ax.lines[0].get_color(), ax.lines[2].get_color())
        self.assertEqual(ax.get_aspect(), 1)
        with Image.open(path) as image:
            self.assertEqual([r["start_id"] for r in json.loads(image.info["Description"])["runs"]], ["s0", "s3"])

    def test_overlay_rejects_inconsistent_embedding(self):
        a, b = self.plot_run("s0"), self.plot_run("s3")
        b["embedding"]["origin"][0] = 2
        with self.assertRaisesRegex(ValueError, "same model"), patch(
            "evaluation.plots.experiment_setting", return_value="Nominal"
        ):
            draw_trajectories([a, b], self.root / "never.png")
        self.assertFalse((self.root / "never.png").exists())


if __name__ == "__main__":
    unittest.main()
