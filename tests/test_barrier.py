"""Check sampled Barrier events independently of simulation and checkpoint inference."""

import unittest
from unittest.mock import patch

import numpy as np

from evaluation.barrier import evaluate_paths, path_events, world_to_model
from evaluation.plots import draw_regions, plt


class BarrierTests(unittest.TestCase):
    def test_closed_box_and_strict_positive_samples_are_independent(self):
        points = [[-1, 0], [0, 0], [1, 0], [1.01, 0], [-1.01, 0]]
        result = path_events([0, 0.1, 0.2, 0.3, 0.4], points,
                             [-0.1, 0.2, 0, 0.3, 0.4], [[-1, 1], [-1, 1]])
        self.assertEqual(result["inside_samples"], 3)
        self.assertEqual(result["positive_samples_inside"], 1)
        self.assertEqual(result["first_positive_inside_s"], 0.1)
        self.assertEqual(result["box_exit_samples"], 2)
        self.assertEqual(result["first_box_exit_s"], 0.3)
        self.assertEqual(result["max_B_inside"], 0.2)
        self.assertEqual(result["max_B_outside"], 0.4)

    def test_zero_barrier_and_box_boundary_are_not_events(self):
        result = path_events([0, 1], [[-1, 0], [1, 0]], [0, -0.1], [[-1, 1], [-1, 1]])
        self.assertEqual(result["positive_samples_inside"], 0)
        self.assertIsNone(result["first_positive_inside_s"])
        self.assertIsNone(result["first_box_exit_s"])
        with self.assertRaisesRegex(ValueError, "finite"):
            path_events([0, 1], [[0, 0], [1, 0]], [0, np.nan], [[-1, 1], [-1, 1]])

    def test_inverse_embedding_uses_world_origin_scale_axes_and_anchor(self):
        embedding = {"origin": [1, 2, 3], "axes": [[0, -1], [0, 0], [1, 0]],
                     "L_m": 0.1, "anchor_n": [0.2, -0.3]}
        np.testing.assert_allclose(world_to_model([[0.98, 2.7, 3.01]], embedding), [[0.3, -0.1]])

    def test_ideal_and_tcp_are_evaluated_at_their_own_saved_states(self):
        embedding = {"origin": [1, 2, 3], "axes": [[1, 0], [0, 1], [0, 0]],
                     "L_m": 0.1, "anchor_n": [0, 0]}
        with patch("evaluation.barrier.barrier_values", side_effect=lambda _, x: x[:, 0]):
            result = evaluate_paths(
                [0, 0.1, 0.2], [[1.05, 2, 3], [1.2, 2, 3], [0.95, 2, 3]],
                [[0.95, 2, 3], [0.96, 2, 3], [0.97, 2, 3]], embedding,
                {"domain": {"range": [[-1, 1], [-1, 1]]}}, "test",
            )
        self.assertEqual(result["ideal"]["positive_samples_inside"], 0)
        self.assertEqual(result["ideal"]["box_exit_samples"], 0)
        self.assertEqual(result["tcp"]["positive_samples_inside"], 1)
        self.assertEqual(result["tcp"]["first_positive_inside_s"], 0)
        self.assertEqual(result["tcp"]["box_exit_samples"], 1)
        self.assertEqual(result["tcp"]["first_box_exit_s"], 0.1)

    def test_green_regions_and_contour_use_only_domain_and_world_box(self):
        embedding = {"origin": [1, 2, 3], "axes": [[1, 0], [0, 1], [0, 0]],
                     "L_m": 0.1, "anchor_n": [0, 0]}
        spec = {"domain": {"range": [[-1, 1], [-1, 1]]},
                "plotting": {"range": [[-2, 2], [-3, 3]]}, "Barrier": True}
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        with patch("evaluation.plots.barrier_values", side_effect=lambda _, x: x[:, 0]) as infer:
            draw_regions(ax, spec, embedding, [0, 1], [np.array([[1.3, 2.4, 3]])], "test")
        points = infer.call_args.args[1]
        np.testing.assert_array_equal(points.min(axis=0), [-1, -1])
        np.testing.assert_array_equal(points.max(axis=0), [1, 1])
        vertices = ax.patches[0].get_xy()
        np.testing.assert_allclose(vertices[:4], [[900, 1900], [1100, 1900],
                                                 [1100, 2100], [900, 2100]])
        for collection in ax.collections:
            self.assertIsNotNone(collection.get_clip_path())


if __name__ == "__main__":
    unittest.main()
