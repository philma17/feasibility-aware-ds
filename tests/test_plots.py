"""Check task-plane projection independently of world XY and simulation state."""

import unittest

import numpy as np

from evaluation.plots import task_plane_mm


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


if __name__ == "__main__":
    unittest.main()
