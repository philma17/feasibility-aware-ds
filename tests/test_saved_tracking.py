"""The accelerated saved-path evaluation must preserve exact segment semantics."""
import unittest

import numpy as np

from evaluation.metrics import _point_to_polyline_tree


def brute_distance(X, P):
    a, ab = P[:-1], np.diff(P, axis=0)
    d = X[:, None] - a
    t = np.clip(np.einsum('kij,ij->ki', d, ab) /
                np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-300), 0, 1)
    return np.sqrt(np.sum((X[:, None] - (a + t[..., None] * ab))**2, axis=2).min(axis=1))


class SegmentDistanceTests(unittest.TestCase):
    def test_random_and_stationary_paths(self):
        rng = np.random.default_rng(819)
        for dim in (2, 3):
            p = rng.normal(size=(160, dim)).cumsum(axis=0)
            p[40:80] = p[40]
            x = np.concatenate([rng.normal(size=(120, dim)) * 15, p])
            np.testing.assert_allclose(_point_to_polyline_tree(x, p), brute_distance(x, p), rtol=1e-13, atol=1e-14)

    def test_long_segment_interior_and_tiny_tail(self):
        p = np.array([[0., 0., 0.], [100., 0., 0.], [100., 1e-12, 0.]])
        x = np.array([[50., 1., 0.], [100., 5e-13, 1e-14]])
        np.testing.assert_allclose(_point_to_polyline_tree(x, p), brute_distance(x, p), rtol=1e-13, atol=1e-15)


if __name__ == '__main__':
    unittest.main()
