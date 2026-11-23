"""Check native learned-field timing through Euler rollout and both controllers."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

import main
from sim.controllers import pic, servo
from sim.ds import LearnedDS, PlanarEmbedding, demo_time_rate, ideal_rollout, load_scenario
from sim.robot import linear_velocity
from sim.sim import build_case


class TimeRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = main.load_config("configs/five_obstacle_pic.yaml", ["sim.T_max=0.004"])
        cls.case = build_case(cls.cfg)

    def test_nominal_pair_uses_native_rate_and_retains_demo_metadata(self):
        pic_cfg = main.load_config("configs/five_obstacle_pic.yaml")
        servo_cfg = main.load_config("configs/five_obstacle_servo.yaml")
        self.assertEqual(pic_cfg["time_rate"], 1.0)
        self.assertEqual(pic_cfg["ds_speed_factor"], 1.0)
        pic_cfg.pop("controller")
        servo_cfg.pop("controller")
        self.assertEqual(pic_cfg, servo_cfg)
        self.assertEqual(self.case.rate, 1.0)
        self.assertEqual(self.case.speed_factor, 1.0)
        old_cfg = deepcopy(self.cfg)
        old_cfg["time_rate"] = "demo"
        old = load_scenario(old_cfg)
        self.assertEqual(old["rate"], demo_time_rate(self.case.sc))
        self.assertEqual(old["sc"].ps_mm, self.case.sc.ps_mm)
        self.assertEqual(old["sc"].vs_mm_s, self.case.sc.vs_mm_s)
        np.testing.assert_array_equal(old["x0"], self.case.x0_n)
        np.testing.assert_array_equal(old["emb"].E, self.case.emb.E)
        self.assertEqual(old["emb"].L, self.case.emb.L)
        for point in ([0.7, -0.2], self.case.x0_n):
            np.testing.assert_array_equal(old["ds"].f_n(point), self.case.ds.f_n(point))

    def test_rate_one_euler_uses_field_at_previous_ideal_state(self):
        c = self.case
        expected = [c.x0_n.copy()]
        for _ in range(8):
            expected.append(expected[-1] + c.sp.dt * c.ds.f_n(expected[-1]))
        actual = ideal_rollout(c.ds, c.x0_n, c.sp.dt, 8, rate=1.0)
        np.testing.assert_array_equal(actual, expected)
        # The three-dimensional framework path must use the same Euler convention.
        ds3 = object.__new__(LearnedDS)
        ds3.f = lambda x: -np.asarray(x)
        x0 = np.array([0.4, -0.2, 0.1])
        expected3 = [x0]
        for _ in range(3):
            expected3.append(expected3[-1] + 0.01 * (-expected3[-1]))
        np.testing.assert_array_equal(ideal_rollout(ds3, x0, 0.01, 3, 1.0), expected3)

    def test_world_velocity_uses_embedding_scale_and_axes_without_demo_rate(self):
        c = self.case
        emb = PlanarEmbedding([0.4, -0.1, 0.6], [1, 1, 0], [0, 0, 1], 0.17, [0.1, -0.2])
        x = np.array([0.65, -0.25])
        p = emb.to_world(x)
        expected = emb.L * (emb.E @ c.ds.f_n(x))
        np.testing.assert_allclose(linear_velocity(c.ds, emb, p, 1.0, c.sp.k_perp),
                                   expected, atol=1e-15, rtol=0)
        shifted = p + 0.02 * emb.normal
        np.testing.assert_allclose(linear_velocity(c.ds, emb, shifted, 1.0, c.sp.k_perp),
                                   expected - c.sp.k_perp * 0.02 * emb.normal,
                                   atol=1e-15, rtol=0)

    def test_pic_receives_native_world_velocity_at_measured_tcp(self):
        c = self.case
        captured = []
        original = pic.PICController.step

        def step(controller, xd, f, dt):
            captured.append(f.copy())
            return original(controller, xd, f, dt)

        with patch.object(pic.PICController, "step", new=step):
            tr = pic.run(c)["traj"]
        for k, actual in enumerate(captured):
            x, height = c.emb.to_plane(tr["P"][k])
            expected = c.emb.L * (c.emb.E @ c.ds.f_n(x))
            expected -= c.sp.k_perp * height * c.emb.normal
            np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)
        np.testing.assert_array_equal(tr["P_ideal"],
                                      c.emb.to_world(ideal_rollout(c.ds, c.x0_n, c.sp.dt, 4, 1.0)))

    def test_servo_ik_receives_same_native_world_velocity(self):
        c = self.case
        captured = []
        original = servo.solve_qdot

        def solve(J, velocity, *args, **kwargs):
            captured.append(velocity.copy())
            return original(J, velocity, *args, **kwargs)

        with patch.object(servo, "solve_qdot", side_effect=solve):
            tr = servo.run(c)["traj"]
        for k, actual in enumerate(captured):
            x, height = c.emb.to_plane(tr["P"][k])
            expected = c.emb.L * (c.emb.E @ c.ds.f_n(x))
            expected -= c.sp.k_perp * height * c.emb.normal
            np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)
        np.testing.assert_array_equal(tr["P_ideal"],
                                      c.emb.to_world(ideal_rollout(c.ds, c.x0_n, c.sp.dt, 4, 1.0)))


if __name__ == "__main__":
    unittest.main()
