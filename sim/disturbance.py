"""World-frame environment forces, independent of the control law.

Pulse boundaries must be on the physics grid. At each physics interval the
current TCP force is represented by an equivalent COM wrench. Only our added
wrench is removed afterwards; other environment loads are preserved.
"""

from contextlib import contextmanager
import numpy as np
import mujoco


def apply_tcp_force(m, d, body, p_tcp, force):
    """Add world F and (p_tcp - COM) × F to a body's existing COM wrench."""
    d.xfrc_applied[body, :3] += force
    d.xfrc_applied[body, 3:] += np.cross(p_tcp - d.xipos[body], force)


class TCPForceEnvironment:
    def __init__(self, robot, pulses=()):
        self.robot = robot
        self.pulses = tuple(pulses)
        self.rows = []
        timestep = robot.model.opt.timestep
        for pulse in self.pulses:
            start, duration = pulse["t_start_s"], pulse["duration_s"]
            force = np.asarray(pulse["force_N"], float)
            if (
                not np.isfinite([start, duration]).all()
                or start < 0
                or duration <= 0
                or force.shape != (3,)
                or not np.isfinite(force).all()
            ):
                raise ValueError("Pulse needs finite start>=0, duration>0, and world force_N[3]")
            if any(
                not np.isclose(value / timestep, round(value / timestep), rtol=0, atol=1e-8)
                for value in (start, duration)
            ):
                raise ValueError("Pulse start and duration must lie on the physics timestep grid")
        self.windows = [
            (
                round(pulse["t_start_s"] / timestep),
                round((pulse["t_start_s"] + pulse["duration_s"]) / timestep),
                np.asarray(pulse["force_N"], float),
            )
            for pulse in self.pulses
        ]

    def force_at_step(self, step):
        return sum(
            (force for start, end, force in self.windows if start <= step < end),
            np.zeros(3),
        )

    def _state(self, m, d):
        # mj_step leaves position-derived arrays at the pre-integration pose.
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        point = d.site_xpos[self.robot.site].copy()
        if self.robot.p_off is not None:
            point += d.site_xmat[self.robot.site].reshape(3, 3) @ self.robot.p_off
        return point

    @contextmanager
    def physics_step(self, m, d, step):
        body = self.robot.site_body
        force = self.force_at_step(step)
        point = self._state(m, d)
        previous = d.xfrc_applied[body].copy()
        apply_tcp_force(m, d, body, point, force)
        added_wrench = d.xfrc_applied[body] - previous
        self.rows.append(
            (
                step * m.opt.timestep,
                point,
                force.copy(),
                added_wrench.copy(),
                d.qpos[self.robot.qadr].copy(),
                d.qvel[self.robot.dadr].copy(),
            )
        )
        try:
            yield
        finally:
            # Restore exactly, avoiding roundoff accumulation in unrelated loads.
            d.xfrc_applied[body] = previous

    def trajectory(self, m, d):
        rows = self.rows + [
            (
                len(self.rows) * m.opt.timestep,
                self._state(m, d),
                np.zeros(3),
                np.zeros(6),
                d.qpos[self.robot.qadr].copy(),
                d.qvel[self.robot.dadr].copy(),
            )
        ]
        return {
            "physics_" + key: np.asarray([row[i] for row in rows])
            for i, key in enumerate(("t", "P", "F_ext", "wrench_ext", "Q", "QD"))
        }
