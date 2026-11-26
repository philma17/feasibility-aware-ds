"""Validated nominal IK candidates and replayable initial states."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import mujoco
import numpy as np

from .robot import FR3, MENAGERIE_DIR, IKResult, multistart_ik, scaled_sigma_min, servo_task_rows

NO_IK = "keine gültige IK-Lösung gefunden"
NO_PAIR = "kein ausreichend verschiedenes gültiges Paar gefunden"


class StartUnavailable(ValueError):
    def __init__(self, row):
        self.row = row
        super().__init__(row["status"])


class StartProblem:
    def __init__(self, case, start_id):
        cfg, robot, scenario = case.cfg, case.robot, case.scenario
        target, p0, sp = case.tgt, case.p0_world, case.sp
        if not np.isfinite(p0).all() or type(sp.n_starts) is not int or sp.n_starts < 1:
            raise ValueError("Nominal IK needs a finite target and a positive integer number of seeds")
        self.robot = robot
        self.scenario = scenario
        self.target = target
        self.p0 = p0
        self.sp = sp
        self.start_id = start_id
        self.q_ref = robot.reference_posture()
        self.position_mode = case.position_mode
        self.orientation = cfg["task"].get("orientation", "none")
        self.rotational_damping_task = self.orientation == "damped" and cfg["controller"] == "pic"
        digest = sha256(start_id.encode()).digest()
        self.seed = int(np.random.SeedSequence([sp.seed, int.from_bytes(digest[:8], "big")])
                        .generate_state(1)[0])
        spec, emb = robot.spec, case.emb
        tcp = robot.model.site(spec.site)
        contact_enabled = not robot.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.check_contacts = bool(contact_enabled and
                                   (robot.model.npair or np.any(robot.model.geom_contype)))
        self.compatibility = {
            "robot": spec.name, "mjcf": spec.mjcf,
            "mjcf_sha256": sha256((Path(MENAGERIE_DIR) / spec.mjcf).read_bytes()).hexdigest(),
            "joint_names": list(spec.joints), "q_min_rad": robot.q_min.tolist(),
            "q_max_rad": robot.q_max.tolist(), "fixed_joints_rad": dict(spec.fixed_joints),
            "tcp_site": spec.site, "tcp_body": robot.model.body(robot.site_body).name,
            "site_pos_m": tcp.pos.tolist(), "site_quat_wxyz": tcp.quat.tolist(),
            "model": cfg["model"], "embedding": {"origin": emb.origin.tolist(), "axes": emb.E.tolist(),
                                                "L_m": emb.L, "anchor_n": emb.anchor_n.tolist()},
            "out_of_plane_offset_m": cfg["task"].get("out_of_plane_offset_m", 0.0),
            "position_mode": self.position_mode, "orientation": self.orientation,
            "axis_local": target.axis_local.tolist() if target.mode == "axis5" else None,
            "direction_world": None if target.a_des is None else target.a_des.tolist(),
            "R_des": None if target.R_des is None else target.R_des.tolist(),
            "position_tol_m": sp.start_position_tol_m, "orientation_tol_rad": sp.start_orientation_tol_rad,
            "sigma_rotation_length_m": sp.sigma_rotation_length_m,
            "rotational_damping_task": self.rotational_damping_task,
            "collision_check": "MuJoCo contacts" if self.check_contacts else "unavailable",
            "tcp_unsafe_set": scenario["unsafe_info"],
            "tcp_obstacle_check": "level > 0" if scenario["obstacle_level"] else "unavailable",
        }

    def valid_candidate(self, q):
        if not self.robot.valid_angles(q):
            return False
        p, _ = self.robot.fk(q)
        if self.check_contacts:
            mujoco.mj_forward(self.robot.model, self.robot.data)
            if self.robot.data.ncon:
                return False
        level = self.scenario["obstacle_level"]
        if level is None:
            return True
        x, _ = self.scenario["emb"].to_plane(p)
        return bool(level(np.atleast_2d(x))[0] > 0)

    def measure(self, q):
        p, R = self.robot.fk(q)
        delta = self.p0 - p
        J6 = self.robot.jac()
        rows, _, ori = servo_task_rows(J6, R, self.target, delta, 1.0,
                                       self.scenario["emb"], self.position_mode)
        rows = J6 if self.rotational_damping_task else rows
        sigma = scaled_sigma_min(rows, self.sp.sigma_rotation_length_m)
        return p, float(np.linalg.norm(delta)), ori, sigma

    def record(self, q=None, *, status="valid"):
        row = {
            "version": 1, "start_id": self.start_id, "status": status,
            "x0_n": self.scenario["x0"].tolist(), "tcp_start_m": self.p0.tolist(),
            "tcp_actual_m": None, "q_start_rad": None, "qd_start_rad_s": None,
            "q_ref_rad": self.q_ref.tolist(), "q_ref_source": self.robot.spec.mjcf + "#home",
            "ik_seed": self.seed, "base_ik_seed": self.sp.seed, "ik_n_starts": self.sp.n_starts,
            "start_criterion": "closest_to_reference", "compatibility": self.compatibility,
        }
        if q is not None:
            p, pe, oe, sigma = self.measure(q)
            full_order = FR3.joints if self.robot.spec.family == "FR3" else self.robot.spec.joints
            row.update(q_start_rad=q.tolist(), qd_start_rad_s=np.zeros_like(q).tolist(),
                       full_joint_names=list(full_order), q_start_full_rad=self.robot.full_angles(q),
                       effective_fixed_joints_rad=self.robot.effective_locks(q), tcp_actual_m=p.tolist(),
                       position_error_m=pe, orientation_error_rad=oe, sigma_min_start=sigma)
        return row

    def from_angles(self, q):
        q = np.asarray(q, float)
        if not self.valid_candidate(q):
            raise ValueError("Invalid start angles or start in collision")
        _, pe, oe, sigma = self.measure(q)
        if pe >= self.sp.start_position_tol_m or oe >= self.sp.start_orientation_tol_rad:
            raise ValueError(f"Start violates task tolerance ({pe:g} m, {oe:g} rad)")
        return IKResult(q, pe, oe, True, 1, 0, sigma, float(self.robot.limit_margin_pct(q).min()))

    def replay(self, row):
        if row["version"] != 1 or row["compatibility"] != self.compatibility:
            raise ValueError("Start row is incompatible with the current robot, placement or task")
        if row["start_id"] != self.start_id or not np.array_equal(row["q_ref_rad"], self.q_ref):
            raise ValueError("Start identity or reference posture differs")
        if not np.array_equal(row["tcp_start_m"], self.p0):
            raise ValueError("Stored TCP target differs from placed start")
        self.seed = row["ik_seed"]
        if row["status"] in (NO_IK, NO_PAIR):
            raise StartUnavailable(row)
        if row["status"] != "valid":
            raise ValueError(f"Unknown saved start status: {row['status']!r}")
        if not np.array_equal(row["qd_start_rad_s"], np.zeros(self.robot.n)):
            raise ValueError("Only zero initial joint velocity is supported")
        result = self.from_angles(row["q_start_rad"])
        measured = self.record(result.q)
        for field in ("full_joint_names", "effective_fixed_joints_rad"):
            if row[field] != measured[field]:
                raise ValueError(f"Start row differs from current model: {field}")
        for field in ("tcp_actual_m", "sigma_min_start", "q_start_full_rad"):
            if np.shape(row[field]) != np.shape(measured[field]) or not np.allclose(
                row[field], measured[field], atol=1e-12, rtol=0
            ):
                raise ValueError(f"Start row differs from current model: {field}")
        angles = dict(zip(measured["full_joint_names"], measured["q_start_full_rad"]))
        for joint, prescribed in row.get("prescribed_joints_rad", {}).items():
            if angles.get(joint) != prescribed:
                raise ValueError(f"Start violates prescribed angle for {joint}")
        return result

    def solve(self, also_valid=None):
        s = self.sp
        def valid(q):
            return self.valid_candidate(q) and (also_valid is None or also_valid(q))

        result = multistart_ik(self.robot, self.p0, self.target, self.q_ref, s.n_starts,
                               self.seed, "closest_to_reference", tol_conv=s.start_position_tol_m,
                               ori_tol=s.start_orientation_tol_rad, valid_candidate=valid)
        if not result.converged:
            raise StartUnavailable(self.record(status=NO_IK))
        result.sigma_min = self.measure(result.q)[-1]
        return result

    def lower_sigma(self, result, gap):
        """Stable choice among the same validated candidates, in measured task units."""
        distinct = [q for q in result.solutions if self.robot.distance_squared(q, result.q) > 1e-16]
        if not distinct:
            return self.record(status=NO_PAIR)
        best = min(distinct, key=lambda q: self.measure(q)[-1])
        sigma = self.measure(best)[-1]
        if result.sigma_min - sigma >= gap:
            return self.record(best)
        return {**self.record(status=NO_PAIR), "lowest_distinct_candidate_sigma_min": sigma}
