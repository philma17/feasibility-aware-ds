"""Initial-state construction and the shared MuJoCo interval schedule."""

from __future__ import annotations
from dataclasses import dataclass, field
import mujoco
import numpy as np
from .ds import load_scenario
from .fr3_variants import MASS_VARIANTS
from .starts import StartProblem
from .robot import (
    Robot,
    ROBOTS,
    IKParams,
    IKResult,
    Target,
    multistart_ik,
    task_rows,
    task_sigma_min,
    resolve_plant,
    apply_plant,
    torque_limit,
    validate_position_mode,
    orientation_params,
    OrientParams,
    orientation_target,
)


@dataclass
class SimParams:
    dt: float = 0.001
    T_max: float = 30.0
    k_perp: float = 5.0
    k_rot: float = 5.0
    success_frac: float = 0.01
    n_starts: int = 32
    seed: int = 0
    start_criterion: str = "nearest_ref"
    start_min_margin_pct: float = 5.0
    start_position_tol_m: float = 1e-6
    start_orientation_tol_rad: float = 1e-6
    sigma_rotation_length_m: float = 1.0
    ik: IKParams = field(default_factory=IKParams)


def sim_params(cfg):
    values = dict(cfg["sim"])
    values.pop("clear_every", None)  # Retained in resolved YAML for legacy run hashes.
    return SimParams(ik=IKParams(**values.pop("ik")), **values)


@dataclass
class Case:
    """Shared robot, learned field, embedding, and initial-state inputs."""

    cfg: dict
    robot: Robot
    ds: object
    sc: object
    rate: float
    sp: object
    emb: object
    tgt: Target
    x0_n: np.ndarray
    q_start: np.ndarray | None = None
    ik: IKResult | None = None
    robot_nominal: Robot | None = None
    tau_limit: np.ndarray | None = None
    speed_factor: float = 1.0
    plant_report: dict | None = None
    unsafe: object = None
    scenario: dict | None = None
    orient: OrientParams | None = None
    p0_world: np.ndarray | None = None
    position_mode: str = "xyz"
    starts: StartProblem | None = None
    initial_state: dict | None = None

    def observe_start(self, q, qd, tcp):
        """Capture state zero from the controller before it issues the first command."""
        if not np.array_equal(q, self.q_start) or np.any(qd != 0):
            raise ValueError("Controller did not initialize at the resolved start")
        state = {"q_rad": q.tolist(), "qd_rad_s": qd.tolist(), "tcp_m": tcp.tolist()}
        if self.starts is not None:
            state["effective_fixed_joints_rad"] = self.robot.effective_locks(q)
        self.initial_state = state


def prepare_case(
    cfg: dict, friction_scale: float | None = None, speed_factor: float | None = None,
) -> Case:
    """Resolve the plant and Cartesian task, before choosing a joint posture."""
    name = cfg["robot"]
    if name not in ROBOTS:
        raise ValueError(f"unknown robot {name!r}; registered: {sorted(ROBOTS)}")
    if cfg["task"]["mode"] != "pos":
        raise NotImplementedError(
            "task.mode must be 'pos' (orientation is chosen with task.orientation)"
        )
    plant = dict(cfg.get("plant") or {})
    if friction_scale is not None:
        plant["dof_frictionloss_scale"] = float(friction_scale)
    plant = resolve_plant(plant)
    t = cfg["task"]
    fixed = t.get("fixed_joints_rad")
    robot = Robot(ROBOTS[name], fixed_joints_rad=fixed)
    plant_report = apply_plant(robot, plant)
    # Same selected MJCF, but without the plant perturbations.
    robot_nominal = Robot(ROBOTS[name], fixed_joints_rad=fixed)
    alpha = float(cfg.get("ds_speed_factor", 1.0) if speed_factor is None else speed_factor)
    if not (np.isfinite(alpha) and alpha > 0):
        raise ValueError(f"ds_speed_factor must be finite and > 0, got {alpha}")
    scn = load_scenario(cfg)
    ds, sc, emb, x0 = (scn["ds"], scn["sc"], scn["emb"], scn["x0"])
    rate = scn["rate"] * alpha
    sp = sim_params(cfg)
    if not np.isfinite(sp.dt) or sp.dt <= 0:
        raise ValueError(f"sim.dt must be finite and positive, got {sp.dt}")
    robot.model.opt.timestep = sp.dt
    robot_nominal.model.opt.timestep = sp.dt
    if not all(
        np.isfinite(value) and value > 0
        for value in (sp.start_position_tol_m, sp.start_orientation_tol_rad, sp.sigma_rotation_length_m)
    ):
        raise ValueError("Start tolerances and sigma_rotation_length_m must be finite and positive")
    orient_mode = t.get("orientation", "none")
    position_mode = validate_position_mode(t.get("position_mode", "xyz"), emb, orient_mode)
    tgt = orientation_target(t, emb)
    p0 = emb.to_world(x0)
    offset = float(t.get("out_of_plane_offset_m", 0.0))
    if offset != 0.0:
        if not np.isfinite(offset):
            raise ValueError("task.out_of_plane_offset_m must be finite")
        if not np.any(emb.normal):
            raise ValueError(
                "task.out_of_plane_offset_m needs a 2D model (planar embedding); 3D models already use all three Cartesian components"
            )
        p0 = p0 + offset * emb.normal
    return Case(
        cfg,
        robot,
        ds,
        sc,
        rate,
        sp,
        emb,
        tgt,
        x0,
        robot_nominal=robot_nominal,
        tau_limit=torque_limit(robot.spec.name, plant["torque_limit_scale"]),
        speed_factor=alpha,
        plant_report=plant_report,
        unsafe=scn["unsafe"],
        scenario=scn,
        p0_world=np.asarray(p0, float).copy(),
        position_mode=position_mode,
    )


def replay_config(cfg, row):
    return {**cfg, "task": {**cfg["task"], "start": {"x0": row["x0_n"]}}}


def build_case(cfg, friction_scale=None, speed_factor=None, *, start_id=None, start_row=None):
    if start_row is not None:
        cfg = replay_config(cfg, start_row)
    case = prepare_case(cfg, friction_scale, speed_factor)
    if case.sp.start_criterion == "closest_to_reference" or start_row is not None:
        if start_id is None:
            start = cfg["task"].get("start")
            index = int(start.get("index", 0)) if isinstance(start, dict) else 0
            start_id = start_row["start_id"] if start_row is not None else f"start_{index:03d}"
        case.starts = StartProblem(case, start_id)
        ik = case.starts.solve() if start_row is None else case.starts.replay(start_row)
    else:
        ik = _legacy_start(case)
    case.ik, case.q_start = ik, ik.q.copy()
    gain_q = ik.q
    if start_row is not None and "orientation_gain_q_rad" in start_row:
        gain_q = np.asarray(start_row["orientation_gain_q_rad"], float)
        case.starts.from_angles(gain_q)
    case.orient = orientation_params(case.robot, gain_q, cfg["task"])
    return case


def _legacy_start(case):
    t, name = case.cfg["task"], case.robot.spec.name
    robot, sp, p0, tgt = case.robot, case.sp, case.p0_world, case.tgt
    q_start = t.get("q_start_rad")
    q_start_robot = t.get("q_start_robot")
    if (
        q_start is not None
        and q_start_robot is not None
        and ROBOTS[q_start_robot].joints != robot.spec.joints
    ):
        q_start = None
    if q_start is not None:
        # Preserve a supplied IK branch; report its TCP error without re-solving it.
        q0 = np.asarray(q_start, float)
        if q0.shape != (robot.n,):
            raise ValueError(f"task.q_start_rad needs {robot.n} values")
        if not robot.valid_angles(q0):
            raise ValueError("task.q_start_rad violates the joint position limits")
        p_fk, R_fk = robot.fk(q0)
        _, _, oe0 = task_rows(robot.jac(), R_fk, tgt, p0 - p_fk, 1.0)
        if tgt.mode == "axis5" and (np.linalg.norm(p0 - p_fk) >= sp.start_position_tol_m
                                   or oe0 >= sp.start_orientation_tol_rad):
            raise ValueError("task.q_start_rad does not satisfy the axis5 start task")
        return IKResult(
            q0,
            float(np.linalg.norm(p0 - p_fk)),
            float(oe0),
            True,
            1,
            0,
            task_sigma_min(robot, q0, tgt, np.ones(robot.n, bool)),
            float(robot.limit_margin_pct(q0).min()),
            True,
        )

    # MuJoCo's COM-based Jacobian computation can change by roundoff after
    # a density change. Use the identical reference model for start IK so
    # paired density experiments retain exactly the same initial angles.
    ik_robot = robot
    if name in MASS_VARIANTS:
        ik_robot = Robot(ROBOTS["FR3"])
        apply_plant(ik_robot, case.plant_report["requested"])
    ikr = multistart_ik(
        ik_robot,
        p0,
        tgt,
        ik_robot.q_home,
        sp.n_starts,
        sp.seed,
        sp.start_criterion,
        free=None,
        min_margin_pct=sp.start_min_margin_pct,
    )
    if not ikr.converged:
        raise RuntimeError(
            f"start IK did not converge (pos err {ikr.pos_err:.3e} m, ori err {ikr.ori_err:.3e} rad)"
        )
    return ikr


def first_contact_at(model, data, time_s):
    """Record the first active contact, with body names under the saved geoms key."""
    contact = data.contact[0]
    return {
        "t_s": time_s,
        "geoms": [
            model.body(int(model.geom_bodyid[geom])).name
            for geom in (contact.geom1, contact.geom2)
        ],
    }


def execute_intervals(
    model,
    data,
    n_steps,
    *,
    observe,
    command,
    after_step,
    n_substeps=1,
    forward_before_observe=False,
    environment=None,
):
    """Record N+1 states and apply N typed commands, each held for n_substeps.

    PIC requires fresh dynamics before observing each state. Servo keeps
    its single initial mj_forward outside this schedule.
    """
    for k in range(n_steps + 1):
        if forward_before_observe:
            mujoco.mj_forward(model, data)
        state = observe(k)
        if k == n_steps:
            # The final state is logged, but has no following control interval.
            return
        command(k, state).apply(data)
        for substep in range(n_substeps):
            if environment is None:
                mujoco.mj_step(model, data)
            else:
                with environment.physics_step(model, data, k * n_substeps + substep):
                    mujoco.mj_step(model, data)
        after_step(k)
