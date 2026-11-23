"""Initial-state construction and the shared MuJoCo interval schedule."""

from __future__ import annotations
from dataclasses import dataclass, field
import mujoco
import numpy as np
from .ds import load_scenario
from .fr3_variants import MASS_VARIANTS
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
    ORIENTATION_MODES,
    R_DES_DEFAULT,
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
    q_start: np.ndarray
    ik: object
    robot_nominal: Robot | None = None
    tau_limit: np.ndarray | None = None
    speed_factor: float = 1.0
    plant_report: dict | None = None
    unsafe: object = None
    scenario: dict | None = None
    orient: OrientParams | None = None
    p0_world: np.ndarray | None = None
    position_mode: str = "xyz"


def build_case(
    cfg: dict, friction_scale: float | None = None, speed_factor: float | None = None
) -> Case:
    """Build the configured plant and map the task start to a joint state."""
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
    robot = Robot(ROBOTS[name])
    plant_report = apply_plant(robot, plant)
    # Same selected MJCF, but without the plant perturbations.
    robot_nominal = Robot(ROBOTS[name])
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
    t = cfg["task"]
    orient_mode = t.get("orientation", "none")
    position_mode = validate_position_mode(t.get("position_mode", "xyz"), emb, orient_mode)
    if orient_mode not in ORIENTATION_MODES:
        raise ValueError(
            f"task.orientation must be one of {ORIENTATION_MODES}, got {orient_mode!r}"
        )
    R_des = np.asarray(t.get("R_des", R_DES_DEFAULT), float)
    tgt = Target("pos")
    if orient_mode == "hold":
        tgt = Target("full6", R_des=R_des)
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
    if t.get("q_start_rad") is not None:
        # Preserve a supplied IK branch; report its TCP error without re-solving it.
        q0 = np.asarray(t["q_start_rad"], float)
        if q0.shape != (robot.n,):
            raise ValueError(f"task.q_start_rad needs {robot.n} values")
        if np.any(q0 < robot.q_min) or np.any(q0 > robot.q_max):
            raise ValueError("task.q_start_rad violates the joint position limits")
        p_fk, R_fk = robot.fk(q0)
        _, _, oe0 = task_rows(robot.jac(), R_fk, tgt, p0 - p_fk, 1.0)
        ikr = IKResult(
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
    else:
        # MuJoCo's COM-based Jacobian computation can change by roundoff after
        # a density change. Use the identical reference model for start IK so
        # paired density experiments retain exactly the same initial angles.
        ik_robot = robot
        if name in MASS_VARIANTS:
            ik_robot = Robot(ROBOTS["FR3"])
            apply_plant(ik_robot, plant)
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
    orient = orientation_params(robot, ikr.q, t)
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
        ikr.q.copy(),
        ikr,
        robot_nominal=robot_nominal,
        tau_limit=torque_limit(robot.spec.name, plant["torque_limit_scale"]),
        speed_factor=alpha,
        plant_report=plant_report,
        unsafe=scn["unsafe"],
        scenario=scn,
        orient=orient,
        p0_world=np.asarray(p0, float).copy(),
        position_mode=position_mode,
    )


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
