"""Measured-state DS feedback with the source persistent position target."""

from __future__ import annotations
import mujoco
import numpy as np
from sim.ds import check_dims, ideal_rollout
from sim.robot import (
    rot_error,
    solve_qdot,
    servo_task_rows,
    validate_position_mode,
    position_servo_map,
    gravity_torque,
    TAU_MAX_DOC,
)
from sim.sim import execute_intervals as run_intervals
from . import Command


def _run_source(
    robot,
    ds,
    emb,
    x0_n,
    rate: float,
    tgt,
    sp,
    q_start,
    free=None,
    q_null=None,
    n_substeps: int = 1,
    contact_check: bool = True,
    gravity_ff: bool = False,
    tau_limit=None,
    position_mode="xyz",
    joint_control=None,
) -> dict:
    """Closed-loop DS execution with MuJoCo dynamics. Same arguments as sim.run (q_start is
    mandatory so that both execution modes start from the identical configuration).

    gravity_ff: start the servo target at q_start + g(q_start)/kp (static equilibrium of the
    PD servo at q_start), so the robot does not sag at t = 0. The robot itself still starts
    at q_start with zero velocity. Default False = previous baseline behavior.
    tau_limit: joint torque limits used for REPORTING only [Nm]; default TAU_MAX_DOC."""
    m = robot.model
    n = robot.n
    validate_position_mode(position_mode, emb, "none" if tgt.mode == "pos" else "hold")
    if not np.isclose(sp.dt, n_substeps * m.opt.timestep):
        raise ValueError(f"controller dt {sp.dt} != n_substeps * timestep {m.opt.timestep}")
    sv = position_servo_map(robot)
    free = np.ones(n, bool) if free is None else np.asarray(free, bool)
    N = int(round(sp.T_max / sp.dt))
    x0_n = np.asarray(x0_n, float)
    dim = check_dims(ds, emb, x0_n)
    X_id = ideal_rollout(ds, x0_n, sp.dt, N, rate)
    P_id = emb.to_world(X_id)
    v_peak = rate * emb.L
    goal = emb.to_world(np.zeros(dim))
    q_mid = 0.5 * (robot.q_min + robot.q_max) if q_null is None else np.asarray(q_null, float)
    q_rng = robot.q_max - robot.q_min
    lo = np.maximum(robot.q_min, sv.ctrl_lo)
    hi = np.minimum(robot.q_max, sv.ctrl_hi)
    tau_doc = (
        TAU_MAX_DOC.get(robot.spec.name, (np.full(n, np.inf), "none"))[0]
        if tau_limit is None
        else np.asarray(tau_limit, float)
    )
    d = mujoco.MjData(m)
    q0 = np.asarray(q_start, float)
    d.qpos[robot.qadr] = q0
    d.qvel[:] = 0.0
    q_des = q0.copy()
    if gravity_ff:
        q_des = q_des + gravity_torque(robot, q0) / sv.kp
    q_des = np.clip(q_des, lo, hi)
    qdes0 = q_des.copy()
    d.ctrl[sv.act] = q_des
    mujoco.mj_forward(m, d)
    P = np.empty((N + 1, 3))
    Xn = np.empty((N + 1, dim))
    Q = np.empty((N + 1, n))
    QD = np.empty((N + 1, n))
    QDES = np.empty((N + 1, n))
    QD_CMD = np.zeros((N, n))
    TAU_CMD = np.zeros((N, n))
    TAU_APP = np.zeros((N, n))
    ori = np.zeros(N + 1)
    dperp = np.zeros(N + 1)
    smin = np.full(N, np.inf)
    vscale = np.zeros(N)
    ncon = np.zeros(N, int)
    n_vel_scale = n_qdes_clip = n_damped = 0
    qdes_clip_joints = set()
    first_contact = None
    R0 = None

    def observe(k):
        nonlocal R0
        q = d.qpos[robot.qadr].copy()
        qd_meas = d.qvel[robot.dadr].copy()
        p, R = robot.fk(q)
        R0 = R if R0 is None else R0
        x_n, dp = emb.to_plane(p)
        P[k], Xn[k], Q[k], QD[k], QDES[k], dperp[k] = (p, x_n, q, qd_meas, q_des, dp)
        v_lin = v_peak * emb.vel_to_world(ds.f_n(x_n)) - sp.k_perp * dp * emb.normal
        if position_mode == "xy":
            v_lin = v_peak * emb.vel_to_world(ds.f_n(x_n))
        Jt, vt, oe = servo_task_rows(robot.jac(), R, tgt, v_lin, sp.k_rot, emb, position_mode)
        ori[k] = oe if tgt.mode != "pos" else float(np.linalg.norm(rot_error(R, R0)))
        return (q, qd_meas, Jt, vt)

    def command(k, state):
        nonlocal q_des, n_vel_scale, n_damped, n_qdes_clip
        q, qd_meas, Jt, vt = state
        qd_cmd, info = solve_qdot(Jt, vt, free, robot.qd_max, sp.ik, q, q_mid, q_rng)
        smin[k] = info["sigma_min"]
        vscale[k] = info["vel_scale"]
        n_vel_scale += info["vel_scale"] > 1.0
        n_damped += info["lam2"] > 0.0
        q_new = q_des + qd_cmd * sp.dt
        q_des = np.clip(q_new, lo, hi)
        hit = q_des != q_new
        if np.any(hit):
            n_qdes_clip += 1
            qdes_clip_joints.update((int(j) for j in np.flatnonzero(hit)))
        QD_CMD[k] = qd_cmd
        TAU_CMD[k] = sv.kp * (q_des - q) - sv.kv * qd_meas
        return Command("joint_position", q_des, sv.act)

    def after_step(k):
        nonlocal first_contact
        TAU_APP[k] = d.actuator_force[sv.act]
        if contact_check:
            ncon[k] = d.ncon
            if d.ncon and first_contact is None:
                c = d.contact[0]
                first_contact = {
                    "t_s": (k + 1) * sp.dt,
                    "geoms": [
                        m.body(int(m.geom_bodyid[c.geom1])).name,
                        m.body(int(m.geom_bodyid[c.geom2])).name,
                    ],
                }
        if not np.all(np.isfinite(d.qpos)):
            raise FloatingPointError(f"{robot.spec.name}: non-finite qpos at step {k}")

    run_intervals(
        m, d, N, observe=observe, command=command, after_step=after_step, n_substeps=n_substeps
    )
    t = np.arange(N + 1) * sp.dt
    L = emb.L
    thr = sp.success_frac * L
    dist_goal = np.linalg.norm(P - goal, axis=1)
    reached = np.flatnonzero(dist_goal < thr)
    gap = QDES - Q
    k_gap = int(np.argmax(np.max(np.abs(gap), axis=1)))
    pos_exc = np.maximum(robot.q_min - Q, Q - robot.q_max)
    vel_ratio = np.abs(QD) / robot.qd_max
    sat = sv.force_limited & ((TAU_CMD < sv.force_lo) | (TAU_CMD > sv.force_hi))
    warn = {
        mujoco.mjtWarning(i).name: int(d.warning[i].number)
        for i in range(len(d.warning))
        if d.warning[i].number
    }
    res = {
        "robot": robot.spec.name,
        "mode": tgt.mode,
        "execution": "dynamic",
        "control_interface": "MuJoCo joint position servo (ctrl = q_des), tau = kp (ctrl - q) - kv q_dot",
        "servo": sv.as_dict(),
        "n_substeps": n_substeps,
        "timestep": m.opt.timestep,
        "L_m": L,
        "v_peak_m_s": v_peak,
        "dt": sp.dt,
        "T_max": sp.T_max,
        "n_steps": N,
        "success": bool(dist_goal[-1] < thr),
        "final_dist_m": float(dist_goal[-1]),
        "t_reach_s": float(reached[0] * sp.dt) if len(reached) else None,
        "max_out_of_plane_m": float(np.max(np.abs(dperp))),
        "max_ori_err_rad": float(np.max(ori)),
        "min_sigma_min": float(np.min(smin)),
        "n_damped_steps": int(n_damped),
        "n_vel_scale_steps": int(n_vel_scale),
        "n_qdes_clip_steps": int(n_qdes_clip),
        "qdes_clip_joints": sorted(qdes_clip_joints),
        "n_force_sat_steps": int(np.sum(np.any(sat, axis=1))),
        "force_sat_steps_per_joint": np.sum(sat, axis=0).tolist(),
        "n_pos_limit_viol_steps": int(np.sum(np.any(pos_exc > 0, axis=1))),
        "max_pos_limit_exceed_rad": float(max(0.0, pos_exc.max())),
        "n_vel_limit_viol_steps": int(np.sum(np.any(vel_ratio > 1.0, axis=1))),
        "max_vel_ratio": float(vel_ratio.max()),
        "max_abs_qd_rad_s": np.abs(QD).max(0).tolist(),
        "max_abs_tau_cmd_Nm": np.abs(TAU_CMD).max(0).tolist(),
        "max_abs_tau_applied_Nm": np.abs(TAU_APP).max(0).tolist(),
        "tau_doc_Nm": tau_doc.tolist(),
        "n_tau_doc_exceed_steps": int(np.sum(np.any(np.abs(TAU_APP) > tau_doc, axis=1))),
        "peak_tau_ratio": float(np.max(np.abs(TAU_APP) / tau_doc)),
        "gravity_ff": bool(gravity_ff),
        "qdes0_minus_q0_rad": (qdes0 - q0).tolist(),
        "min_limit_margin_pct": float(min((robot.limit_margin_pct(qq).min() for qq in Q))),
        "q_final_rad": Q[-1].tolist(),
        "max_abs_gap_rad": float(np.abs(gap).max()),
        "max_abs_gap_per_joint_rad": np.abs(gap).max(0).tolist(),
        "t_max_gap_s": float(k_gap * sp.dt),
        "final_gap_rad": gap[-1].tolist(),
        "n_contact_steps": int(np.sum(ncon > 0)),
        "first_contact": first_contact,
        "mujoco_warnings": warn,
        "finite": bool(np.all(np.isfinite(Q)) and np.all(np.isfinite(QD))),
    }
    res["traj"] = {
        "t": t,
        "P": P,
        "P_ideal": P_id,
        "Xn": Xn,
        "X_ideal": X_id,
        "Q": Q,
        "QD": QD,
        "Q_des": QDES,
        "QD_cmd": QD_CMD,
        "tau_cmd": TAU_CMD,
        "tau_applied": TAU_APP,
        "vel_scale": vscale,
        "ncon": ncon,
        "tau_limit": tau_doc,
    }
    return res


def run(case):
    """Run the native position servo without extra gravity feedforward."""
    return _run_source(
        case.robot,
        case.ds,
        case.emb,
        case.x0_n,
        case.rate,
        case.tgt,
        case.sp,
        q_start=case.q_start,
        n_substeps=1,
        tau_limit=case.tau_limit,
        position_mode=case.position_mode,
    )
