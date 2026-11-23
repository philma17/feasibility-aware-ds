"""PIC force law, energy tank, and joint-torque execution callbacks.

[P] Kronander and Billard, "Passive Interaction Control With Dynamical Systems",
IEEE RA-L 1(1), 2016. [R] epfl-lasa/passive-ds-control reference implementation.
"""

from __future__ import annotations
from dataclasses import dataclass
import mujoco
import numpy as np
from sim.ds import check_dims, ideal_rollout
from sim.robot import rot_error, plane_component, gravity_torque, TAU_MAX_DOC
from sim.sim import Case, execute_intervals, first_contact_at
from evaluation.metrics import tracking_metrics
from . import Command
from sim.disturbance import TCPForceEnvironment


def smooth_step(x, a: float, b: float, kind: str = "quintic"):
    """H_{a,b}(x): 0 for x < a, 1 for x > b, smooth in between. [P] Eq. 25a (quintic).
    kind="cosine" is the step used by [R] (smooth_rise in smooth_truncation.h)."""
    if not b > a:
        raise ValueError(f"smooth step needs b > a, got a={a}, b={b}")
    u = np.clip((np.asarray(x, float) - a) / (b - a), 0.0, 1.0)
    if kind == "quintic":
        return 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3
    if kind == "cosine":
        return 0.5 - 0.5 * np.cos(np.pi * u)
    raise ValueError(f"unknown smooth step '{kind}'")


def smooth_step_inv(x, a: float, b: float, kind: str = "quintic"):
    """H^-_{a,b}(x) = 1 - H_{a,b}(x), [P] Eq. 25b."""
    return 1.0 - smooth_step(x, a, b, kind)


@dataclass(frozen=True)
class Coupling:
    """alpha, beta_s, beta_R of [P] Eq. 26-28 for tank bounds [0, s_max]."""

    s_max: float
    delta_s: float
    delta_z: float
    kind: str = "quintic"

    def __post_init__(self):
        if not (self.s_max > 0 and self.delta_z > 0 and (self.delta_s > 0)):
            raise ValueError("s_max, delta_s, delta_z must be > 0")
        if self.delta_s > 0.5 * self.s_max:
            raise ValueError("delta_s must be <= s_max / 2")

    def alpha(self, s):
        """[P] Eq. 26: alpha(s) = H^-_{s_max - delta_s, s_max}(s)."""
        return smooth_step_inv(s, self.s_max - self.delta_s, self.s_max, self.kind)

    def beta_s(self, z, s):
        """[P] Eq. 27 (lower tank bound = 0)."""
        k = self.kind
        t1 = smooth_step(z, -self.delta_z, 0.0, k) * smooth_step_inv(s, 0.0, self.delta_s, k)
        t2 = smooth_step_inv(z, 0.0, self.delta_z, k) * smooth_step(
            s, self.s_max - self.delta_s, self.s_max, k
        )
        return 1.0 - t1 - t2

    def beta_R(self, z, s):
        """[P] Eq. 28 (lower tank bound = 0)."""
        k = self.kind
        a = smooth_step(z, -self.delta_z, 0.0, k)
        A = smooth_step_inv(s, 0.0, self.delta_s, k)
        b = smooth_step_inv(z, 0.0, self.delta_z, k)
        B = smooth_step(s, self.s_max - self.delta_s, self.s_max, k)
        return (1.0 - a * A) * (1.0 - a * b * B)


def _gram_schmidt(Q: np.ndarray) -> np.ndarray:
    """Orthonormalize the columns of Q in order (column 0 is kept in direction). A column that
    degenerates (parallel to earlier ones) is replaced by the standard basis vector that is
    most orthogonal to the earlier columns: deterministic, no random numbers."""
    n = Q.shape[1]
    out = np.zeros_like(Q)
    for i in range(n):
        v = Q[:, i].copy()
        for j in range(i):
            v -= out[:, j] @ v * out[:, j]
        nv = np.linalg.norm(v)
        if nv < 1e-08:
            best = None
            for c in np.eye(Q.shape[0]):
                w = c - sum((out[:, j] @ c * out[:, j] for j in range(i)))
                if best is None or np.linalg.norm(w) > np.linalg.norm(best):
                    best = w
            v, nv = (best, np.linalg.norm(best))
        out[:, i] = v / nv
    return out


class DampingBasis:
    """Orthonormal Q(x) with first column e1 = f / |f|. If |f| < eta the previous basis is
    kept ([P] Sec. III-A, [R] MINSPEED)."""

    def __init__(self, dim: int = 3, eta: float = 1e-06):
        self.Q = np.eye(dim)
        self.eta = float(eta)
        self.n_kept = 0

    def update(self, f: np.ndarray) -> np.ndarray:
        nf = float(np.linalg.norm(f))
        if nf >= self.eta:
            Q = self.Q.copy()
            Q[:, 0] = f / nf
            self.Q = _gram_schmidt(Q)
        else:
            self.n_kept += 1
        return self.Q


def damping_matrix(Q: np.ndarray, lam) -> np.ndarray:
    """D = Q diag(lam) Q^T (symmetric positive semi-definite for lam >= 0)."""
    return Q * np.asarray(lam, float) @ Q.T


@dataclass
class PICParams:
    lam: tuple = (20.0, 200.0, 200.0)
    s_max: float = 10.0
    s0: float = 10.0
    delta_s_frac: float = 0.1
    delta_z: float = 0.01
    eta: float = 1e-06
    step: str = "quintic"
    decomposition: str = "none"

    def validate(self) -> None:
        if len(self.lam) != 3 or min(self.lam) < 0:
            raise ValueError("lambda must be 3 non-negative values")
        if not 0.0 < self.s0 <= self.s_max:
            raise ValueError(f"need 0 < s0 <= s_max, got s0={self.s0}, s_max={self.s_max}")
        if self.decomposition != "none":
            raise NotImplementedError(
                f"decomposition '{self.decomposition}' is not implemented in this step; only 'none' (f_C = 0, f_R = f)"
            )

    @classmethod
    def from_cfg(cls, c: dict) -> "PICParams":
        p = cls(
            lam=tuple((float(x) for x in c["lambda"])),
            s_max=float(c["s_max"]),
            s0=float(c["s0"]),
            delta_s_frac=float(c["delta_s_frac"]),
            delta_z=float(c["delta_z"]),
            eta=float(c["eta"]),
            step=str(c["step"]),
            decomposition=str(c["decomposition"]),
        )
        p.validate()
        return p


class PICController:
    """PIC-DS force law and energy tank in task space. State: tank level s and damping basis."""

    def __init__(self, p: PICParams, dim: int = 3):
        p.validate()
        self.p = p
        self.cp = Coupling(p.s_max, p.delta_s_frac * p.s_max, p.delta_z, p.step)
        self.basis = DampingBasis(dim, p.eta)
        self.s = float(p.s0)
        self.n_clamp = 0
        self.max_excursion = 0.0

    def step(self, xd: np.ndarray, f: np.ndarray, dt: float) -> dict:
        """One control step at the measured TCP velocity xd and DS value f(x) (both world, 3D).

        decomposition "none": f_C = 0, f_R = f. Returns the task force F and diagnostics; the
        tank is advanced to the next step after F has been computed from the CURRENT s."""
        p, cp = (self.p, self.cp)
        Q = self.basis.update(f)
        D = damping_matrix(Q, p.lam)
        lam1 = p.lam[0]
        f_C = np.zeros_like(f)
        f_R = f
        z = float(xd @ f_R)
        s = self.s
        beta_R = float(cp.beta_R(z, s))
        beta_s = float(cp.beta_s(z, s))
        alpha = float(cp.alpha(s))
        F = -D @ xd + lam1 * (f_C + beta_R * f_R)
        p_diss = float(xd @ D @ xd)
        sdot = alpha * p_diss - beta_s * lam1 * z
        s_new = s + sdot * dt
        exc = max(s_new - p.s_max, -s_new, 0.0)
        if exc > 0.0:
            # Record Euler overshoot before clamping; the clamp is not a passivity proof.
            self.n_clamp += 1
            self.max_excursion = max(self.max_excursion, exc)
            s_new = min(max(s_new, 0.0), p.s_max)
        self.s = s_new
        return {
            "F": F,
            "D": D,
            "s": s,
            "s_next": s_new,
            "z": z,
            "alpha": alpha,
            "beta_s": beta_s,
            "beta_R": beta_R,
            "sdot": sdot,
            "p_diss": p_diss,
            "e1": Q[:, 0].copy(),
        }


def null_projector(Jp: np.ndarray, rtol: float = 1e-06):
    """Orthogonal projector onto null(Jp) and its dimension. Dimension 0 -> zero matrix (the
    null-space term is blocked, as required for an empty null space)."""
    _, S, Vt = np.linalg.svd(Jp)
    rank = int(np.sum(S > rtol * S[0]))
    n = Jp.shape[1]
    if rank >= n:
        return (np.zeros((n, n)), 0)
    V0 = Vt[rank:].T
    return (V0 @ V0.T, n - rank)


def _run_source(
    case: Case,
    params: PICParams,
    *,
    null_d_q: float = 0.0,
    clip: bool = True,
    pulses=(),
    use_f: bool = True,
    gravity_comp: str = "nominal",
    joint_friction_feedforward=None,
    compute_tracking: bool = True,
) -> dict:
    """Closed-loop PIC-DS run. use_f=False sets f = 0 (damping + gravity compensation only).
    pulses: iterable of dicts {t_start_s, duration_s, force_N[3]} (world frame, at the TCP).
    gravity_comp: "nominal" (g(q) of the unmodified model, the default: the controller does not know
    about plant changes) or "plant" (g(q) of the simulated, possibly modified model; separates the effect
    of a changed plant from the effect of the compensation mismatch).
    Orientation (case.orient): none -> tau = g + Jp^T F + tau_null (bit-identical to the historical code);
    damped/hold -> tau = g + J6^T [F; tau_rot] + tau_null with the null space of J6 (FR3: 1-dim, UR5e: 0).
    """
    robot, ds, emb, sp = (case.robot, case.ds, case.emb, case.sp)
    m = robot.model
    n = robot.n
    if m.nv != n:
        raise NotImplementedError("model has DoF outside the arm")
    dt = sp.dt
    if not np.isclose(dt, m.opt.timestep):
        raise ValueError(f"controller dt {dt} != model timestep {m.opt.timestep}")
    T = sp.T_max
    N = int(round(T / dt))
    x0_n = case.x0_n
    dim = check_dims(ds, emb, x0_n)
    X_id = ideal_rollout(ds, x0_n, dt, N, case.rate)
    P_id = emb.to_world(X_id)
    v_peak = case.rate * emb.L
    goal = emb.to_world(np.zeros(dim))
    if case.tau_limit is not None:
        tau_lim = np.asarray(case.tau_limit, float)
    else:
        tau_lim = TAU_MAX_DOC[robot.spec.name][0]
    if gravity_comp not in ("nominal", "plant"):
        raise ValueError("gravity_comp must be 'nominal' or 'plant'")
    g_robot = (
        case.robot_nominal
        if gravity_comp == "nominal" and case.robot_nominal is not None
        else robot
    )
    gravity_data = mujoco.MjData(g_robot.model)
    orient = case.orient
    pulses = list(pulses)
    environment = TCPForceEnvironment(robot, pulses) if pulses else None
    q0 = np.asarray(case.q_start, float)
    ctrl = PICController(params)
    if joint_friction_feedforward and joint_friction_feedforward.get("enabled", False):
        raise NotImplementedError(
            "Experimental friction feedforward is not included in this checkout"
        )
    d = mujoco.MjData(m)
    d.qpos[robot.qadr] = q0
    d.qvel[:] = 0.0
    keys = (
        "P",
        "Xn",
        "Q",
        "QD",
        "XD",
        "F",
        "F_ext",
        "TAU_CMD",
        "TAU_APP",
        "TAU_G",
        "QACC",
        "OMEGA",
        "TAU_ROT",
    )
    shapes = {
        "P": 3,
        "Xn": dim,
        "Q": n,
        "QD": n,
        "XD": 3,
        "F": 3,
        "F_ext": 3,
        "TAU_CMD": n,
        "TAU_APP": n,
        "TAU_G": n,
        "QACC": n,
        "OMEGA": 3,
        "TAU_ROT": 3,
    }
    L = {k: np.zeros((N + 1, shapes[k])) for k in keys}
    sc_keys = (
        "s",
        "z",
        "alpha",
        "beta_s",
        "beta_R",
        "p_diss",
        "KE",
        "P_ext",
        "dperp",
        "f_norm",
        "null_dim",
        "E_rot",
        "ori_err",
        "sigma_min_trans",
    )
    S = {k: np.zeros(N + 1) for k in sc_keys}
    old_flags = int(m.opt.disableflags)
    # Disable MJCF position servos so they cannot add torque to PIC's direct input.
    m.opt.disableflags = old_flags | int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    tmp = np.zeros(m.nv)
    first_contact = None
    R_ref = None

    def observe(k):
        nonlocal R_ref
        q = d.qpos[robot.qadr].copy()
        qd = d.qvel[robot.dadr].copy()
        p, R = robot.fk(q)
        J6 = robot.jac()
        Jp = J6[:3]
        S["sigma_min_trans"][k] = np.linalg.svd(Jp, compute_uv=False)[-1]
        xd = Jp @ qd
        omega = J6[3:] @ qd
        if R_ref is None:
            R_ref = orient.R_des if orient is not None and orient.mode == "hold" else R.copy()
        e_rot = rot_error(R, R_ref)
        S["ori_err"][k] = float(np.linalg.norm(e_rot))
        L["OMEGA"][k] = omega
        if orient is not None and orient.mode == "hold":
            S["E_rot"][k] = 0.5 * orient.K_r * float(e_rot @ e_rot)
        x_n, dp = emb.to_plane(p)
        # Feedback evaluates the DS at the measured TCP, not along the ideal rollout.
        if use_f:
            f = v_peak * emb.vel_to_world(ds.f_n(x_n))
            if case.position_mode == "xy":
                f = plane_component(f, emb)
            else:
                f = f - sp.k_perp * dp * emb.normal
        else:
            f = np.zeros(3)
        mujoco.mj_mulM(m, d, tmp, d.qvel)
        F_ext = environment.force_at_step(k) if environment is not None and k < N else np.zeros(3)
        L["P"][k], L["Xn"][k], L["Q"][k], L["QD"][k], L["XD"][k] = (p, x_n, q, qd, xd)
        L["F_ext"][k] = F_ext
        S["KE"][k] = 0.5 * float(d.qvel @ tmp)
        S["P_ext"][k] = float(F_ext @ xd)
        S["dperp"][k] = dp
        S["f_norm"][k] = float(np.linalg.norm(f))
        if k == N:
            S["s"][k] = ctrl.s
        return (q, qd, J6, Jp, xd, omega, e_rot, f)

    def command(k, state):
        q, qd, J6, Jp, xd, omega, e_rot, f = state
        out = ctrl.step(plane_component(xd, emb) if case.position_mode == "xy" else xd, f, dt)
        if case.position_mode == "xy":
            out["F"] = plane_component(out["F"], emb)
        g = gravity_torque(g_robot, q, data=gravity_data)
        if orient is None:
            Nproj, nd = null_projector(Jp)
            tau_task = Jp.T @ out["F"]
        else:
            tau_rot = -orient.D_r @ omega
            if orient.mode == "hold":
                tau_rot = tau_rot + orient.K_r * e_rot
            L["TAU_ROT"][k] = tau_rot
            Nproj, nd = null_projector(J6)
            tau_task = J6.T @ np.concatenate([out["F"], tau_rot])
        tau_null = -null_d_q * (Nproj @ qd) if null_d_q > 0 and nd > 0 else np.zeros(n)
        tau_cmd = g + tau_task + tau_null
        tau_app = np.clip(tau_cmd, -tau_lim, tau_lim) if clip else tau_cmd.copy()
        L["F"][k], L["TAU_CMD"][k], L["TAU_APP"][k], L["TAU_G"][k] = (out["F"], tau_cmd, tau_app, g)
        for key in ("s", "z", "alpha", "beta_s", "beta_R", "p_diss"):
            S[key][k] = out[key]
        S["null_dim"][k] = nd
        return Command("joint_torque", tau_app, robot.dadr)

    def after_step(k):
        nonlocal first_contact
        L["QACC"][k] = d.qacc[robot.dadr]
        if d.ncon and first_contact is None:
            first_contact = first_contact_at(m, d, (k + 1) * dt)
        if not np.all(np.isfinite(d.qpos)):
            raise FloatingPointError(f"non-finite qpos at step {k}")

    try:
        execute_intervals(
            m,
            d,
            N,
            observe=observe,
            command=command,
            after_step=after_step,
            forward_before_observe=True,
            environment=environment,
        )
    finally:
        m.opt.disableflags = old_flags
        d.qfrc_applied[:] = 0.0
    P, Q, QD = (L["P"], L["Q"], L["QD"])
    if compute_tracking:
        mt, e_sync, g_dev = tracking_metrics(P, P_id, goal, emb.L, dt, sp.success_frac)
    else:
        # Optional postprocessing only; the physics and control schedule are identical.
        mt = {"status": "not_requested"}
        e_sync = np.linalg.norm(P - P_id, axis=1)
        g_dev = None
    dist = np.linalg.norm(P - goal, axis=1)
    reached = np.flatnonzero(dist < sp.success_frac * emb.L)
    # Terminal command rows are padding; only N intervals actually apply torque.
    TC, TA = (L["TAU_CMD"][:N], L["TAU_APP"][:N])
    clipped = np.abs(TC) > tau_lim + 1e-12
    pos_exc = np.maximum(robot.q_min - Q, Q - robot.q_max)
    vel_ratio = np.abs(QD) / robot.qd_max
    W = S["KE"] + S["s"] + S["E_rot"]
    dW = W[1:] - W[:-1] - S["P_ext"][:-1] * dt
    res = {
        "robot": robot.spec.name,
        "joint_names": list(robot.spec.joints),
        "controller": "PIC-DS",
        "mode": "pos",
        "n_steps": N,
        "dt": dt,
        "T_max": T,
        "L_m": emb.L,
        "v_peak_m_s": v_peak,
        "use_f": bool(use_f),
        "interface": "mjDSBL_ACTUATION + data.qfrc_applied (joint torque)",
        "params": {
            "lambda": list(params.lam),
            "s_max": params.s_max,
            "s0": params.s0,
            "delta_s": params.delta_s_frac * params.s_max,
            "delta_z": params.delta_z,
            "eta": params.eta,
            "step": params.step,
            "decomposition": params.decomposition,
            "null_d_q": null_d_q,
            "clip": bool(clip),
        },
        "success": bool(dist[-1] < sp.success_frac * emb.L),
        "final_dist_m": float(dist[-1]),
        "success_threshold_m": float(sp.success_frac * emb.L),
        "t_reach_s": float(reached[0] * dt) if len(reached) else None,
        "max_dist_after_pulse_m": None,
        "tracking": mt,
        "max_out_of_plane_m": float(np.max(np.abs(S["dperp"]))),
        "tank": {
            "s_min": float(S["s"].min()),
            "s_max_reached": float(S["s"].max()),
            "s_final": float(S["s"][-1]),
            "n_clamp_steps": int(ctrl.n_clamp),
            "max_excursion_J": float(ctrl.max_excursion),
            "min_beta_R": float(S["beta_R"][:N].min()),
            "n_beta_R_below_1_steps": int(np.sum(S["beta_R"][:N] < 1.0 - 1e-12)),
            "n_basis_kept_steps": int(ctrl.basis.n_kept),
        },
        "torque": {
            "limit_Nm": tau_lim.tolist(),
            "clip_enabled": bool(clip),
            "max_abs_cmd_Nm": np.abs(TC).max(0).tolist(),
            "max_abs_applied_Nm": np.abs(TA).max(0).tolist(),
            "peak_cmd_ratio": float(np.max(np.abs(TC) / tau_lim)),
            "peak_applied_ratio": float(np.max(np.abs(TA) / tau_lim)),
            "n_clip_steps": int(np.sum(np.any(clipped, axis=1))),
            "n_clip_per_joint": np.sum(clipped, axis=0).tolist(),
        },
        "n_pos_limit_viol_steps": int(np.sum(np.any(pos_exc > 0, axis=1))),
        "max_pos_limit_exceed_rad": float(max(0.0, pos_exc.max())),
        "n_vel_limit_viol_steps": int(np.sum(np.any(vel_ratio > 1.0, axis=1))),
        "max_vel_ratio": float(vel_ratio.max()),
        "min_limit_margin_pct": float(min((robot.limit_margin_pct(qq).min() for qq in Q))),
        "null_dim_min_max": [int(S["null_dim"][:N].min()), int(S["null_dim"][:N].max())],
        "first_contact": first_contact,
        "energy": {
            "KE_max_J": float(S["KE"].max()),
            "KE_final_J": float(S["KE"][-1]),
            "W_dW_max_J": float(dW.max()),
            "n_dW_positive_steps": int(np.sum(dW > 1e-09)),
            "sum_positive_dW_J": float(dW[dW > 0].sum()),
        },
        "q_start_rad": q0.tolist(),
        "q_final_rad": Q[-1].tolist(),
        "orientation": {
            "mode": "none" if orient is None else orient.mode,
            "max_ori_err_rad": float(S["ori_err"].max()),
            "ori_err_final_rad": float(S["ori_err"][-1]),
            "reference": (
                "R_des" if orient is not None and orient.mode == "hold" else "R(t=0) (drift)"
            ),
            "gravity_comp": gravity_comp,
        },
        "max_ori_err_rad": float(S["ori_err"].max()),
        "finite": bool(np.all(np.isfinite(Q)) and np.all(np.isfinite(QD))),
    }
    if orient is not None:
        res["orientation"].update(
            K_r=orient.K_r,
            zeta=orient.zeta,
            D_r=orient.D_r.tolist(),
            Lambda_r_start=orient.Lambda_r.tolist(),
            R_des=orient.R_des.tolist(),
        )
    clearance = None
    if case.unsafe is not None:
        clearance = emb.L * case.unsafe.signed_distance(L["Xn"])
        res["clearance_min_m"] = float(clearance.min())
    res["traj"] = {
        "t": np.arange(N + 1) * dt,
        "P": P,
        "P_ideal": P_id,
        "Xn": L["Xn"],
        "X_ideal": X_id,
        "Q": Q,
        "QD": QD,
        "xd": L["XD"],
        "F": L["F"],
        "F_ext": L["F_ext"],
        "tau_cmd": L["TAU_CMD"],
        "tau_applied": L["TAU_APP"],
        "tau_grav": L["TAU_G"],
        "tau_limit": tau_lim,
        "err_sync": e_sync,
        "dist_goal": dist,
        "dW": dW,
        "qacc": L["QACC"],
        "omega": L["OMEGA"],
        "tau_rot": L["TAU_ROT"],
        **S,
    }
    if g_dev is not None:
        res["traj"]["geom_dev"] = g_dev
    if environment is not None:
        res["traj"].update(environment.trajectory(m, d))
    if clearance is not None:
        res["traj"]["clearance_m"] = clearance
    return res


def validate_config(cfg):
    pc = cfg["pic"]
    PICParams.from_cfg(pc)
    for value, label in (
        (pc["use_f"], "pic.use_f"),
        (pc["torque_limit"]["clip"], "pic.torque_limit.clip"),
        (pc["null_space"]["enabled"], "pic.null_space.enabled"),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{label} must be a boolean")


def run(case, *, compute_tracking: bool = True):
    """Configure the source PIC path from one resolved experiment."""
    pc = case.cfg["pic"]
    null = pc["null_space"]
    pulses = case.cfg["disturbance"]["pulses"] if case.cfg["disturbance"]["enabled"] else []
    return _run_source(
        case,
        PICParams.from_cfg(pc),
        null_d_q=float(null["d_q"]) if null["enabled"] else 0.0,
        clip=pc["torque_limit"]["clip"],
        pulses=pulses,
        use_f=pc["use_f"],
        gravity_comp=pc["gravity_comp"],
        joint_friction_feedforward=pc["joint_friction_feedforward"],
        compute_tracking=compute_tracking,
    )
