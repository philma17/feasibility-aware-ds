"""Registered arm models, kinematics, physical parameters, actuator mapping, and IK."""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

MENAGERIE_DIR = str(Path(__file__).resolve().parents[1] / "vendor")
TASK_DIMS = {"pos": 3, "axis5": 5, "full6": 6}
TAU_MAX_DOC = {
    "FR3": (
        np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]),
        "franka_description robots/fr3/joint_limits.yaml effort; enforced in memory",
    )
}
import os


@dataclass(frozen=True)
class RobotSpec:
    name: str
    mjcf: str
    joints: tuple
    site: str
    q_min_doc: tuple
    q_max_doc: tuple
    qd_max: tuple
    sources: dict = field(default_factory=dict)
    caveats: tuple = ()
    family: str = ""
    fixed_joints: tuple = ()  # (original joint name, fixed angle [rad]); absent from q.
    acceleration_reference_rad_s2: float | None = None


FR3 = RobotSpec(
    "FR3",
    "franka_fr3/fr3.xml",
    tuple((f"fr3_joint{i}" for i in range(1, 8))),
    "attachment_site",
    (-2.9007, -1.8361, -2.9007, -3.077, -2.8763, 0.4398, -3.0508),
    (2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508),
    (2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26),
    sources={"limits": "franka_description robots/fr3/joint_limits.yaml"},
    caveats=("Constant velocity limits omit the source position-dependent limits.",),
    family="FR3",
    acceleration_reference_rad_s2=10.0,
)


def _fr3_variant(name, filename, locked):
    """Keep physical references in original joint order after removing fixed joints."""
    indices = [i for i, joint in enumerate(FR3.joints) if joint not in locked]
    TAU_MAX_DOC[name] = (TAU_MAX_DOC["FR3"][0][indices], TAU_MAX_DOC["FR3"][1])
    return replace(
        FR3, name=name, mjcf=f"franka_fr3/{filename}",
        joints=tuple(FR3.joints[i] for i in indices),
        q_min_doc=tuple(FR3.q_min_doc[i] for i in indices),
        q_max_doc=tuple(FR3.q_max_doc[i] for i in indices),
        qd_max=tuple(FR3.qd_max[i] for i in indices),
        fixed_joints=tuple((joint, 0.0) for joint in locked),
    )


FR3_FIXED_J5 = _fr3_variant("FR3_fixed_J5", "fr3_fixed_j5.xml", ("fr3_joint5",))
FR3_FIXED_J3 = _fr3_variant("FR3_fixed_J3", "fr3_fixed_j3.xml", ("fr3_joint3",))
FR3_FIXED_J3_J5 = _fr3_variant(
    "FR3_fixed_J3_J5", "fr3_fixed_j3_j5.xml", ("fr3_joint3", "fr3_joint5")
)
UR5E = RobotSpec(
    "UR5e", "universal_robots_ur5e/ur5e.xml",
    ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
     "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"),
    "attachment_site",
    (-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi),
    (2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi),
    (np.pi,) * 6,
    sources={
        "limits": "UniversalRobots/Universal_Robots_ROS2_Description/config/ur5e/joint_limits.yaml",
        "effort": "Menagerie ur5e.xml actuator forcerange: 150/150/150/28/28/28 N m",
    },
    caveats=("MJCF position limits are rounded; elbow limited to approximately +/- pi.",
             "No acceleration reference configured; native MJCF has no joint frictionloss."),
    family="UR5e",
)
TAU_MAX_DOC["UR5e"] = (np.array([150., 150., 150., 28., 28., 28.]), UR5E.sources["effort"])
ROBOTS = {spec.name: spec for spec in (FR3, FR3_FIXED_J5, FR3_FIXED_J3, FR3_FIXED_J3_J5, UR5E)}


def _quat_z_to(a: np.ndarray) -> np.ndarray:
    """Quaternion rotating the z axis onto the given unit direction."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    c = np.cross(z, a)
    s, d = (np.linalg.norm(c), float(z @ a))
    if s < 1e-12:
        return np.array([1.0, 0, 0, 0]) if d > 0 else np.array([0.0, 1.0, 0, 0])
    ang = np.arctan2(s, d)
    ax = c / s
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * ax])


class Robot:
    """MuJoCo FK and world-frame Jacobians evaluated on a separate kinematic data object."""

    def __init__(self, spec: RobotSpec, tcp_offset=None, tool_radius: float = 0.0, obstacles=None):
        self.spec = spec
        path = os.path.join(MENAGERIE_DIR, spec.mjcf)
        if tool_radius > 0 or obstacles:
            self.model = self._build(path, spec, tcp_offset, tool_radius, obstacles or [])
        else:
            self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.p_off = None if tcp_offset is None else np.asarray(tcp_offset[0], float)
        self.R_off = None if tcp_offset is None else np.asarray(tcp_offset[1], float)
        self._p_tcp = np.zeros(3)
        jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in spec.joints]
        if min(jids) < 0:
            raise ValueError(f"{spec.name}: joint not found in MJCF")
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([self.model.jnt_dofadr[j] for j in jids])
        self.n = len(jids)
        rng = np.array([self.model.jnt_range[j] for j in jids])
        self.q_min, self.q_max = (rng[:, 0].copy(), rng[:, 1].copy())
        self.qd_max = np.asarray(spec.qd_max, float)
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, spec.site)
        if self.site < 0:
            raise ValueError(f"{spec.name}: Site {spec.site} is missing")
        self.site_body = int(self.model.site_bodyid[self.site])
        g = self.model
        self.obstacle_geoms = [
            i for i in range(g.ngeom) if (g.geom(i).name or "").startswith("obstacle")
        ]
        self.obstacle_tags = {}
        for i in self.obstacle_geoms:
            tag = g.geom(i).name.split("|")[1] if "|" in g.geom(i).name else ""
            self.obstacle_tags.setdefault(tag, []).append(i)
        self.tool_geoms = [i for i in range(g.ngeom) if g.geom(i).name == "tool"]
        self.link_geoms = [
            i
            for i in range(g.ngeom)
            if g.geom_group[i] == 3 and g.geom_bodyid[i] != 0 and (i not in self.tool_geoms)
        ]
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.q_home = self.model.key_qpos[key][self.qadr].copy() if key >= 0 else np.zeros(self.n)
        self._jp = np.zeros((3, self.model.nv))
        self._jr = np.zeros((3, self.model.nv))

    def set_q(self, q: np.ndarray) -> None:
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    @staticmethod
    def _build(path, spec, tcp_offset, tool_radius, obstacles):
        """Build the existing optional tool and obstacle geometry in memory."""
        base = mujoco.MjModel.from_xml_path(path)
        sid = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_SITE, spec.site)
        bname = base.body(int(base.site_bodyid[sid])).name
        s_pos = base.site_pos[sid].copy()
        s_R = np.zeros(9)
        mujoco.mju_quat2Mat(s_R, base.site_quat[sid])
        s_R = s_R.reshape(3, 3)
        ms = mujoco.MjSpec.from_file(path)
        if tool_radius > 0:
            p_off = np.zeros(3) if tcp_offset is None else np.asarray(tcp_offset[0], float)
            a, b = (s_pos, s_pos + s_R @ p_off)
            ms.body(bname).add_geom(
                name="tool",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                pos=0.5 * (a + b),
                quat=_quat_z_to(b - a),
                size=[tool_radius, 0.5 * np.linalg.norm(b - a), 0.0],
                group=3,
                contype=0,
                conaffinity=0,
                rgba=[0.2, 0.2, 0.8, 1.0],
            )
        for i, ob in enumerate(obstacles):
            ax = np.asarray(ob["axis"], float) / np.linalg.norm(ob["axis"])
            c = np.asarray(ob["center"], float) + 0.5 * ob["height"] * ax
            tag = f"|{ob['tag']}" if ob.get("tag") else ""
            ms.worldbody.add_geom(
                name=f"obstacle{i}{tag}",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=c,
                quat=_quat_z_to(ax),
                size=[ob["radius"], 0.5 * ob["height"], 0.0],
                group=4,
                contype=0,
                conaffinity=0,
                rgba=[0.9, 0.1, 0.1, 0.6],
            )
        return ms.compile()

    def fk(self, q: np.ndarray):
        self.set_q(q)
        p = self.data.site_xpos[self.site].copy()
        R = self.data.site_xmat[self.site].reshape(3, 3).copy()
        if self.p_off is None:
            return (p, R)
        self._p_tcp = p + R @ self.p_off
        return (self._p_tcp.copy(), R @ self.R_off)

    def jac(self) -> np.ndarray:
        """World-frame geometric Jacobian [Jp; Jr] at the TCP after FK."""
        if self.p_off is None:
            mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self.site)
        else:
            mujoco.mj_jac(self.model, self.data, self._jp, self._jr, self._p_tcp, self.site_body)
        return np.vstack([self._jp[:, self.dadr], self._jr[:, self.dadr]])

    def clearance(self, geoms, distmax: float = 1.0, tag=None) -> np.ndarray:
        out = np.full(len(geoms), distmax)
        ft = np.zeros(6)
        obs = self.obstacle_geoms if tag is None else self.obstacle_tags[tag]
        for k, g in enumerate(geoms):
            for o in obs:
                out[k] = min(
                    out[k], mujoco.mj_geomDistance(self.model, self.data, g, o, distmax, ft)
                )
        return out

    def limit_margin_pct(self, q: np.ndarray) -> np.ndarray:
        rng = self.q_max - self.q_min
        return 100.0 * np.minimum(q - self.q_min, self.q_max - q) / rng


def svd_pinv(J: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    keep = S > rcond * max(S.max(initial=0.0), 1e-300)
    Sinv = np.where(keep, 1.0 / np.where(keep, S, 1.0), 0.0)
    return Vt.T * Sinv @ U.T


def nullspace_projector(J: np.ndarray) -> np.ndarray:
    return np.eye(J.shape[1]) - svd_pinv(J) @ J


def dls(J: np.ndarray, v: np.ndarray, eps: float, lam_max: float):
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    smin = float(S[-1])
    lam2 = 0.0 if smin >= eps else (1.0 - (smin / eps) ** 2) * lam_max**2
    denom = S**2 + lam2
    coef = np.where(denom > 0, S / np.where(denom > 0, denom, 1.0), 0.0)
    return (Vt.T @ (coef * (U.T @ v)), smin, lam2)


def manipulability(J: np.ndarray) -> float:
    S = np.linalg.svd(J, compute_uv=False)
    return float(np.prod(S))


def perp_basis(a: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a)
    h = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    b1 = np.cross(a, h)
    b1 /= np.linalg.norm(b1)
    return np.stack([b1, np.cross(a, b1)], axis=1)


def rot_error(R: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R_des @ R.T).as_rotvec()


def axis_error(R: np.ndarray, a_des: np.ndarray, axis_local=(0.0, 0.0, 1.0)) -> np.ndarray:
    a = R @ np.asarray(axis_local, float)
    c = np.cross(a, a_des)
    s = np.linalg.norm(c)
    th = np.arctan2(s, float(a @ a_des))
    if s < 1e-15:
        return np.zeros(3) if th < 1.0 else perp_basis(a)[:, 0] * th
    return c / s * th


@dataclass
class Target:
    mode: str = "pos"
    R_des: np.ndarray | None = None
    a_des: np.ndarray | None = None
    axis_local: tuple = (0.0, 0.0, 1.0)


def task_rows(J6: np.ndarray, R: np.ndarray, tgt: Target, v_lin: np.ndarray, k_rot: float):
    if tgt.mode == "pos":
        return (J6[:3], v_lin, 0.0)
    if tgt.mode == "full6":
        e = rot_error(R, tgt.R_des)
        return (J6, np.concatenate([v_lin, k_rot * e]), float(np.linalg.norm(e)))
    if tgt.mode == "axis5":
        e = axis_error(R, tgt.a_des, tgt.axis_local)
        B = perp_basis(R @ np.asarray(tgt.axis_local, float))
        Jt = np.vstack([J6[:3], B.T @ J6[3:]])
        return (Jt, np.concatenate([v_lin, k_rot * (B.T @ e)]), float(np.linalg.norm(e)))
    raise ValueError(tgt.mode)


@dataclass
class IKParams:
    eps: float = 0.02
    lam_max: float = 0.05
    k_null: float = 0.0


def solve_qdot(
    J_task: np.ndarray,
    v_task: np.ndarray,
    free: np.ndarray,
    qd_max: np.ndarray,
    p: IKParams,
    q: np.ndarray | None = None,
    q_mid: np.ndarray | None = None,
    q_rng: np.ndarray | None = None,
):
    n = J_task.shape[1]
    idx = np.flatnonzero(free)
    Jf = J_task[:, idx]
    qd_f, smin, lam2 = dls(Jf, v_task, p.eps, p.lam_max)
    if p.k_null > 0 and len(idx) > Jf.shape[0] and (q is not None):
        grad = -p.k_null * (q[idx] - q_mid[idx]) / (0.5 * q_rng[idx])
        qd_f = qd_f + nullspace_projector(Jf) @ grad
    qd = np.zeros(n)
    qd[idx] = qd_f
    scale = float(np.max(np.abs(qd) / qd_max))
    if scale > 1.0:
        qd = qd / scale
    return (
        qd,
        {
            "sigma_min": smin,
            "lam2": lam2,
            "vel_scale": scale,
            "manip": manipulability(Jf) if Jf.shape[0] <= Jf.shape[1] else 0.0,
        },
    )


@dataclass
class IKResult:
    q: np.ndarray
    pos_err: float
    ori_err: float
    converged: bool
    n_converged: int
    n_starts: int
    sigma_min: float = float("nan")
    margin_pct: float = float("nan")
    margin_ok: bool = True
    solutions: list = field(default_factory=list)


def ik_pose(
    robot,
    p_des,
    tgt: Target,
    q_init,
    free=None,
    iters: int = 400,
    tol: float = 1e-09,
    lam: float = 0.001,
    max_step: float = 0.2,
):
    q = np.clip(np.array(q_init, float), robot.q_min, robot.q_max)
    free = np.ones(robot.n, bool) if free is None else free
    idx = np.flatnonzero(free)
    for _ in range(iters):
        pc, R = robot.fk(q)
        J6 = robot.jac()
        e_p = p_des - pc
        Jt, err, ori = task_rows(J6, R, tgt, e_p, 1.0)
        if np.linalg.norm(e_p) < tol and ori < tol:
            break
        A = Jt[:, idx]
        dq = A.T @ np.linalg.solve(A @ A.T + lam**2 * np.eye(A.shape[0]), err)
        nrm = np.linalg.norm(dq)
        if nrm > max_step:
            dq *= max_step / nrm
        q[idx] = np.clip(q[idx] + dq, robot.q_min[idx], robot.q_max[idx])
    pc, R = robot.fk(q)
    _, _, ori = task_rows(robot.jac(), R, tgt, p_des - pc, 1.0)
    pe = float(np.linalg.norm(p_des - pc))
    return (q, pe, ori)


def task_sigma_min(robot, q, tgt: Target, free) -> float:
    _, R = robot.fk(q)
    Jt, _, _ = task_rows(robot.jac(), R, tgt, np.zeros(3), 1.0)
    return float(np.linalg.svd(Jt[:, free], compute_uv=False)[-1])


def multistart_ik(
    robot,
    p_des,
    tgt: Target,
    q_ref,
    n_starts: int,
    seed: int,
    criterion: str = "nearest_ref",
    free=None,
    tol_conv: float = 1e-06,
    min_margin_pct: float = 5.0,
) -> IKResult:
    rng = np.random.default_rng(seed)
    q_ref = np.asarray(q_ref, float)
    free = np.ones(robot.n, bool) if free is None else np.asarray(free, bool)
    starts = [q_ref] + list(rng.uniform(robot.q_min, robot.q_max, size=(n_starts - 1, robot.n)))
    starts = [np.where(free, s, q_ref) for s in starts]
    sols = []
    for s in starts:
        q, pe, oe = ik_pose(robot, p_des, tgt, s, free=free)
        sols.append((q, pe, oe))
    conv = [s for s in sols if s[1] < tol_conv and s[2] < tol_conv]
    margin_ok = True
    if conv:
        if criterion == "nearest_ref":
            best = min(conv, key=lambda s: np.linalg.norm(s[0] - q_ref))
        elif criterion == "max_margin":
            best = max(conv, key=lambda s: robot.limit_margin_pct(s[0]).min())
        elif criterion == "max_sigma_margin":
            ok = [s for s in conv if robot.limit_margin_pct(s[0]).min() > min_margin_pct]
            margin_ok = bool(ok)
            pool = ok if ok else conv
            best = max(pool, key=lambda s: task_sigma_min(robot, s[0], tgt, free))
        else:
            raise ValueError(criterion)
    else:
        best = min(sols, key=lambda s: s[1] + s[2])
        margin_ok = False
    res = IKResult(best[0], best[1], best[2], bool(conv), len(conv), n_starts)
    res.sigma_min = task_sigma_min(robot, best[0], tgt, free)
    res.margin_pct = float(robot.limit_margin_pct(best[0]).min())
    res.margin_ok = margin_ok
    res.solutions = [s[0].copy() for s in conv]
    return res


@dataclass
class ServoMap:
    """Position-servo actuators, indexed in Robot joint order (robot.spec.joints)."""

    act: np.ndarray
    kp: np.ndarray
    kv: np.ndarray
    ctrl_lo: np.ndarray
    ctrl_hi: np.ndarray
    force_limited: np.ndarray
    force_lo: np.ndarray
    force_hi: np.ndarray

    def as_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in self.__dataclass_fields__}


def position_servo_map(robot) -> ServoMap:
    """Read and verify the actuator model. Raises NotImplementedError for anything that is
    not a 1:1 joint-space PD position servo (then the controller design must be revisited)."""
    m = robot.model
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in robot.spec.joints]
    act = np.full(robot.n, -1)
    for a in range(m.nu):
        name = m.actuator(a).name
        if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            raise NotImplementedError(f"{name}: transmission is not a joint")
        j = int(m.actuator_trnid[a, 0])
        if j not in jids:
            raise NotImplementedError(f"{name}: drives joint {m.joint(j).name} outside the arm")
        i = jids.index(j)
        if act[i] >= 0:
            raise NotImplementedError(f"joint {robot.spec.joints[i]} has several actuators")
        act[i] = a
        kp, b = (m.actuator_gainprm[a, 0], m.actuator_biasprm[a, :3])
        ok = (
            m.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED
            and m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
            and (m.actuator_dyntype[a] == mujoco.mjtDyn.mjDYN_NONE)
            and (m.actuator_gear[a, 0] == 1.0)
            and (b[0] == 0.0)
            and (b[1] == -kp)
            and (b[2] <= 0.0)
            and (kp > 0)
        )
        if not ok:
            raise NotImplementedError(f"{name}: not a PD position servo (gain/bias/gear differ)")
    if np.any(act < 0) or m.nu != robot.n:
        raise NotImplementedError("actuators do not cover every arm joint exactly once")
    return ServoMap(
        act=act,
        kp=m.actuator_gainprm[act, 0].copy(),
        kv=-m.actuator_biasprm[act, 2].copy(),
        ctrl_lo=np.where(m.actuator_ctrllimited[act], m.actuator_ctrlrange[act, 0], -np.inf),
        ctrl_hi=np.where(m.actuator_ctrllimited[act], m.actuator_ctrlrange[act, 1], np.inf),
        force_limited=m.actuator_forcelimited[act].astype(bool),
        force_lo=m.actuator_forcerange[act, 0].copy(),
        force_hi=m.actuator_forcerange[act, 1].copy(),
    )


def gravity_torque(robot, q) -> np.ndarray:
    """Generalized gravity force g(q) (qfrc_bias at q_dot = 0) in robot joint order."""
    d = mujoco.MjData(robot.model)
    d.qpos[robot.qadr] = q
    mujoco.mj_forward(robot.model, d)
    return d.qfrc_bias[robot.dadr].copy()


PLANT_DEFAULTS = {
    "friction_enabled": True,
    "dof_frictionloss_scale": 1.0,
    "dof_damping_scale": 1.0,
    "gravity_scale": 1.0,
    "mass_scale": 1.0,
    "inertia_scale": 1.0,
    "torque_limit_scale": 1.0,
    "servo_enforce_torque_limit": True,
    "synthetic_friction": None,
    "com_shift": None,
}
FR3_REF = ("FR3", np.array([1.137, 1.137, 1.137, 1.137, 0.763, 0.44, 0.248]))


def resolve_plant(plant: dict | None) -> dict:
    """Defaults + unknown-key check (a typo in a YAML file must not be silently ignored)."""
    plant = dict(plant or {})
    unknown = sorted(set(plant) - set(PLANT_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown plant keys {unknown}; allowed: {sorted(PLANT_DEFAULTS)}")
    out = {**PLANT_DEFAULTS, **plant}
    for k, v in out.items():
        if k in ("servo_enforce_torque_limit", "friction_enabled"):
            if not isinstance(v, bool):
                raise ValueError(f"plant.{k} must be true or false, got {v!r}")
            out[k] = v
            continue
        if k == "synthetic_friction":
            if v not in (None, "none", "relative", "absolute"):
                raise ValueError("plant.synthetic_friction must be none | relative | absolute")
            out[k] = v
            continue
        if k == "com_shift":
            if v is not None:
                if set(v) - {"link", "dxyz_m"} or len(v.get("dxyz_m", [])) != 3:
                    raise ValueError("plant.com_shift needs {link: ..., dxyz_m: [dx, dy, dz]}")
                v = {"link": str(v.get("link", "last")), "dxyz_m": [float(x) for x in v["dxyz_m"]]}
            out[k] = v
            continue
        v = float(v)
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"plant.{k} must be finite and >= 0, got {v}")
        if k in ("mass_scale", "inertia_scale", "torque_limit_scale") and v <= 0:
            raise ValueError(f"plant.{k} must be > 0, got {v}")
        out[k] = v
    return out


def synthetic_frictionloss(robot_name: str, mode: str) -> np.ndarray:
    """Explicit FR3-only diagnostic reference, indexed by original joint names."""
    spec = ROBOTS[robot_name]
    if spec.family != "FR3":
        raise ValueError("FR3 synthetic friction is not defined for this robot")
    indices = [FR3.joints.index(joint) for joint in spec.joints]
    ref = FR3_REF[1][indices]
    if mode == "absolute":
        return ref.copy()
    return ref / TAU_MAX_DOC["FR3"][0][indices] * TAU_MAX_DOC[robot_name][0]


def torque_limit(robot_name: str, scale: float) -> np.ndarray:
    return TAU_MAX_DOC[robot_name][0] * float(scale)


def apply_plant(robot, plant: dict | None) -> dict:
    """Modify robot.model in memory; return the report. Call once per freshly built Robot."""
    p = resolve_plant(plant)
    m = robot.model
    nominal = {"gravity": m.opt.gravity.copy()}
    syn = None
    if np.all(m.dof_frictionloss[robot.dadr] == 0.0):
        syn = p["synthetic_friction"] or "none"
        if syn != "none":
            m.dof_frictionloss[robot.dadr] = synthetic_frictionloss(robot.spec.name, syn)
    elif p["synthetic_friction"] not in (None, "none"):
        raise ValueError(
            f"{robot.spec.name} MJCF already has joint friction; synthetic_friction would overwrite a real model value"
        )
    eff_friction_scale = p["dof_frictionloss_scale"] if p["friction_enabled"] else 0.0
    m.dof_frictionloss[:] *= eff_friction_scale
    m.dof_damping[:] *= p["dof_damping_scale"]
    m.opt.gravity[:] = m.opt.gravity * p["gravity_scale"]
    m.body_mass[:] *= p["mass_scale"]
    m.body_inertia[:] *= p["mass_scale"] * p["inertia_scale"]
    com = None
    if p["com_shift"] is not None:
        link = p["com_shift"]["link"]
        b = (
            robot.site_body
            if link == "last"
            else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link)
        )
        if b < 0:
            raise ValueError(f"plant.com_shift: body '{link}' not found")
        m.body_ipos[b] += np.asarray(p["com_shift"]["dxyz_m"], float)
        com = {
            "body": m.body(b).name,
            "dxyz_m": p["com_shift"]["dxyz_m"],
            "ipos_after_m": m.body_ipos[b].tolist(),
        }
    mujoco.mj_setConst(m, mujoco.MjData(m))
    lim = torque_limit(robot.spec.name, p["torque_limit_scale"])
    if p["servo_enforce_torque_limit"]:
        sv = position_servo_map(robot)
        m.actuator_forcelimited[sv.act] = 1
        m.actuator_forcerange[sv.act, 0] = -lim
        m.actuator_forcerange[sv.act, 1] = lim
    robot.friction_scale = eff_friction_scale
    return {
        "requested": p,
        "nominal_gravity_m_s2": nominal["gravity"].tolist(),
        "applied": {
            "friction_enabled": p["friction_enabled"],
            "effective_frictionloss_scale": eff_friction_scale,
            "dof_frictionloss_Nm": m.dof_frictionloss[robot.dadr].tolist(),
            "dof_damping_Nms_rad": m.dof_damping[robot.dadr].tolist(),
            "gravity_m_s2": m.opt.gravity.tolist(),
            "total_mass_kg": float(m.body_mass.sum()),
            "torque_limit_Nm": lim.tolist(),
            "servo_forcelimited": bool(p["servo_enforce_torque_limit"]),
            "synthetic_friction": syn,
            "com_shift": com,
            "armature_kg_m2": m.dof_armature[robot.dadr].tolist(),
        },
    }


def validate_position_mode(mode, emb, orientation="none"):
    if mode not in ("xy", "xyz"):
        raise ValueError("task.position_mode must be 'xy' or 'xyz'")
    if mode == "xy" and (emb.E.shape[1] != 2 or orientation != "none"):
        raise ValueError("XY requires a planar DS and free orientation")
    return mode


def linear_velocity(ds, emb, p, rate, k_perp, mode="xyz"):
    x, dp = emb.to_plane(p)
    v = rate * emb.L * emb.vel_to_world(ds.f_n(x))
    return v - k_perp * dp * emb.normal if mode == "xyz" else v


def plane_component(v, emb):
    return emb.E @ (emb.E.T @ v)


def servo_task_rows(J6, R, tgt, v, k_rot, emb, mode="xyz"):
    if mode == "xy":
        if tgt.mode != "pos":
            raise ValueError("XY requires free orientation")
        return (emb.E.T @ J6[:3], emb.E.T @ v, 0.0)
    return task_rows(J6, R, tgt, v, k_rot)


R_DES_DEFAULT = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
ORIENT_DEFAULTS = {"K_r": 50.0, "zeta": 1.414}
ORIENTATION_MODES = ("none", "damped", "hold")


@dataclass
class OrientParams:
    """Rotational part of the PIC-DS torque (mode 'damped' | 'hold').
    tau_rot = -D_r omega                       (damped)
    tau_rot =  K_r e_rot - D_r omega           (hold),  e_rot = rot_error(R, R_des) [rad, world]
    K_r = ORIENT_DEFAULTS (recording), D_r = 2 zeta sqrt(K_r) sqrtm(Lambda_r(q_start)) (constant 3x3):
    Lambda_r = (Jr M^-1 Jr^T)^-1 is the rotational task-space inertia at the start configuration, so each
    principal rotational axis gets damping ratio zeta (1.414 = the value of the recording) for that
    inertia. ASSUMPTION: the recording's rotational_stiffness is isotropic [Nm/rad] in the world frame.
    """

    mode: str
    R_des: np.ndarray
    K_r: float
    zeta: float
    D_r: np.ndarray
    Lambda_r: np.ndarray


def rotational_inertia(robot, q) -> np.ndarray:
    """Lambda_r = (Jr M^-1 Jr^T)^-1 [kg m^2] of the TCP frame at q (world axes)."""
    m = robot.model
    d = mujoco.MjData(m)
    d.qpos[robot.qadr] = q
    mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)
    robot.fk(q)
    Jr = robot.jac()[3:]
    return np.linalg.inv(Jr @ np.linalg.solve(M, Jr.T))


def orientation_params(robot, q, task_cfg: dict) -> OrientParams | None:
    mode = task_cfg.get("orientation", "none")
    if mode not in ORIENTATION_MODES:
        raise ValueError(f"task.orientation must be one of {ORIENTATION_MODES}, got {mode!r}")
    if mode == "none":
        return None
    g = {**ORIENT_DEFAULTS, **(task_cfg.get("orientation_gains") or {})}
    K_r, zeta = (float(g["K_r"]), float(g["zeta"]))
    if not (K_r > 0 and zeta > 0):
        raise ValueError("orientation gains K_r and zeta must be > 0")
    Lam = rotational_inertia(robot, q)
    w, V = np.linalg.eigh(0.5 * (Lam + Lam.T))
    D = 2.0 * zeta * np.sqrt(K_r) * (V * np.sqrt(w)) @ V.T
    R_des = np.asarray(task_cfg.get("R_des", R_DES_DEFAULT), float)
    if not (np.allclose(R_des @ R_des.T, np.eye(3), atol=1e-09) and np.linalg.det(R_des) > 0):
        raise ValueError("task.R_des must be a rotation matrix")
    return OrientParams(mode, R_des, K_r, zeta, D, Lam)
