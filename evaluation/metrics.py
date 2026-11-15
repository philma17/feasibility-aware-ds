"""Source tracking and clearance definitions; MSE sums world-coordinate errors."""

from __future__ import annotations
import numpy as np


def arclength(P: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_arclength(P: np.ndarray, n: int) -> np.ndarray:
    """Sample n points uniformly in arc length, removing stationary segments."""
    P = np.asarray(P, float)
    s = arclength(P)
    keep = np.concatenate([[True], np.diff(s) > 0])
    P, s = (P[keep], s[keep])
    if len(P) < 2:
        return np.repeat(P[:1], n, axis=0)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, P[:, d]) for d in range(P.shape[1])], axis=1)


def point_to_polyline(X: np.ndarray, P: np.ndarray, chunk: int = 512) -> np.ndarray:
    A, B = (P[:-1], P[1:])
    AB = B - A
    L2 = np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-300)
    out = np.empty(len(X))
    for i in range(0, len(X), chunk):
        x = X[i : i + chunk, None, :]
        t = np.clip(np.einsum("kij,ij->ki", x - A[None], AB) / L2, 0.0, 1.0)
        proj = A[None] + t[..., None] * AB[None]
        out[i : i + chunk] = np.sqrt(np.min(np.sum((x - proj) ** 2, axis=2), axis=1))
    return out


def chamfer(P: np.ndarray, Q: np.ndarray, n: int = 1000):
    Pr, Qr = (resample_arclength(P, n), resample_arclength(Q, n))
    d1 = point_to_polyline(Pr, Qr)
    d2 = point_to_polyline(Qr, Pr)
    return (0.5 * (d1.mean() + d2.mean()), max(d1.max(), d2.max()))


def frechet_discrete(P: np.ndarray, Q: np.ndarray, n: int = 400) -> float:
    Pr, Qr = (resample_arclength(P, n), resample_arclength(Q, n))
    D = np.linalg.norm(Pr[:, None] - Qr[None], axis=2).tolist()
    prev = np.maximum.accumulate(np.asarray(D[0])).tolist()
    for i in range(1, n):
        Di = D[i]
        row = [max(prev[0], Di[0])] + [0.0] * (n - 1)
        for j in range(1, n):
            a = prev[j] if prev[j] < prev[j - 1] else prev[j - 1]
            if row[j - 1] < a:
                a = row[j - 1]
            row[j] = a if a > Di[j] else Di[j]
        prev = row
    return float(prev[-1])


def dtw_mean(P: np.ndarray, Q: np.ndarray, n: int = 400) -> float:
    Pr, Qr = (resample_arclength(P, n), resample_arclength(Q, n))
    D = np.linalg.norm(Pr[:, None] - Qr[None], axis=2).tolist()
    inf = float("inf")
    Cp = [0.0] + [inf] * n
    Kp = [0] * (n + 1)
    for i in range(1, n + 1):
        Di = D[i - 1]
        Cr = [inf] * (n + 1)
        Kr = [0] * (n + 1)
        for j in range(1, n + 1):
            c, k = (Cp[j - 1], Kp[j - 1])
            if Cp[j] < c:
                c, k = (Cp[j], Kp[j])
            if Cr[j - 1] < c:
                c, k = (Cr[j - 1], Kr[j - 1])
            Cr[j] = Di[j - 1] + c
            Kr[j] = k + 1
        Cp, Kp = (Cr, Kr)
    return float(Cp[n] / Kp[n])


def rmse_time_sync(P: np.ndarray, Q: np.ndarray) -> float:
    if P.shape != Q.shape:
        raise ValueError("Time-synchronous RMSE requires matching sample shapes")
    return float(np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1))))


def max_dev_time_sync(P: np.ndarray, Q: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(P - Q, axis=1)))


class UnsafeSet:

    def __init__(self, spec: dict | None, dim: int):
        self.spec = spec or {}
        self.dim = dim
        self.shape = self.spec.get("shape")
        self.supported = False
        self.reason = None
        self.centers = self.radius = self.lo = self.hi = None
        if not spec:
            self.reason = "no 'unsafe' entry in the S2NNDS config"
        elif self.shape == "Circle":
            self.centers = np.asarray(spec["center"], float).reshape(-1, dim)
            self.radius = float(spec["radius"])
            self.supported = True
        elif self.shape == "Rectangle":
            rng = spec["range"]
            if "unbounded" not in spec:
                self.lo = np.array([rng[i][0] for i in range(dim)], float)
                self.hi = np.array([rng[i][1] for i in range(dim)], float)
                self.supported = True
            elif dim == 2:
                u, mm = (spec["unbounded"], spec["max_min"])
                if u == "x" and mm == "min":
                    self.lo, self.hi = ([rng[0][0], rng[1][0]], [100.0, rng[1][1]])
                elif u == "x" and mm == "max":
                    self.lo, self.hi = ([-100.0, rng[1][0]], [rng[0][0], rng[1][1]])
                elif u == "y" and mm == "min":
                    self.lo, self.hi = ([rng[0][0], rng[1][0]], [rng[0][1], 100.0])
                elif u == "y" and mm == "max":
                    self.lo, self.hi = ([rng[0][0], -100.0], [rng[0][1], rng[1][0]])
                if self.lo is not None:
                    self.lo, self.hi = (np.asarray(self.lo, float), np.asarray(self.hi, float))
                    self.supported = True
                else:
                    self.reason = f"Rectangle variant unbounded={u}, max_min={mm} not known"
            else:
                self.reason = "unbounded Rectangle is only defined for 2D in S2NNDS"
        elif self.shape == "Custom":
            self.reason = "unsafe shape 'Custom' (level set of a config expression): clearance metric not implemented, obstacle safety NOT evaluated"
        else:
            self.reason = f"unsafe shape '{self.shape}' not supported"

    def signed_distance(self, X):
        """X (N, dim) normalized -> (N,) signed distance in normalized units."""
        X = np.atleast_2d(np.asarray(X, float))
        if not self.supported:
            raise RuntimeError(self.reason)
        if self.shape == "Circle":
            d = np.linalg.norm(X[:, None, :] - self.centers[None], axis=2) - self.radius
            return d.min(axis=1)
        q = np.maximum(self.lo - X, X - self.hi)
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(q.max(axis=1), 0.0)
        return outside + inside

    def describe(self) -> dict:
        d = {"shape": self.shape, "supported": self.supported, "reason": self.reason}
        if self.shape == "Circle" and self.supported:
            d.update(
                centers_n=self.centers.tolist(),
                radius_n=self.radius,
                geometry="sphere" if self.dim == 3 else "circle",
            )
        if self.shape == "Rectangle" and self.supported:
            d.update(lo_n=self.lo.tolist(), hi_n=self.hi.tolist())
        if self.shape == "Custom":
            d["function"] = self.spec.get("function")
        return d


def tracking_metrics(P, P_id, goal, L, dt, success_frac) -> dict:
    """Shared task-space metrics for both execution modes (world frame, metres).

    time-synchronized error e_k = ||P_k - P_id,k||, P_id,k = ideal DS Euler rollout at the same
    t_k = k dt from the same start x0 (open-loop reference, NOT re-anchored at the robot).
    geometric deviation g_k = distance of P_k to the ideal path polyline (timing-invariant)."""
    e = np.linalg.norm(P - P_id, axis=1)
    g = point_to_polyline(P, P_id)
    ch_mean, ch_max = chamfer(P, P_id)
    dist_goal = np.linalg.norm(P - goal, axis=1)
    thr = success_frac * L
    reached = np.flatnonzero(dist_goal < thr)
    return (
        {
            "final_goal_err_m": float(dist_goal[-1]),
            "mse_sync_m2": float(np.mean(e**2)),
            "rmse_sync_m": float(np.sqrt(np.mean(e**2))),
            "max_err_sync_m": float(e.max()),
            "t_max_err_sync_s": float(np.argmax(e) * dt),
            "geom_dev_mean_m": float(g.mean()),
            "geom_dev_max_m": float(g.max()),
            "chamfer_mean_m": float(ch_mean),
            "hausdorff_m": float(ch_max),
            "frechet_m": frechet_discrete(P, P_id),
            "t_reach_s": float(reached[0] * dt) if len(reached) else None,
        },
        e,
        g,
    )


def summarize(result, case):
    """Evaluate finite saved states; torque and domain events remain separate."""
    tr = result["traj"]
    n = len(tr["t"]) - 1
    goal = case.emb.to_world(np.zeros(len(case.x0_n)))
    tracking = result.get("tracking")
    if tracking is None:
        tracking, _, _ = tracking_metrics(
            tr["P"], tr["P_ideal"], goal, case.emb.L, case.sp.dt, case.sp.success_frac
        )
    command = tr["tau_cmd"][:n]
    applied = tr["tau_applied"][:n]
    saturated = np.any(np.abs(command) > case.tau_limit + 1e-9, axis=1)
    domain = case.scenario["domain"]
    outside = np.any((tr["Xn"] < domain[:, 0]) | (tr["Xn"] > domain[:, 1]), axis=1)
    clearance = (
        case.emb.L * case.unsafe.signed_distance(tr["Xn"]) if case.unsafe is not None else None
    )
    return {
        **tracking,
        "dtw_mean_m": dtw_mean(tr["P"], tr["P_ideal"]),
        "success": bool(result["success"]),
        "finite": bool(result["finite"]),
        "final_distance_m": float(result["final_dist_m"]),
        "min_clearance_m": None if clearance is None else float(clearance.min()),
        "clearance_status": case.scenario["unsafe_info"],
        "torque_saturation_steps": int(np.count_nonzero(saturated)),
        "torque_saturation_fraction": float(np.mean(saturated)),
        "peak_applied_torque_ratio": float(np.max(np.abs(applied) / case.tau_limit)),
        "domain_exit": bool(outside.any()),
        "domain_exit_time_s": float(tr["t"][np.argmax(outside)]) if outside.any() else None,
        "n_intervals": n,
        "n_states": len(tr["t"]),
    }
