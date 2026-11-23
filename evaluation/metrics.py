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
    if len(X) * len(P) > 1_000_000:
        return _point_to_polyline_tree(np.asarray(X), np.asarray(P))
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


def _point_to_polyline_tree(X, P):
    """Exact segment distances with bounding-box pruning for long saved paths."""
    from scipy.spatial import cKDTree

    A, AB = P[:-1], np.diff(P, axis=0)
    L2 = np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-300)
    # A nearest vertex supplies an upper bound, never a segment approximation.
    _, nearest = cKDTree(P).query(X)
    best = np.sum((X - P[nearest]) ** 2, axis=1)
    scale = max(1.0, float(np.abs(P).max()), float(np.abs(X).max()))
    eps = np.finfo(float).eps
    slack = 64 * eps * (best + scale * np.sqrt(best) + eps * scale**2)

    def visit(segments, queries):
        lo = np.minimum(A[segments], P[segments + 1]).min(axis=0)
        hi = np.maximum(A[segments], P[segments + 1]).max(axis=0)
        delta = np.maximum(np.maximum(lo - X[queries], X[queries] - hi), 0)
        queries = queries[np.sum(delta**2, axis=1) <= best[queries] + slack[queries]]
        if not len(queries):
            return
        if len(segments) <= 32:
            for offset in range(0, len(queries), 512):
                ids = queries[offset : offset + 512]
                d = X[ids, None] - A[segments]
                t = np.clip(np.einsum("kij,ij->ki", d, AB[segments]) / L2[segments], 0, 1)
                proj = A[segments] + t[..., None] * AB[segments]
                distances = np.sum((X[ids, None] - proj) ** 2, axis=2)
                best[ids] = np.minimum(best[ids], distances.min(axis=1))
            return
        centers = (A[segments] + P[segments + 1]) / 2
        axis = np.argmax(np.ptp(centers, axis=0))
        order = np.argsort(centers[:, axis], kind="stable")
        middle = len(order) // 2
        visit(segments[order[:middle]], queries)
        visit(segments[order[middle:]], queries)

    visit(np.arange(len(A)), np.arange(len(X)))
    return np.sqrt(best)


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


def tracking_metrics(P, P_id, goal, L, dt, success_frac) -> tuple[dict, np.ndarray, np.ndarray]:
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


def summarize_trajectory(tr, cfg, manifest, tracking=None):
    """Evaluate logged samples using the coordinates and limits recorded by the producer."""
    metrics = {}
    times = tr.get("t")
    n = len(times) - 1 if times is not None else None
    measured, ideal = tr.get("P"), tr.get("P_ideal")
    embedding = manifest.get("embedding")
    inputs = manifest.get("evaluation_inputs", {})
    if n is not None:
        metrics.update(n_intervals=n, n_states=len(times))
    if "Q" in tr and "QD" in tr:
        metrics["finite"] = bool(np.isfinite(tr["Q"]).all() and np.isfinite(tr["QD"]).all())
    if measured is not None and len(measured) and np.isfinite(measured).all() and embedding is not None:
        scale = embedding["L_m"]
        anchor = np.asarray(embedding["anchor_n"], float)
        axes = np.asarray(embedding["axes"], float)
        goal = np.asarray(embedding["origin"], float) + scale * ((np.zeros_like(anchor) - anchor) @ axes.T)
        distance = np.linalg.norm(measured - goal, axis=1)
        sim = cfg["sim"]
        radius = sim["success_frac"] * scale
        metrics.update(success=bool(distance[-1] < radius), final_distance_m=float(distance[-1]))
        if times is not None and len(times) == len(distance):
            reached = np.flatnonzero(distance < radius)
            metrics["t_reach_s"] = float(times[reached[0]]) if len(reached) else None
        if ideal is not None and ideal.shape == measured.shape and len(ideal) > 1 and np.isfinite(ideal).all():
            if tracking is None:
                tracking, _, _ = tracking_metrics(measured, ideal, goal, scale, sim["dt"], sim["success_frac"])
            metrics.update(tracking, dtw_mean_m=dtw_mean(measured, ideal))

    limits = inputs.get("torque_limit_Nm", tr.get("tau_limit"))
    if n is not None and n > 0 and limits is not None:
        limits = np.asarray(limits)
        for name in ("tau_cmd", "tau_applied"):
            torque = tr.get(name)
            if torque is None or torque.shape not in ((n, len(limits)), (n + 1, len(limits))):
                continue
            if not np.isfinite(torque[:n]).all():
                continue
            if name == "tau_cmd":
                saturated = np.any(np.abs(torque[:n]) > limits + 1e-9, axis=1)
                metrics.update(torque_saturation_steps=int(saturated.sum()),
                               torque_saturation_fraction=float(saturated.mean()))
            else:
                metrics["peak_applied_torque_ratio"] = float(np.max(np.abs(torque[:n]) / limits))

    normalized = tr.get("Xn")
    domain = inputs.get("domain")
    if normalized is not None and len(normalized) and np.isfinite(normalized).all():
        if domain is not None:
            domain = np.asarray(domain)
            outside = np.any((normalized < domain[:, 0]) | (normalized > domain[:, 1]), axis=1)
            metrics["domain_exit"] = bool(outside.any())
            if times is not None and len(times) == len(outside):
                metrics["domain_exit_time_s"] = float(times[np.argmax(outside)]) if outside.any() else None
        if inputs.get("unsafe") is not None and embedding is not None:
            unsafe = UnsafeSet(inputs["unsafe"], normalized.shape[1])
            if unsafe.supported:
                metrics["min_clearance_m"] = float((embedding["L_m"] * unsafe.signed_distance(normalized)).min())
    clearance = tr.get("clearance_m")
    if "min_clearance_m" not in metrics and clearance is not None and len(clearance) and np.isfinite(clearance).all():
        metrics["min_clearance_m"] = float(clearance.min())
    metrics.setdefault("min_clearance_m", None)
    metrics["clearance_status"] = inputs.get("clearance_status")
    return metrics


def evaluate_saved_run(directory, manifest=None, *, tracking=None):
    """Compute derived metrics without constructing a live simulation case."""
    import json
    from pathlib import Path
    from evaluation.run_artifacts import read_config, read_trajectory

    directory = Path(directory)
    if manifest is None:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    trajectory = read_trajectory(directory, manifest)
    cfg = read_config(directory, manifest)
    if "evaluation_inputs" not in manifest and ("config" in manifest or "input_sha256" in manifest):
        from evaluation.barrier import saved_model_inputs

        try:
            _, _, spec = saved_model_inputs(directory, manifest)
        except (OSError, KeyError, ValueError):
            # Historical runs can still provide tracking and recorded clearance.
            pass
        else:
            unsafe = UnsafeSet(spec.get("unsafe"), len(manifest["embedding"]["anchor_n"]))
            manifest = {**manifest, "evaluation_inputs": {
                "domain": spec["domain"]["range"],
                "unsafe": spec.get("unsafe"),
                "clearance_status": unsafe.describe(),
            }}
    return summarize_trajectory(trajectory, cfg, manifest, tracking)
