"""PTH vector fields, source normalization, embeddings, and ideal Euler rollout."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
import yaml
from scipy.io import loadmat
from evaluation.metrics import UnsafeSet

ROOT = Path(__file__).resolve().parents[1]


def input_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def model_catalog(path="configs/models.yaml"):
    with input_path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["models"]


def load_torch_reference(path):
    """Load trusted local pickled classes using the vendored S2NNDS adapter."""
    import torch

    adapter = ROOT / "vendor/S2NNDS/NNModels.py"
    spec = importlib.util.spec_from_file_location("NNModels", adapter)
    module = importlib.util.module_from_spec(spec)
    sys.modules["NNModels"] = module
    spec.loader.exec_module(module)
    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    return model


class PthMLP:
    """Evaluate checkpoint weights with the source float64 NumPy operations."""

    def __init__(self, path: str):
        import torch.nn as nn

        m = load_torch_reference(path)
        acts = {nn.Tanh: "Tanh", nn.ReLU: "Relu"}
        act = acts.get(type(m.sigmoid_f))
        if act is None and type(m.sigmoid_f).__name__ != "ELU":
            raise ValueError(f"Activation {type(m.sigmoid_f).__name__} is not supported")
        self.path = path
        self.dim = m.layers_f[0].in_features
        self.ops = []
        for i, layer in enumerate(m.layers_f):
            if layer.bias is not None:
                raise ValueError("Unexpected bias in checkpoint layer")
            self.ops.append(("matmul", layer.weight.detach().double().numpy().T))
            if i < len(m.layers_f) - 1:
                self.ops.append(("elu", 1.0) if act is None else ("act", act))
            else:
                self.ops.append(("act", "Tanh"))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float64)
        for typ, p in self.ops:
            if typ == "matmul":
                y = y @ p
            elif typ == "affine":
                y = y @ p[0] + p[1]
            elif typ == "add":
                y = y + p
            elif typ == "elu":
                y = np.where(y > 0, y, p * (np.exp(np.minimum(y, 0.0)) - 1.0))
            else:
                y = _ACT[p](y)
        return y


_ACT = {
    "Tanh": np.tanh,
    "Relu": lambda x: np.maximum(x, 0.0),
    "Sin": np.sin,
    "Softplus": lambda x: np.logaddexp(0.0, x),
}


class LearnedDS:
    """Normalized vector field; rates and world embedding are applied separately."""

    def __init__(self, shape, dataset, checkpoint):
        self.shape, self.dataset, self.source = shape, dataset, "pth"
        self.f = PthMLP(str(checkpoint))

    def f_n(self, x_n: np.ndarray) -> np.ndarray:
        return self.f(np.asarray(x_n, float).reshape(1, -1))[0]

    def rollout_ideal(self, x0_n, dt: float, n_steps: int, rate: float) -> np.ndarray:
        X = np.empty((n_steps + 1, 2))
        X[0] = x0_n
        for k in range(n_steps):
            X[k + 1] = X[k] + dt * rate * self.f_n(X[k])
        return X


@dataclass(frozen=True)
class LasaScaling:
    shape: str
    ps_mm: float
    vs_mm_s: float
    x0_n: np.ndarray
    demos_n: list = field(repr=False)
    dt_demo: float = 0.0


def lasa_scaling(shape: str, demo_path) -> LasaScaling:
    from pyLasaDataset.dataset import _Data

    ds = _Data(loadmat(demo_path), shape)
    P = np.concatenate([d.pos for d in ds.demos], axis=1)
    V = np.concatenate([d.vel for d in ds.demos], axis=1)
    ends = np.array([d.pos[:, -1] for d in ds.demos])
    if np.max(np.abs(ends)) > 1e-09:
        raise ValueError(
            f"{shape}: demonstration endpoints are not at the origin; check normalization"
        )
    ps = float(np.max(np.abs(P)))
    vs = float(np.max(np.abs(V)))
    x0 = np.mean([d.pos[:, 0] for d in ds.demos], axis=0) / ps
    return LasaScaling(shape, ps, vs, x0, [d.pos.T / ps for d in ds.demos], float(ds.dt))


def shape_scaling(dataset: str, name: str, package) -> LasaScaling:
    if dataset == "LASA":
        return lasa_scaling(package["demo_name"], input_path(package["demos"][0]))
    cfg = json.loads(input_path(package["metadata"]).read_text())
    pos, vel = ([], [])
    if dataset == "2D_Shapes":
        folder = input_path(package["demos"][0]).parent
        sub = cfg["hyperparameters"].get("subsample", 10)
        for f in sorted(p.name for p in folder.iterdir()):
            if not f.endswith(".csv"):
                continue
            df = pd.read_csv(folder / f)
            p = np.vstack([df.x, df.y])[:, ::sub]
            v = np.vstack([df.dx, df.dy])[:, ::sub]
            pos.append(p - p[:, -1:])
            vel.append(v)
        dt = 0.01
    elif dataset == "3D_Shapes":
        import scipy.io

        mat = scipy.io.loadmat(input_path(package["demos"][0]))
        for traj in np.squeeze(mat["data"]):
            p, v = (traj[:3, :], traj[3:, :])
            pos.append(p - p[:, -1:])
            vel.append(v)
        dt = 0.01
    else:
        raise ValueError(dataset)
    P = np.concatenate(pos, axis=1)
    V = np.concatenate(vel, axis=1)
    ps = float(np.max(np.abs(P)))
    vs = float(np.max(np.abs(V)))
    x0 = np.mean([p[:, 0] for p in pos], axis=0) / ps
    return LasaScaling(name, ps, vs, x0, [p.T / ps for p in pos], dt)


@dataclass
class PlanarEmbedding:
    """p = origin + L * [e1 e2] (x_n - anchor_n)."""

    origin: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    L: float
    anchor_n: np.ndarray

    def __post_init__(self):
        self.origin = np.asarray(self.origin, float)
        self.e1 = np.asarray(self.e1, float) / np.linalg.norm(self.e1)
        e2 = np.asarray(self.e2, float)
        e2 = e2 - e2 @ self.e1 * self.e1
        self.e2 = e2 / np.linalg.norm(e2)
        self.normal = np.cross(self.e1, self.e2)
        self.anchor_n = np.asarray(self.anchor_n, float)
        self.E = np.stack([self.e1, self.e2], axis=1)

    def to_world(self, x_n: np.ndarray) -> np.ndarray:
        x_n = np.asarray(x_n, float)
        return self.origin + self.L * ((x_n - self.anchor_n) @ self.E.T)

    def to_plane(self, p: np.ndarray):
        d = np.asarray(p, float) - self.origin
        x_n = d @ self.E / self.L + self.anchor_n
        return (x_n, d @ self.normal)

    def vel_to_world(self, v2: np.ndarray) -> np.ndarray:
        return np.asarray(v2, float) @ self.E.T


def demo_time_rate(sc: LasaScaling) -> float:
    return sc.vs_mm_s / sc.ps_mm


class SpatialEmbedding:
    """p = origin + L * A (x_n - anchor_n), A orthonormal (3x3)."""

    def __init__(self, origin, axes, L: float, anchor_n):
        A = np.asarray(axes, float)
        if A.shape != (3, 3):
            raise ValueError(f"spatial axes must be 3x3, got {A.shape}")
        if not np.allclose(A.T @ A, np.eye(3), atol=1e-09):
            raise ValueError("spatial axes must be orthonormal (distances are scaled by L only)")
        if L <= 0:
            raise ValueError("L must be > 0")
        self.origin = np.asarray(origin, float).reshape(3)
        self.E = A
        self.L = float(L)
        self.anchor_n = np.asarray(anchor_n, float).reshape(3)
        self.normal = np.zeros(3)

    def to_world(self, x_n):
        return self.origin + self.L * ((np.asarray(x_n, float) - self.anchor_n) @ self.E.T)

    def to_plane(self, p):
        """Name kept for interface compatibility with PlanarEmbedding: exact inverse, no
        projection. Returns (x_n, 0.0)."""
        d = np.asarray(p, float) - self.origin
        return (d @ self.E / self.L + self.anchor_n, 0.0)

    def vel_to_world(self, v_n):
        return np.asarray(v_n, float) @ self.E.T


def make_embedding(dim: int, center, L: float, plane=None, axes=None, anchor=None):
    """dim 2 -> PlanarEmbedding(center, e1, e2, L, anchor); dim 3 -> SpatialEmbedding."""
    anchor = np.zeros(dim) if anchor is None else np.asarray(anchor, float)
    if len(anchor) != dim:
        raise ValueError(f"anchor has {len(anchor)} entries, model dimension is {dim}")
    if dim == 2:
        return PlanarEmbedding(
            np.asarray(center, float), plane["e1"], plane["e2"], float(L), anchor
        )
    if dim == 3:
        return SpatialEmbedding(center, np.eye(3) if axes is None else axes, L, anchor)
    raise ValueError(f"no world embedding for {dim}D models")


def check_dims(ds, emb, x0_n) -> int:
    """Model input dimension == embedding dimension == len(x0_n); returns it."""
    d_model = int(ds.f.dim)
    d_emb = int(np.asarray(emb.E).shape[1])
    if not d_model == d_emb == len(x0_n):
        raise ValueError(
            f"dimension mismatch: model input {d_model}, embedding {d_emb}, x0 {len(x0_n)}"
        )
    return d_model


def ideal_rollout(ds, x0_n, dt: float, n_steps: int, rate: float) -> np.ndarray:
    """Explicit Euler of the ideal DS, (n_steps+1, dim). For 2D this calls
    LearnedDS.rollout_ideal (unchanged reference); the same scheme for other dimensions."""
    x0_n = np.asarray(x0_n, float)
    if len(x0_n) == 2:
        return ds.rollout_ideal(x0_n, dt, n_steps, rate)
    X = np.empty((n_steps + 1, len(x0_n)))
    X[0] = x0_n
    for k in range(n_steps):
        X[k + 1] = X[k] + dt * rate * ds.f_n(X[k])
    return X


SPATIAL_DEFAULT = {
    "center": [0.50, 0.0, 0.50],
    "L_m": 0.10,
    "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "anchor": [0.0, 0.0, 0.0],
}
MULTISTART_DEFAULT = {"seed": 0, "n_starts": 20}


def sample_starts(box, seed, n_starts):
    """Draw the source uniform start set; prefixes are stable when the count changes."""
    box = np.asarray(box, float)
    return np.random.default_rng(int(seed)).uniform(
        box[:, 0], box[:, 1], size=(int(n_starts), len(box))
    )


def load_scenario(cfg: dict) -> dict:
    package = model_catalog(cfg["model_catalog"])[cfg["model"]]
    if not package["selectable"]:
        raise ValueError(package["reason"])
    dataset, shape = package["dataset"], package["shape"]
    if cfg.get("model_source", "pth") != "pth":
        raise ValueError("only the verified model_f.pth is supported (model_source: pth)")
    ds = LearnedDS(shape, dataset, input_path(package["checkpoint"]))
    dim = int(ds.f.dim)
    sc = shape_scaling(dataset, shape, package)
    if dim != len(sc.x0_n):
        raise ValueError(f"{dataset}/{shape}: model dim {dim} != demonstration dim {len(sc.x0_n)}")
    if dim not in (2, 3):
        raise ValueError(f"{dim}D models are not supported")
    rate = demo_time_rate(sc) if cfg.get("time_rate", "demo") == "demo" else float(cfg["time_rate"])
    s2_path = str(input_path(package["metadata"]))
    s2 = json.loads(Path(s2_path).read_text())
    warnings = []
    t = cfg["task"]
    start = t.get("start", "demo_mean")
    box = None
    if s2 is not None and s2.get("plotting", {}).get("init_range"):
        box = np.asarray(s2["plotting"]["init_range"], float)
    if start == "demo_mean":
        x0, label = (np.asarray(sc.x0_n, float), "demo_mean")
    elif isinstance(start, dict) and "x0" in start:
        x0, label = (np.asarray(start["x0"], float), "x0")
        if len(x0) != dim:
            raise ValueError(f"task.start.x0 needs {dim} values")
    elif isinstance(start, dict) and "index" in start:
        if box is None:
            raise ValueError(
                f"no init_range for {dataset}/{shape} in {s2_path}; cannot draw a verified start set"
            )
        ms = {**MULTISTART_DEFAULT, **(cfg.get("multistart") or {})}
        X0 = sample_starts(box, ms["seed"], ms["n_starts"])
        i = int(start["index"])
        if not 0 <= i < len(X0):
            raise ValueError(f"task.start.index must be in 0..{len(X0) - 1}")
        x0, label = (X0[i], f"start{i}")
    else:
        raise ValueError(f"unknown task.start {start!r}")
    dom = None
    if s2 is not None:
        dom = np.asarray(s2["domain"]["range"], float)
        if np.any(x0 < dom[:, 0]) or np.any(x0 > dom[:, 1]):
            warnings.append("x0 is outside the S2NNDS training domain")
    if box is not None and (np.any(x0 < box[:, 0]) or np.any(x0 > box[:, 1])):
        warnings.append("x0 is outside the verified initial set (no convergence guarantee)")
    if dim == 2:
        emb = make_embedding(2, t["center"], float(t["L_m"]), plane=t["plane"], anchor=t["anchor"])
    else:
        spatial = cfg.get("spatial") or {}
        unknown = set(spatial) - set(SPATIAL_DEFAULT)
        if unknown:
            raise ValueError(f"unknown spatial keys {sorted(unknown)}")
        sp3 = {**SPATIAL_DEFAULT, **{k: t[k] for k in ("center", "L_m") if k in t}, **spatial}
        emb = make_embedding(
            3, sp3["center"], float(sp3["L_m"]), axes=sp3["axes"], anchor=sp3["anchor"]
        )
    unsafe = UnsafeSet(s2.get("unsafe") if s2 else None, dim)
    if not unsafe.supported:
        warnings.append(f"unsafe-set clearance disabled: {unsafe.reason}")
    return {
        "package": package,
        "ds": ds,
        "sc": sc,
        "rate": rate,
        "dim": dim,
        "x0": x0,
        "start_label": label,
        "emb": emb,
        "unsafe": unsafe if unsafe.supported else None,
        "unsafe_info": unsafe.describe(),
        "s2nnds_config": s2_path if s2 is not None else None,
        "domain": dom,
        "init_box": box,
        "warnings": warnings,
        "dataset": dataset,
        "shape": shape,
    }
