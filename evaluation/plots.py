"""Plot saved trajectories and model regions without running a simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np
import yaml

from evaluation.metrics import UnsafeSet


def task_plane_mm(points, embedding):
    """Project world positions onto the saved task basis, relative to its origin."""
    origin = np.asarray(embedding["origin"], dtype=float)
    axes = np.asarray(embedding["axes"], dtype=float)
    if origin.shape != (3,) or axes.shape != (3, 2):
        raise ValueError("Trajectory plotting requires a saved two-dimensional task embedding")
    if not np.isfinite(origin).all() or not np.isfinite(axes).all():
        raise ValueError("Task embedding must be finite")
    if not np.allclose(axes.T @ axes, np.eye(2), rtol=0, atol=1e-10):
        raise ValueError("Task axes must be orthonormal")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("Trajectory must have nonempty shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("Trajectory must be finite")
    # Same projection as LfD_sim/experiments/run_pic.py::figure:
    # 1000 * L * (to_plane(points) - anchor_n). L and anchor cancel here.
    return 1000.0 * ((points - origin) @ axes)


def world_axes(embedding):
    """Choose the world-coordinate view with the largest projected task-plane area."""
    axes = np.asarray(embedding["axes"], float)
    if axes.shape != (3, 2):
        raise ValueError("Trajectory plotting requires a saved two-dimensional task embedding")
    pairs = [(0, 1), (0, 2), (1, 2)]
    return list(max(pairs, key=lambda pair: abs(np.linalg.det(axes[list(pair)]))))


def model_to_world(points, embedding):
    """Use the run's effective embedding, as in LfD_sim/pic/plots.py::to_world."""
    return np.asarray(embedding["origin"]) + embedding["L_m"] * (
        (np.asarray(points) - np.asarray(embedding["anchor_n"])) @ np.asarray(embedding["axes"]).T
    )


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def saved_model_inputs(run_dir, manifest):
    """Resolve catalog inputs only if they still match the hashes saved by the run."""
    from sim.ds import input_path, model_catalog

    cfg_path = run_dir / "config_resolved.yaml"
    if file_sha256(cfg_path) != manifest["artifact_sha256"]["config_resolved.yaml"]:
        raise ValueError("Saved effective configuration hash does not match the run")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    catalog = cfg["model_catalog"]
    if file_sha256(input_path(catalog)) != manifest["input_sha256"][catalog]:
        raise ValueError("Model catalog has changed since this run")
    package = model_catalog(catalog)[manifest["model"]]
    for key in ("metadata", "checkpoint"):
        path = package[key]
        if file_sha256(input_path(path)) != manifest["input_sha256"][path]:
            raise ValueError(f"Model input has changed since this run: {path}")
    spec = json.loads(input_path(package["metadata"]).read_text(encoding="utf-8"))
    return cfg, package, spec


def barrier_values(path, points):
    """Reuse LfD_sim/experiments/inspect_lasa_packages.py float32 network evaluation."""
    import torch
    from sim.ds import load_torch_reference

    model = load_torch_reference(path).cpu()
    if points.shape[1] != model.input_size or not np.isfinite(points).all():
        raise ValueError("Barrier input dimension or finite values do not match model")
    with torch.no_grad():
        values = model(torch.as_tensor(points, dtype=torch.float32)).numpy().reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite barrier output")
    return values


def obstacle_level(spec, points):
    """Retain the source level sets; never evaluate arbitrary config expressions."""
    unsafe = UnsafeSet(spec, 2)
    if unsafe.supported:
        return unsafe.signed_distance(points)
    # Exact supported Custom cases from LfD_sim/analysis/lasa_comparison.py::unsafe_level.
    expression = "".join(spec.get("function", "").split())
    x, y = points.T
    if expression == "(((x+0.3)/6)**2)**(1/3)+(((y-0.4)/3)**2)**(1/3)-0.15":
        return (((x + 0.3) / 6) ** 2) ** (1 / 3) + (((y - 0.4) / 3) ** 2) ** (1 / 3) - 0.15
    if expression == "torch.max(torch.max(y,-4*x-0.8*y-1.6),-0.8*y+4*x+0.8)":
        return np.maximum(np.maximum(y, -4 * x - 0.8 * y - 1.6), -0.8 * y + 4 * x + 0.8)
    raise ValueError(f"Unsupported obstacle expression: {spec}")


def draw_regions(ax, spec, embedding, displayed_axes, paths, barrier_model):
    """Map normalized regions to the same world view as paths and the model goal."""
    # Adapted from LfD_sim/analysis/failure/report_figures.py::draw_regions.
    bounds = np.asarray(spec.get("plotting", {}).get("range", spec["domain"]["range"]), float)
    normalized = np.concatenate(
        [
            task_plane_mm(p, embedding) / (1000 * embedding["L_m"])
            + np.asarray(embedding["anchor_n"])
            for p in paths
        ]
        + [np.zeros((1, 2))]
    )
    lo = np.minimum(bounds[:, 0], normalized.min(axis=0))
    hi = np.maximum(bounds[:, 1], normalized.max(axis=0))
    grid = np.stack(np.meshgrid(*[np.linspace(a, b, 320) for a, b in zip(lo, hi)]), -1)
    points = grid.reshape(-1, 2)
    world = (1000 * model_to_world(points, embedding)[:, displayed_axes]).reshape(320, 320, 2)
    wx, wy = world[..., 0], world[..., 1]
    handles = []
    if spec.get("Barrier", False):
        B = barrier_values(barrier_model, points).reshape(grid.shape[:2])
        ax.contourf(wx, wy, B, levels=[-np.inf, 0], colors=["#d4f5d4"], zorder=0)
        ax.contour(wx, wy, B, levels=[0], colors=["#1a8f1a"], linewidths=1, zorder=1)
        handles += [
            Patch(fc="#d4f5d4", ec="#1a8f1a", label=r"$B\leq 0$"),
            Line2D([], [], color="#1a8f1a", label=r"$B=0$"),
        ]
    unsafe_spec = spec.get("unsafe")
    if unsafe_spec:
        unsafe = UnsafeSet(unsafe_spec, 2)
        boundaries = []
        if unsafe.supported and unsafe.shape == "Circle":
            theta = np.linspace(0, 2 * np.pi, 200)
            circle = np.column_stack([np.cos(theta), np.sin(theta)])
            boundaries = [c + unsafe.radius * circle for c in unsafe.centers]
        elif unsafe.supported and unsafe.shape == "Rectangle":
            a, b = np.maximum(unsafe.lo, lo), np.minimum(unsafe.hi, hi)
            if np.all(a <= b):
                boundaries = [np.array([[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]])]
        else:
            level = obstacle_level(unsafe_spec, points).reshape(grid.shape[:2])
            ax.contourf(
                wx, wy, level, levels=[-np.inf, 0], colors=["tab:red"], alpha=0.65, zorder=2
            )
            ax.contour(wx, wy, level, levels=[0], colors=["tab:red"], linewidths=1, zorder=2)
        for boundary in boundaries:
            vertices = 1000 * model_to_world(boundary, embedding)[:, displayed_axes]
            ax.add_patch(
                Polygon(vertices, facecolor="tab:red", edgecolor="tab:red", alpha=0.65, zorder=2)
            )
        handles.append(Patch(fc="tab:red", ec="tab:red", alpha=0.65, label="Obstacles"))
    return handles


def experiment_setting(cfg):
    """Describe nominal plant conditions only when the recorded settings establish them."""
    p = cfg["plant"]
    nominal = p["friction_enabled"] and all(
        p[k] == 1
        for k in (
            "dof_frictionloss_scale",
            "dof_damping_scale",
            "gravity_scale",
            "mass_scale",
            "inertia_scale",
            "torque_limit_scale",
        )
    )
    nominal = nominal and p["synthetic_friction"] is None and p["com_shift"] is None
    nominal = nominal and not cfg["disturbance"]["enabled"] and cfg.get("ds_speed_factor", 1) == 1
    return "Nominal" if nominal else "Configured"


def plot_trajectory(run_dir, barrier_model=None):
    """Write a world-coordinate view; model goal is x_n=0, not the rollout endpoint."""
    from sim.ds import input_path

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg, package, spec = saved_model_inputs(run_dir, manifest)
    emb = manifest["embedding"]
    displayed_axes = world_axes(emb)
    with np.load(run_dir / "traj.npz", allow_pickle=False) as arrays:
        measured_world, ideal_world = arrays["P"], arrays["P_ideal"]
    measured = 1000 * measured_world[:, displayed_axes]
    ideal = 1000 * ideal_world[:, displayed_axes]
    goal = 1000 * model_to_world(np.zeros(2), emb)[displayed_axes]
    plot_inputs = {
        "world_axes": displayed_axes,
        "metadata": package["metadata"],
        "metadata_sha256": manifest["input_sha256"][package["metadata"]],
    }
    if spec.get("Barrier", False):
        barrier_model = (
            Path(barrier_model)
            if barrier_model
            else input_path(package["checkpoint"]).with_name("model_b.pth")
        )
        if not barrier_model.is_file():
            raise ValueError(
                "Barrier model missing; supply the matching local --barrier-model PATH"
            )
        # The extra plotting input must belong to the same vector-field package.
        sibling_f = barrier_model.with_name("model_f.pth")
        if file_sha256(sibling_f) != manifest["input_sha256"][package["checkpoint"]]:
            raise ValueError("Barrier package model_f.pth does not match this run")
        plot_inputs.update(
            barrier_model=str(barrier_model.resolve()), barrier_sha256=file_sha256(barrier_model)
        )

    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    try:
        regions = draw_regions(
            ax, spec, emb, displayed_axes, [measured_world, ideal_world], barrier_model
        )
        ax.plot(*ideal.T, "k--", lw=1.2, label="Ideal DS", zorder=3)
        ax.plot(*measured.T, color="tab:orange", lw=1.8, label="Robot Path", zorder=4)
        ax.plot(*goal, "*", color="black", ms=13, label="DS Goal", zorder=5)
        names = "xyz"
        ax.set(
            xlabel=rf"${names[displayed_axes[0]]}\,[\mathrm{{mm}}]$",
            ylabel=rf"${names[displayed_axes[1]]}\,[\mathrm{{mm}}]$",
            title=f"{package['shape']} / {manifest['robot']} / {experiment_setting(cfg)}",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles + regions)
        path = run_dir / "plots" / "trajectory.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, metadata={"Description": json.dumps(plot_inputs)})
    finally:
        plt.close(fig)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", type=Path, help="saved run containing traj.npz and manifest.json"
    )
    parser.add_argument(
        "--barrier-model",
        type=Path,
        help="trusted local model_b.pth; default: beside the run's model_f.pth",
    )
    args = parser.parse_args(argv)
    try:
        path = plot_trajectory(args.run_dir, args.barrier_model)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Plot error: {exc}\n")
    print(f"Trajectory plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
