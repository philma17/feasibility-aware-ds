"""Plot saved trajectories and model regions without running a simulation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np

from evaluation.metrics import UnsafeSet
from evaluation.run_artifacts import read_trajectory
from evaluation.barrier import (
    barrier_values, domain_box, evaluate_run, file_sha256, model_to_world,
    recorded_model_hashes, resolve_barrier_model, saved_model_inputs, world_to_model,
)
from evaluation.aggregate import EVENTS, event_value


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
    box = domain_box(spec)
    bounds = np.asarray(spec.get("plotting", {}).get("range", spec["domain"]["range"]), float)
    normalized = np.concatenate(
        [
            task_plane_mm(p, embedding) / (1000 * embedding["L_m"])
            + np.asarray(embedding["anchor_n"])
            for p in paths
        ]
        + [np.zeros((1, 2))]
    )
    lo = np.minimum(np.minimum(bounds[:, 0], box[:, 0]), normalized.min(axis=0))
    hi = np.maximum(np.maximum(bounds[:, 1], box[:, 1]), normalized.max(axis=0))
    grid = np.stack(np.meshgrid(*[np.linspace(a, b, 320) for a, b in zip(lo, hi)]), -1)
    points = grid.reshape(-1, 2)
    world = (1000 * model_to_world(points, embedding)[:, displayed_axes]).reshape(320, 320, 2)
    wx, wy = world[..., 0], world[..., 1]
    corners = np.array([
        [box[0, 0], box[1, 0]], [box[0, 1], box[1, 0]],
        [box[0, 1], box[1, 1]], [box[0, 0], box[1, 1]],
    ])
    vertices = 1000 * model_to_world(corners, embedding)[:, displayed_axes]
    domain_patch = Polygon(vertices, fill=False, edgecolor="0.35", ls="--", lw=1.2, zorder=2)
    ax.add_patch(domain_patch)
    handles = [Line2D([], [], color="0.35", ls="--", label=r"$X$ (model box)")]
    if spec.get("Barrier", False):
        # Evaluate on X itself: neither the fill nor its contour extrapolates beyond X.
        bgrid = np.stack(np.meshgrid(*[np.linspace(a, b, 320) for a, b in box]), -1)
        bpoints = bgrid.reshape(-1, 2)
        bworld = (1000 * model_to_world(bpoints, embedding)[:, displayed_axes]).reshape(320, 320, 2)
        bx, by = bworld[..., 0], bworld[..., 1]
        B = barrier_values(barrier_model, bpoints).reshape(bgrid.shape[:2])
        fill = ax.contourf(bx, by, B, levels=[-np.inf, 0], colors=["#d4f5d4"], zorder=0)
        contour = ax.contour(bx, by, B, levels=[0], colors=["#1a8f1a"], linewidths=1, zorder=1)
        fill.set_clip_path(domain_patch)
        contour.set_clip_path(domain_patch)
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


def load_plot_run(run_dir, barrier_model=None, start_id=None):
    """Load the same verified inputs for a single trajectory or a batch overlay."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg, package, spec = saved_model_inputs(run_dir, manifest)
    # Older TCP manifests record identity in the effective configuration.
    manifest = {**{name: cfg.get(name) for name in ("model", "robot", "controller")}, **manifest}
    emb = manifest["embedding"]
    displayed_axes = world_axes(emb)
    arrays = read_trajectory(run_dir, manifest)
    measured_world, ideal_world = arrays["P"], arrays["P_ideal"]
    # Validate both arrays before drawing or combining several saved runs.
    task_plane_mm(measured_world, emb)
    task_plane_mm(ideal_world, emb)
    hashes, _ = recorded_model_hashes(run_dir, manifest)
    plot_inputs = {
        "world_axes": displayed_axes,
        "metadata": package["metadata"],
        "metadata_sha256": hashes[package["metadata"]],
    }
    if spec.get("Barrier", False):
        barrier_model = resolve_barrier_model(package, manifest, barrier_model, run_dir=run_dir)
        plot_inputs.update(
            barrier_model=str(barrier_model.resolve()), barrier_sha256=file_sha256(barrier_model)
        )
    return {
        "start_id": start_id or manifest.get("start_id", run_dir.name),
        "run_dir": str(run_dir.resolve()),
        "manifest": manifest,
        "cfg": cfg,
        "package": package,
        "spec": spec,
        "embedding": emb,
        "displayed_axes": displayed_axes,
        "measured_world": measured_world,
        "ideal_world": ideal_world,
        "barrier_model": barrier_model,
        "plot_inputs": plot_inputs,
    }


def draw_trajectories(runs, path, selection=None):
    """Draw one shared world frame, with a color per start and a line style per path type."""
    first = runs[0]
    emb, displayed_axes = first["embedding"], first["displayed_axes"]
    for run in runs[1:]:
        if (
            run["manifest"]["model"] != first["manifest"]["model"]
            or run["manifest"]["robot"] != first["manifest"]["robot"]
            or run["embedding"] != emb
            or run["spec"] != first["spec"]
            or experiment_setting(run["cfg"]) != experiment_setting(first["cfg"])
            or run["plot_inputs"] != first["plot_inputs"]
        ):
            raise ValueError("Overlay requires the same model, robot, embedding and plot inputs")
    goal = 1000 * model_to_world(np.zeros(2), emb)[displayed_axes]
    plot_inputs = {
        **first["plot_inputs"],
        "selection": selection,
        "runs": [{k: run[k] for k in ("start_id", "run_dir")} for run in runs],
    }

    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    try:
        regions = draw_regions(
            ax,
            first["spec"],
            emb,
            displayed_axes,
            [run[key] for run in runs for key in ("measured_world", "ideal_world")],
            first["barrier_model"],
        )
        overlay = len(runs) > 1
        handles = []
        if overlay:
            handles = [
                Line2D([], [], color="black", ls="--", lw=1.2, label="Ideal DS"),
                Line2D([], [], color="black", lw=1.8, label="Robot Path"),
            ]
        colors = plt.get_cmap("tab20" if len(runs) > 10 else "tab10")
        for i, run in enumerate(runs):
            color = colors(i % colors.N)
            ideal = 1000 * run["ideal_world"][:, displayed_axes]
            measured = 1000 * run["measured_world"][:, displayed_axes]
            ax.plot(
                *ideal.T,
                ls="--",
                color=color if overlay else "black",
                lw=1.2,
                label="_nolegend_" if overlay else "Ideal DS",
                zorder=3,
            )
            ax.plot(
                *measured.T,
                color=color if overlay else "tab:orange",
                lw=1.8,
                label="_nolegend_" if overlay else "Robot Path",
                zorder=4,
            )
            if overlay:
                handles.append(Line2D([], [], color=color, label=run["start_id"]))
        ax.plot(*goal, "*", color="black", ms=13, label="DS Goal", zorder=5)
        names = "xyz"
        ax.set(
            xlabel=rf"${names[displayed_axes[0]]}\,[\mathrm{{mm}}]$",
            ylabel=rf"${names[displayed_axes[1]]}\,[\mathrm{{mm}}]$",
            title=(
                f"{first['package']['shape']} / {first['manifest']['robot']} / "
                f"{experiment_setting(first['cfg'])}"
            ),
        )
        ax.set_aspect("equal", adjustable="box")
        # Keep all four domain edges visible rather than coincident with the axes spines.
        ax.use_sticky_edges = False
        ax.margins(0.04)
        ax.grid(alpha=0.3)
        path_handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles + path_handles + regions)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, metadata={"Description": json.dumps(plot_inputs)})
    finally:
        plt.close(fig)
    return path


def plot_trajectory(run_dir, barrier_model=None):
    """Write the existing single-run view with the model goal at x_n=0."""
    run = load_plot_run(run_dir, barrier_model)
    report = evaluate_run(run_dir, run["barrier_model"]) if run["spec"].get("Barrier") else None
    path = draw_trajectories([run], Path(run_dir) / "plots" / "trajectory.png")
    if report is not None:
        path.with_name("barrier.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print_barrier_events(report)
    return path


def post_pulse_ideal_rollout(run):
    """Plot-only DS continuation from the state after the last forced physics interval."""
    from sim.ds import ideal_rollout, load_scenario

    with np.load(Path(run["run_dir"]) / "traj.npz", allow_pickle=False) as tr:
        t = tr["t"]
        prefix = "physics_" if "physics_F_ext" in tr else ""
        force, times, positions = (tr[prefix + key] for key in ("F_ext", "t", "P"))
        active = np.flatnonzero(np.any(force[:-1] != 0, axis=1))
        if not len(active):
            raise ValueError("Post-pulse DS rollout requires a logged external force pulse")
        # Force row k acts over [times[k], times[k+1]); its end state is k+1.
        last = int(active[-1])
        end = last + 1
        end_s, point = float(times[end]), positions[end].copy()
        last_force_start_s = float(times[last])
        if np.any(force[end] != 0):
            raise ValueError("External force must be zero at the post-pulse state")

    scenario = load_scenario(run["cfg"])
    dt = float(run["cfg"]["sim"]["dt"])
    rate = scenario["rate"] * run["cfg"].get("ds_speed_factor", 1.0)
    # Check the integration against the original saved ideal path before re-anchoring.
    original = scenario["emb"].to_world(ideal_rollout(
        scenario["ds"], scenario["x0"], dt, len(t) - 1, rate
    ))
    if not np.array_equal(original, run["ideal_world"]):
        raise ValueError("DS integration does not reproduce the saved original ideal rollout")
    x0 = world_to_model(point[None], run["embedding"])[0]
    n_steps = int(np.floor((t[-1] - end_s) / dt + 1e-9))
    normalized = ideal_rollout(scenario["ds"], x0, dt, n_steps, rate)
    world = model_to_world(normalized, run["embedding"])
    axes = run["displayed_axes"]
    metadata = {
        "state_source": prefix + "P",
        "state_index": end,
        "start_time_s": end_s,
        "last_force_interval_start_s": last_force_start_s,
        "tcp_position_m": point.tolist(),
        "initial_model_position": x0.tolist(),
        "initial_world_position_m": world[0].tolist(),
        "start_xy_error_mm": float(1000 * np.max(np.abs(world[0, axes] - point[axes]))),
        "dt_s": dt,
        "rate_per_s": rate,
        "n_steps": n_steps,
        "original_rollout_reproduced": True,
    }
    return world, metadata


def plot_nominal_perturbed(nominal_dir, perturbed_dir, path, barrier_model, onset_s=None):
    """Compare full TCP paths, the original DS, and a plot-only post-pulse DS rollout."""
    nominal = load_plot_run(nominal_dir, barrier_model)
    perturbed = load_plot_run(perturbed_dir, barrier_model)
    for key in ("embedding", "spec", "plot_inputs", "displayed_axes"):
        if nominal[key] != perturbed[key]:
            raise ValueError(f"Paired plot inputs differ: {key}")
    for key in ("model", "robot", "start_id"):
        if nominal["manifest"][key] != perturbed["manifest"][key]:
            raise ValueError(f"Paired plot identity differs: {key}")
    if not np.array_equal(nominal["ideal_world"], perturbed["ideal_world"]):
        raise ValueError("Paired runs must share the original ideal DS rollout")
    axes, emb = nominal["displayed_axes"], nominal["embedding"]
    post_pulse, rollout_metadata = post_pulse_ideal_rollout(perturbed)
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    try:
        regions = draw_regions(ax, nominal["spec"], emb, axes,
                               [nominal["measured_world"], perturbed["measured_world"],
                                nominal["ideal_world"], post_pulse], barrier_model)
        ideal = 1000 * nominal["ideal_world"][:, axes]
        ax.plot(*ideal.T, color="black", lw=1.3, label="Ideal DS rollout", zorder=3)
        for run, label, color in (
            (nominal, "Nominal", "#e5b500"),
            (perturbed, "Perturbed", "#1769cc"),
        ):
            measured = 1000 * run["measured_world"][:, axes]
            ax.plot(*measured.T, color=color, lw=1.8, label=f"{label} TCP", zorder=4)
        ax.plot(*(1000 * post_pulse[:, axes]).T, color="#9131ac", ls="--", lw=1.5,
                label="Ideal DS Pertubation", zorder=5)
        goal = 1000 * model_to_world(np.zeros(2), emb)[axes]
        ax.plot(*goal, marker="*", color="black", ms=10, ls="none", label="Goal", zorder=5)
        if onset_s is not None:
            with np.load(Path(perturbed_dir) / "traj.npz") as tr:
                k = int(np.argmin(np.abs(tr["t"] - onset_s)))
            point = 1000 * perturbed["measured_world"][k, axes]
            ax.plot(*point, marker="x", color="black", ms=6, ls="none",
                    label="Disturbance applied", zorder=6)
        names = "xyz"
        ax.set(xlabel=f"{names[axes[0]]} [mm]", ylabel=f"{names[axes[1]]} [mm]",
               title=f"Five Obstacle / FR3 / PIC / {nominal['start_id']}")
        ax.set_aspect("equal", adjustable="box")
        ax.use_sticky_edges = False
        ax.margins(0.04)
        ax.grid(alpha=0.3)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles + regions, fontsize=8, loc="best")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, metadata={"Description": json.dumps({
            "nominal_run": str(nominal_dir), "perturbed_run": str(perturbed_dir),
            "nominal_states": len(nominal["measured_world"]),
            "perturbed_states": len(perturbed["measured_world"]),
            "plot_inputs": nominal["plot_inputs"], "onset_s": onset_s,
            "post_pulse_ideal_rollout": rollout_metadata,
        })})
    finally:
        plt.close(fig)
    return path


def print_barrier_events(report):
    """Show the first observed sample for each path and event, without merging them."""
    if report.get("status") != "evaluated":
        return
    for name, result in report["paths"].items():
        for label, key in (("B>0 inside X", "first_positive_inside_s"),
                           ("box exit", "first_box_exit_s")):
            time = result[key]
            detail = "none" if time is None else f"{report['start_id']} at t={time:.9g} s"
            print(f"{name}: first {label}: {detail}")


def select_batch_runs(batch_dir, start_ids=None, event=None):
    """Select only explicit IDs or true saved events; never interpret NA as false."""
    batch_dir = Path(batch_dir)
    if bool(start_ids) == bool(event):
        raise ValueError("Select batch runs with either --start-id ID (repeatable) or --event EVENT")
    with (batch_dir / "runs.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"start_id", "run_dir"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("Batch runs.csv requires start_id and run_dir")
        rows = list(reader)
    ids = [row["start_id"] for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Batch runs.csv requires unique, nonempty start IDs")
    if any(not row["run_dir"] or not row["run_dir"].strip() for row in rows):
        raise ValueError("Batch runs.csv requires nonempty run directories")
    if start_ids:
        missing = set(start_ids) - set(ids)
        if missing:
            raise ValueError(f"Unknown start IDs: {', '.join(sorted(missing))}")
        rows = [row for row in rows if row["start_id"] in set(start_ids)]
    else:
        checked = sum(event_value(row, event) is not None for row in rows)
        rows = [row for row in rows if event_value(row, event) is True]
        if not rows:
            raise ValueError(f"No runs match {event}; evaluated: {checked}/{len(ids)}; others: NA")
    return [
        (
            row["start_id"],
            Path(row["run_dir"])
            if Path(row["run_dir"]).is_absolute()
            else batch_dir / row["run_dir"],
        )
        for row in rows
    ]


def plot_batch(batch_dir, barrier_model=None, *, start_ids=None, event=None):
    """Overlay selected saved runs in the batch folder, without plotting other runs."""
    selected = select_batch_runs(batch_dir, start_ids, event)
    runs = [load_plot_run(directory, barrier_model, start_id) for start_id, directory in selected]
    name = f"trajectory_{event}.png" if event else "trajectory_selected.png"
    path = draw_trajectories(
        runs,
        Path(batch_dir) / "plots" / name,
        selection={"event": event, "start_ids": [start_id for start_id, _ in selected]},
    )
    reports = [evaluate_run(run["run_dir"], run["barrier_model"], run["start_id"])
               for run in runs if run["spec"].get("Barrier")]
    if reports:
        path.with_suffix(".barrier.json").write_text(
            json.dumps({"runs": reports}, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        for report in reports:
            print_barrier_events(report)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", type=Path, help="saved run or batch directory containing runs.csv"
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--start-id", action="append", help="batch start ID (repeatable)")
    selection.add_argument("--event", choices=EVENTS, help="select true saved events from runs.csv")
    parser.add_argument(
        "--barrier-model",
        type=Path,
        help="trusted local model_b.pth; default: beside the run's model_f.pth",
    )
    args = parser.parse_args(argv)
    try:
        if (args.run_dir / "manifest.json").is_file():
            if args.start_id or args.event:
                raise ValueError("--start-id and --event require a batch directory")
            path = plot_trajectory(args.run_dir, args.barrier_model)
        else:
            selected = select_batch_runs(args.run_dir, args.start_id, args.event)
            print("Selected starts: " + ", ".join(start_id for start_id, _ in selected))
            path = plot_batch(
                args.run_dir, args.barrier_model, start_ids=args.start_id, event=args.event
            )
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Plot error: {exc}\n")
    print(f"Trajectory plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
