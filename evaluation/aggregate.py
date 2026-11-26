"""Summarize saved runs in runs.csv and summary.md without simulating or plotting."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
from tempfile import mkdtemp
from zipfile import BadZipFile

import numpy as np
import yaml

from sim.robot import ROBOTS
from evaluation.metrics import evaluate_saved_run
from evaluation.run_artifacts import read_config, sha256

TRACKING_METRICS = (
    # Historical full-world, full-duration measurements retain their own names.
    "mse_sync_m2", "dtw_mean_m", "max_err_sync_m", "geom_dev_max_m",
    "mse_sync_plane_m2",
    "rmse_sync_plane_m",
    "max_err_sync_plane_m",
    "geom_dev_mean_plane_m",
    "geom_dev_max_plane_m",
    "chamfer_mean_plane_m",
    "hausdorff_plane_m",
    "frechet_plane_m",
    "dtw_mean_plane_m",
    "perp_rms_m",
    "perp_max_m",
)
METRICS = (
    "sigma_min_start",
    "sigma_min_run",
    "max_ori_err_rad",
    "final_ori_err_rad",
    "final_distance_m",
    "t_reach_s",
    "t_reach_ideal_s",
    "mse_window_end_s",
    "reach_delay_s",
    *TRACKING_METRICS,
    "t_max_err_sync_plane_s",
    "min_clearance_m",
    "domain_exit",
    "peak_applied_torque_ratio",
    "torque_saturation_steps",
)
EVENTS = (
    "goal_failure",
    "tracking_failure",
    "obstacle_entry",
    "domain_exit",
    "barrier_exit",
    "torque_saturation",
    "nonfinite",
)
FIELDS = (
    "start_id",
    "pair_id",
    "member",
    "initialization_status",
    "task_orientation",
    "sigma_rotation_length_m",
    "config_sha256",
    "run_id",
    "run_dir",
    "status",
    "model",
    "robot",
    "controller",
    "pulse",
    "metrics_status",
    "finite",
    *METRICS,
    "goal_failure",
    "goal_success",
    "goal_criterion",
    "goal_threshold_m",
    "goal_entry_status",
    "tracking_failure",
    "tracking_thresholds",
    "obstacle_entry",
    "barrier_exit",
    "barrier_exit_status",
    "sampled_barrier_exit",
    "peak_abs_qacc_rad_s2",
    "qacc_intervals",
    "qacc_status",
    "notes",
)


def event_value(row, event):
    """Read existing flags from typed rows or CSV; unavailable is never false."""
    if event not in EVENTS:
        raise ValueError(f"Unknown event: {event}")
    key = {"torque_saturation": "torque_saturation_steps", "nonfinite": "finite"}.get(
        event, event
    )
    value = row.get(key)
    if event == "torque_saturation":
        if isinstance(value, bool):
            return None
        try:
            steps = float(value)
        except (TypeError, ValueError):
            return None
        return steps > 0 if math.isfinite(steps) and steps >= 0 else None
    if isinstance(value, str):
        value = {"true": True, "false": False}.get(value.lower())
    if not isinstance(value, bool):
        return None
    return not value if event == "nonfinite" else value


def read_object(path, notes):
    """Keep missing or malformed artifacts visible instead of skipping their run."""
    if not path.is_file():
        notes.append(f"Missing {path.name}")
        return {}, "missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value, "available"
    except (OSError, ValueError) as exc:
        notes.append(f"Invalid {path.name}: {exc}")
        return {}, "invalid"


def load_evaluation(path):
    """Only explicit, unit-bearing evaluation thresholds may change task decisions."""
    if path is None:
        return {}
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or set(cfg) - {"goal_threshold_m", "tracking_thresholds"}:
        raise ValueError("Evaluation YAML accepts goal_threshold_m and tracking_thresholds only")
    thresholds = cfg.get("tracking_thresholds", {})
    if not isinstance(thresholds, dict) or set(thresholds) - set(TRACKING_METRICS):
        raise ValueError(f"Tracking thresholds must use saved metrics: {sorted(TRACKING_METRICS)}")
    values = list(thresholds.items())
    if "goal_threshold_m" in cfg:
        values.append(("goal_threshold_m", cfg["goal_threshold_m"]))
    for key, value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (key == "goal_threshold_m" and value == 0)
        ):
            raise ValueError(f"Invalid evaluation threshold {key}: {value!r}")
    return cfg


def run_entries(batch_dir, starts):
    """Select a single run first; otherwise use the batch plan or discover runs."""
    if (batch_dir / "manifest.json").is_file():
        return [(batch_dir.name, batch_dir)], False
    if starts is None and (batch_dir / "starts.csv").is_file():
        starts = batch_dir / "starts.csv"
    if starts is not None:
        with Path(starts).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not {"start_id", "run_dir"} <= set(reader.fieldnames or ()):
                raise ValueError("Start CSV requires start_id and run_dir columns")
            entries, ids, paths = [], set(), set()
            for number, item in enumerate(reader, 2):
                start_id, directory = item.get("start_id", ""), item.get("run_dir", "")
                if not start_id or not start_id.strip() or not directory or not directory.strip():
                    raise ValueError(f"Empty start_id or run_dir in start CSV line {number}")
                path = Path(directory)
                path = (path if path.is_absolute() else batch_dir / path).resolve()
                identity = (start_id, item.get("member"))
                if identity in ids or path in paths:
                    raise ValueError(f"Duplicate start_id or run_dir in start CSV line {number}")
                ids.add(identity)
                paths.add(path)
                entries.append((start_id, path))
            return entries, True
    # Include damaged runs even when their manifest alone is absent.
    names = {"manifest.json", "metrics.json", "config_resolved.yaml", "traj.npz", "error.txt"}
    paths = sorted({p.parent for p in batch_dir.rglob("*") if p.is_file() and p.name in names})
    return [(str(p.relative_to(batch_dir)) if p != batch_dir else p.name, p) for p in paths], False


def scalar(metrics, key, notes, boolean=False):
    value = metrics.get(key)
    if value is None:
        return None
    valid = (
        isinstance(value, bool)
        if boolean
        else (
            not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        )
    )
    if not valid:
        notes.append(f"Invalid metric {key}")
        return None
    return value


def raw_diagnostics(directory, manifest, override, notes):
    """Read only logged acceleration intervals and, if requested, goal-entry samples."""
    result = {"qacc_status": "not_logged"}
    path = directory / "traj.npz"
    if not path.is_file():
        result["qacc_status"] = "missing_trajectory"
        return result
    try:
        with np.load(path, allow_pickle=False) as tr:
            t = tr["t"]
            if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
                raise ValueError("expected at least two finite, increasing timestamps")
            n = len(t) - 1
            if "qacc" in tr:
                qacc = tr["qacc"]
                if (
                    qacc.ndim != 2
                    or qacc.shape[0] not in (n, n + 1)
                    or qacc.shape[1] == 0
                    or not np.isfinite(qacc[:n]).all()
                ):
                    raise ValueError("qacc must contain N or N+1 finite joint-acceleration rows")
                # PIC logs N executed intervals followed by a terminal padding row.
                result.update(
                    peak_abs_qacc_rad_s2=float(np.abs(qacc[:n]).max()),
                    qacc_intervals=n,
                    qacc_status="available",
                )
            if override is not None:
                if "dist_goal" in tr:
                    distances = tr["dist_goal"]
                else:
                    emb = manifest["embedding"]
                    goal = np.asarray(emb["origin"]) - emb["L_m"] * (
                        np.asarray(emb["axes"]) @ np.asarray(emb["anchor_n"])
                    )
                    P = tr["P"]
                    if P.shape != (len(t), 3):
                        raise ValueError("P must contain one world position per timestamp")
                    distances = np.linalg.norm(P - goal, axis=1)
                if distances.shape != t.shape or not np.isfinite(distances).all():
                    raise ValueError("goal distances must be finite and aligned with t")
                entered = np.flatnonzero(distances < override)
                result.update(
                    t_reach_s=float(t[entered[0]]) if len(entered) else None,
                    goal_entry_status="entered" if len(entered) else "not_entered",
                )
    except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile) as exc:
        notes.append(f"Invalid trajectory diagnostics: {exc}")
        # Do not discard an acceleration result if only goal reconstruction failed.
        if result["qacc_status"] != "available":
            result["qacc_status"] = "invalid"
    return result


def summarize_run(start_id, directory, evaluation, *, recompute=False, window_radius_m=None):
    notes = []
    row = dict.fromkeys(FIELDS)
    row.update(
        start_id=start_id,
        run_id=str(directory),
        run_dir=str(directory),
        status="missing",
        goal_criterion="saved_success",
        goal_entry_status="unavailable",
        metrics_status="missing",
        barrier_exit_status="not_evaluated",
        qacc_status="missing_trajectory",
    )
    thresholds = evaluation.get("tracking_thresholds", {})
    row["tracking_thresholds"] = json.dumps(thresholds, sort_keys=True) if thresholds else None
    override = evaluation.get("goal_threshold_m")
    if override is not None:
        row.update(goal_criterion="final_distance_m < goal_threshold_m", goal_threshold_m=override)
    if not directory.is_dir():
        row["notes"] = "Run directory is missing"
        return row
    manifest, manifest_state = read_object(directory / "manifest.json", notes)
    if manifest.get("error"):
        notes.append(f"{manifest.get('failure_stage', 'run')}: {manifest['error']}")
    metrics, row["metrics_status"] = read_object(directory / "metrics.json", notes)
    metrics_hash = manifest.get("artifact_sha256", {}).get("metrics.json")
    if row["metrics_status"] == "available" and metrics_hash is not None:
        if sha256(directory / "metrics.json") != metrics_hash:
            metrics = {}
            row["metrics_status"] = "invalid"
            notes.append("Saved metrics hash does not match the run")
    status = manifest.get("status")
    row["status"] = (
        status
        if isinstance(status, str)
        else ("missing_status" if manifest_state == "available" else f"{manifest_state}_manifest")
    )
    if status == "complete" and (recompute or row["metrics_status"] == "missing") and (
        "evaluation_schema" in manifest or "trajectory_sha256" in manifest or "artifact_sha256" in manifest
    ):
        try:
            metrics = evaluate_saved_run(directory, manifest, window_radius_m=window_radius_m)
            row["metrics_status"] = "recomputed"
        except (OSError, KeyError, ValueError, TypeError, EOFError, BadZipFile) as exc:
            metrics = {}
            row["metrics_status"] = "invalid"
            notes.append(f"Cannot evaluate saved run: {exc}")
    row["start_id"] = manifest.get("start_id", start_id)
    for key in ("pair_id", "member", "initialization_status", "task_orientation",
                "sigma_rotation_length_m", "config_sha256"):
        row[key] = manifest.get(key)
    saved_config = manifest.get("config", {})
    for key in ("model", "robot", "controller"):
        row[key] = manifest.get(key, saved_config.get(key))
    pulse = manifest.get("disturbance", saved_config.get("disturbance"))
    if pulse is not None:
        row["pulse"] = json.dumps(pulse, sort_keys=True)
    for key in METRICS:
        row[key] = scalar(metrics, key, notes, boolean=(key == "domain_exit"))
    if row["sigma_min_start"] is None:
        row["sigma_min_start"] = scalar(manifest, "sigma_min_start", notes)
    row["finite"] = scalar(metrics, "finite", notes, boolean=True)
    row["barrier_exit"] = scalar(metrics, "barrier_exit", notes, boolean=True)
    if row["barrier_exit"] is not None:
        row["barrier_exit_status"] = "evaluated"

    if override is not None:
        row["t_reach_s"] = None
    else:
        # The saved success flag remains authoritative; this only records its radius.
        path = directory / "config_resolved.yaml"
        if (path.is_file() or saved_config) and "embedding" in manifest:
            try:
                cfg = read_config(directory, manifest)
                radius = cfg["sim"]["success_frac"] * manifest["embedding"]["L_m"]
                if not isinstance(radius, bool) and math.isfinite(radius) and radius > 0:
                    row["goal_threshold_m"] = radius
            except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
                notes.append(f"Cannot read saved goal radius: {exc}")
    raw = raw_diagnostics(directory, manifest, override, notes)
    row.update(raw)
    if row["status"] == "complete" and row["finite"] is not False:
        if override is not None:
            if row["final_distance_m"] is not None:
                row["goal_failure"] = not row["final_distance_m"] < override
            reached, ideal_reached = row["t_reach_s"], row["t_reach_ideal_s"]
            row["reach_delay_s"] = reached - ideal_reached if None not in (reached, ideal_reached) else None
        else:
            success = scalar(metrics, "success", notes, boolean=True)
            row["goal_failure"] = None if success is None else not success
            if "t_reach_s" in metrics:
                row["goal_entry_status"] = (
                    "not_entered"
                    if metrics["t_reach_s"] is None
                    else ("entered" if row["t_reach_s"] is not None else "unavailable")
                )
        if thresholds:
            comparisons = [
                row[key] > limit if row[key] is not None else None
                for key, limit in thresholds.items()
            ]
            row["tracking_failure"] = (
                True
                if True in comparisons
                else (False if all(value is False for value in comparisons) else None)
            )
        if row["min_clearance_m"] is not None:
            # The source unsafe-level convention includes its boundary (level <= 0).
            row["obstacle_entry"] = row["min_clearance_m"] <= 0
    else:
        notes.append("Task decisions unavailable: run is not complete or has nonfinite states")
        row.update(t_reach_s=None, reach_delay_s=None, goal_entry_status="unavailable")
        if row["qacc_status"] == "available":
            row["qacc_status"] = "available_partial"
    if not thresholds:
        notes.append("Tracking failure not evaluated: no threshold supplied")
    row["goal_success"] = None if row["goal_failure"] is None else not row["goal_failure"]
    row["notes"] = "; ".join(notes) or None
    return row


def finite_pairs(rows, key):
    """Use only available finite numeric observations; zero remains a valid value."""
    return [
        (row["start_id"], row[key])
        for row in rows
        if not isinstance(row.get(key), bool)
        and isinstance(row.get(key), (int, float))
        and math.isfinite(row[key])
    ]


def metric_statistics(rows, key):
    """Describe saved per-run values, using sample SD and linear-interpolated quartiles."""
    pairs = finite_pairs(rows, key)
    result = dict.fromkeys(("mean", "sd", "median", "q1", "q3", "minimum", "maximum"))
    result.update(n=len(pairs), min_ids=[], max_ids=[])
    if pairs:
        values = np.array([value for _, value in pairs])
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
        maximum = float(values.max())
        minimum = float(values.min())
        result.update(
            mean=float(values.mean()),
            sd=float(values.std(ddof=1)) if len(values) > 1 else None,
            median=float(median),
            q1=float(q1),
            q3=float(q3),
            maximum=maximum,
            minimum=minimum,
            min_ids=[start_id for start_id, value in pairs if value == minimum],
            max_ids=[start_id for start_id, value in pairs if value == maximum],
        )
    return result


def observed_torque_peak(directory):
    """Read applied joint torque in N m, excluding the source terminal padding row."""
    try:
        with np.load(Path(directory) / "traj.npz", allow_pickle=False) as tr:
            t = tr["t"]
            n = len(t) - 1
            torque = tr["tau_applied"]
            if (
                t.ndim != 1
                or n < 1
                or not np.isfinite(t).all()
                or np.any(np.diff(t) <= 0)
                or torque.ndim != 2
                or torque.shape[0] not in (n, n + 1)
                or torque.shape[1] == 0
                or not np.isfinite(torque[:n]).all()
            ):
                return None
            return float(np.abs(torque[:n]).max())
    except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile):
        return None


def write_summary(path, rows, barrier_reports=None, *, window_radius_mm=None):
    """Render compact millimeter statistics and keep checked and unavailable events distinct."""
    def number(value):
        return "NA" if value is None else f"{value:.9g}"

    def ids(values, checked=True):
        return ", ".join(f"`{value}`" for value in values) or ("keine" if checked else "NA")

    tracking = (("mse_sync_plane_m2", "Windowed task-space MSE", "mm²", 1e6),
                ("dtw_mean_plane_m", "Task-space DTW", "mm", 1e3),
                ("geom_dev_max_plane_m", "Max task-space geometric deviation", "mm", 1e3))
    goal = (("final_distance_m", "Final goal distance", "mm", 1e3),
            ("t_reach_s", "Goal entry time", "s", 1),
            ("reach_delay_s", "Reach delay", "s", 1))
    radius = "`success_frac * L`" if window_radius_mm is None else f"{window_radius_mm:g} mm"
    statuses = Counter(row["status"] for row in rows)
    lines = [
        "# Run summary", "",
        f"Rows: {len(rows)}. Technical status: "
        + ", ".join(f"{status}={count}" for status, count in sorted(statuses.items())) + ".",
        f"MSE, RMSE and perpendicular RMS use samples from time zero through the first ideal "
        f"sample inside the goal radius ({radius}), or through the final sample when there is no entry.",
        "Tracking n counts complete runs with finite=true and all three tracking metrics available. "
        "Goal statistics use available values from complete runs with finite=true. "
        "Reached n counts runs with a recorded goal entry time. "
        "Values are mean ± sample SD (SD=NA for n<2).", "",
        "| Model | Robot | Controller | Tracking n | Windowed task-space MSE [mm²] | Task-space DTW [mm] | "
        "Max task-space geometric deviation [mm] | Final goal distance [mm] | Reached n | Goal entry time [s] | Reach delay [s] |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    def group_for(row):
        label = row.get("controller") or "NA"
        if row.get("member"):
            label += " / " + row["member"]
        return row.get("model") or "NA", row.get("robot") or "NA", label

    groups = sorted({group_for(row) for row in rows if row.get("model")}) or [("NA", "NA", "NA")]
    extrema = []
    for group in groups:
        complete = [row for row in rows
                    if group_for(row) == group
                    and row["status"] == "complete" and row.get("finite") is True]
        valid = [row for row in complete if all(finite_pairs([row], key) for key, _, _, _ in tracking)]
        values = [str(len(valid))]
        for metric in (*tracking, *goal):
            key, label, unit, scale = metric
            stats = metric_statistics(valid if metric in tracking else complete, key)
            mean = None if stats["mean"] is None else scale * stats["mean"]
            sd = None if stats["sd"] is None else scale * stats["sd"]
            if key == "t_reach_s":
                values.append(str(stats["n"]))
            if mean is None:
                values.append("NA")
            elif sd is None:
                values.append(f"{mean:.3f} ± NA")
            else:
                values.append(f"{mean:.3f} ± {sd:.3f}")
            extrema.append((group, label, unit, scale, stats))
        lines.append("| " + " | ".join((*group, *values)) + " |")
    lines += ["", "| Metric | Unit | Minimum | Start ID(s) | Maximum | Start ID(s) |",
              "| --- | --- | ---: | --- | ---: | --- |"]
    for group, label, unit, scale, stats in extrema:
        if len(groups) > 1:
            label = " / ".join(group) + ": " + label
        minimum = None if stats["minimum"] is None else scale * stats["minimum"]
        maximum = None if stats["maximum"] is None else scale * stats["maximum"]
        lines.append(f"| {label} | {unit} | {number(minimum)} | "
                     f"{ids(stats['min_ids'], bool(stats['n']))} | {number(maximum)} | "
                     f"{ids(stats['max_ids'], bool(stats['n']))} |")

    legacy = [r for r in rows if r.get("mse_sync_m2") is not None and r.get("mse_sync_plane_m2") is None]
    if legacy:
        lines += ["", "Historical full-duration world-coordinate tracking (saved metric names)", "",
                  "| Model / robot / controller | MSE [mm²] | DTW [mm] |",
                  "| --- | ---: | ---: |"]
        for group in sorted({group_for(r) for r in legacy}):
            complete = [r for r in legacy if group_for(r) == group and r["status"] == "complete" and r["finite"]]
            values = []
            for key, factor in (("mse_sync_m2", 1e6), ("dtw_mean_m", 1e3)):
                mean = metric_statistics(complete, key)["mean"]
                values.append(number(None if mean is None else mean * factor))
            lines.append("| " + " | ".join([" / ".join(group), *values]) + " |")

    torque_rows = [dict(row, torque_peak=observed_torque_peak(row["run_dir"])) for row in rows]
    lines += ["", "| Observed peak | Unit | n | Maximum | Start ID(s) |",
              "| --- | --- | ---: | ---: | --- |"]
    for source, key, unit, scale in (
        (torque_rows, "torque_peak", "N·m", 1),
        (rows, "peak_applied_torque_ratio", "% of joint limit", 100),
        (rows, "peak_abs_qacc_rad_s2", "rad/s²", 1),
    ):
        pairs = finite_pairs(source, key)
        maximum = max((value for _, value in pairs), default=None)
        peak_ids = [start_id for start_id, value in pairs if value == maximum]
        label = "peak_abs_applied_torque_nm" if key == "torque_peak" else key
        value = None if maximum is None else scale * maximum
        lines.append(f"| `{label}` | {unit} | {len(pairs)} | {number(value)} | "
                     f"{ids(peak_ids, bool(pairs))} |")
    lines += ["", "Applied torque ratio is max over joints and executed intervals of "
              "|tau_applied,j| / tau_limit,j. This observed ratio is separate from commanded "
              "torque saturation. Torque and acceleration peaks exclude terminal padding; "
              "available partial-run peaks remain observations."]

    goal_criteria = sorted({
        f"{row['goal_criterion']}; radius={number(row.get('goal_threshold_m'))} m"
        for row in rows if row.get("goal_criterion")
    })
    tracking = sorted({row["tracking_thresholds"] for row in rows if row.get("tracking_thresholds")})
    lines += ["", "Goal criterion: " + ("; ".join(goal_criteria) or "NA") + ".",
              "Explicit tracking thresholds: "
              + ("; ".join(f"`{v}`" for v in tracking) if tracking else "NA (not configured)")
              + ".", "",
              "| Event | Count | Evaluated denominator | NA | Affected start IDs |",
              "| --- | ---: | ---: | ---: | --- |"]
    for event in EVENTS:
        if event == "barrier_exit" and all(event_value(row, event) is None for row in rows):
            continue  # With no saved flag, use the explicit ideal/TCP sampled checks below.
        checks = [(row["start_id"], event_value(row, event)) for row in rows]
        evaluated = sum(value is not None for _, value in checks)
        affected = [start_id for start_id, value in checks if value is True]
        count = str(len(affected)) if evaluated else "NA"
        lines.append(f"| `{event}` | {count} | {evaluated} | {len(rows) - evaluated} | "
                     f"{ids(affected, bool(evaluated))} |")
    # The FR3 reference is not a generic arm limit. UR5e peaks stay observable
    # in the peak table above, but have no acceleration-reference verdict.
    fr3_rows = [row for row in rows if row.get("robot") in ROBOTS
                and ROBOTS[row["robot"]].acceleration_reference_rad_s2 == 10.0]
    accelerations = finite_pairs(fr3_rows, "peak_abs_qacc_rad_s2")
    exceeded = [start_id for start_id, value in accelerations if value > 10]
    lines += [f"| FR3 acceleration reference exceeded in simulation (>10 rad/s²) | "
              f"{len(exceeded) if accelerations else 'NA'} | {len(accelerations)} | "
              f"{len(rows) - len(accelerations)} | {ids(exceeded, bool(accelerations))} |", "",
              "Obstacle entry: clearance ≤ 0. Box exit: saved domain_exit. "
              "Torque saturation: saved torque_saturation_steps > 0. "
              "The 10 rad/s² acceleration reference is a simulation diagnostic, "
              "not a hardware safety verdict; it applies only to the FR3 family. "
              "UR5e and unknown robots have no configured acceleration reference (NA).", "",
              "| Trajectory | Event | Runs | Evaluated | NA | Samples | Affected start IDs | First sample |",
              "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |"]
    evaluated = [report for report in (barrier_reports or []) if report["status"] == "evaluated"]
    for name, label in (("ideal", "Ideal DS"), ("tcp", "TCP")):
        for event, count_key, time_key in (
            ("B>0 inside X", "positive_samples_inside", "first_positive_inside_s"),
            ("Box exit", "box_exit_samples", "first_box_exit_s"),
        ):
            affected = [report["start_id"] for report in evaluated
                        if report["paths"][name][count_key] > 0]
            samples = sum(report["paths"][name][count_key] for report in evaluated)
            firsts = sorted((report["paths"][name][time_key], report["start_id"])
                            for report in evaluated if report["paths"][name][time_key] is not None)
            first = (f"`{firsts[0][1]}` at {number(firsts[0][0])} s" if firsts
                     else ("keine" if evaluated else "NA"))
            lines.append(f"| {label} | {event} | {len(affected) if evaluated else 'NA'} | "
                         f"{len(evaluated)} | {len(rows) - len(evaluated)} | "
                         f"{samples if evaluated else 'NA'} | {ids(affected, bool(evaluated))} | {first} |")
    barrier_statuses = Counter(report["status"] for report in (barrier_reports or []))
    lines += ["", "X is the closed box domain.range. B>0 is strict and uses float32 model "
              "evaluation. Samples outside X are counted only as box exits. First sample "
              "means the minimum logged time across runs (ties use start-ID order); "
              "no crossing-time interpolation or continuous-time safety claim is made. "
              "Per-run counts, first times and input hashes are in barrier.json.",
              "Barrier evaluation status: " + (", ".join(f"{k}={v}" for k, v in sorted(barrier_statuses.items()))
              or "NA") + "."]
    posture = [r for r in rows if r.get("member") in ("nominal", "lower_sigma") and r.get("pair_id")]
    if posture:
        lines += ["", "### Paired posture differences", "",
                  "Differences are lower-sigma minus nominal for complete pairs with matching run settings.", "",
                  "| Pair | Robot | Orientation | Rotation scale [m/rad] | Δ start σ | Δ minimum σ | "
                  "Δ MSE [mm²] | Δ DTW [mm] | Goal success (nominal/lower) |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        comparisons = {}
        for r in posture:
            key = (r["pair_id"], r["config_sha256"], r["robot"], r["task_orientation"], r["sigma_rotation_length_m"])
            comparisons.setdefault(key, []).append(r)
        for (pair, _, robot, task, scale), members in comparisons.items():
            complete = {r["member"]: r for r in members if r["status"] == "complete" and r["finite"]}
            values = ["NA"] * 5
            if len(members) == len(complete) == 2:
                a, b = complete["nominal"], complete["lower_sigma"]
                keys = (("sigma_min_start", 1), ("sigma_min_run", 1),
                        ("mse_sync_plane_m2", 1e6), ("dtw_mean_plane_m", 1e3))
                values = [number((b[k] - a[k]) * factor) if None not in (a[k], b[k]) else "NA"
                          for k, factor in keys]
                values.append(f"{a['goal_success']}/{b['goal_success']}")
            lines.append("| " + " | ".join([pair, robot or "NA", task or "NA", number(scale), *values]) + " |")
    technical = [row["start_id"] for row in rows if row["status"] != "complete"]
    lines += ["", "Runs with technical status other than complete: " + ids(technical) + ".", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def aggregate(batch_dir, *, starts=None, evaluation=None, output=None, barrier_model=None,
              recompute=False, window_radius_mm=None):
    batch_dir = Path(batch_dir).resolve()
    if not batch_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {batch_dir}")
    window_radius_m = None
    if window_radius_mm is not None:
        if not recompute:
            raise ValueError("--window-radius-mm requires --recompute")
        window_radius_m = window_radius_mm / 1000
        if isinstance(window_radius_mm, bool) or not math.isfinite(window_radius_m) or window_radius_m <= 0:
            raise ValueError("--window-radius-mm must be finite and positive")
        if output is not None:
            parent = Path(output).resolve().parent
            if parent.is_relative_to(batch_dir) or parent.exists():
                raise ValueError("--window-radius-mm requires a new output directory outside the input directory")
    cfg = load_evaluation(evaluation)
    if barrier_model is not None and not Path(barrier_model).is_file():
        raise ValueError(f"Barrier model does not exist: {barrier_model}")
    entries, planned = run_entries(batch_dir, starts)
    study_runs = batch_dir / "study_runs.csv"
    planned_study = {}
    if starts is None and not planned and study_runs.is_file():
        with study_runs.open(newline="", encoding="utf-8") as handle:
            planned_study = {(batch_dir / row["run_dir"]).resolve(): row for row in csv.DictReader(handle)}
        present = {path for _, path in entries}
        entries.extend((row["start_id"], path) for path, row in planned_study.items() if path not in present)
        planned = True
    elif planned:
        with Path(starts or batch_dir / "starts.csv").open(newline="", encoding="utf-8") as handle:
            planned_study = {(batch_dir / row["run_dir"]).resolve(): row for row in csv.DictReader(handle)}
    rows = [summarize_run(start_id, directory, cfg, recompute=recompute, window_radius_m=window_radius_m)
            for start_id, directory in entries]
    batch_manifest = batch_dir / "batch.json"
    batch_settings, _ = read_object(batch_manifest, []) if planned and batch_manifest.is_file() else ({}, "missing")
    for row in rows:
        plan = {**batch_settings, **planned_study.get(Path(row["run_dir"]), {})}
        if plan and row["status"] == "missing":
            for name in ("start_id", "robot", "model", "controller", "pulse", "pair_id", "member"):
                if plan.get(name) not in (None, "", "NA"):
                    row[name] = plan[name]
            row["notes"] = "Planned run has not been executed"
    if output is None:
        path = (Path(mkdtemp(prefix="plane-metrics-")) if recompute else batch_dir) / "runs.csv"
    else:
        path = Path(output)
    if path.resolve().suffix != ".csv" or path.resolve().name in {"starts.csv", "study_runs.csv", "planned_starts.csv"}:
        raise ValueError("Derived run tables must be written to a .csv file")
    path.parent.mkdir(parents=True, exist_ok=window_radius_mm is None or output is None)
    reports = []
    for row in rows:
        report = {"start_id": row["start_id"], "status": "not_evaluated"}
        if barrier_model is not None:
            from evaluation.barrier import evaluate_run

            try:
                report = evaluate_run(row["run_dir"], barrier_model, row["start_id"])
            except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, BadZipFile) as exc:
                report.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
        if row.get("member"):
            report.update(member=row["member"], pair_id=row.get("pair_id"))
        reports.append(report)
        if report["status"] == "evaluated":
            row["sampled_barrier_exit"] = report["paths"]["tcp"]["positive_samples_inside"] > 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        "NA"
                        if value is None
                        else (str(value).lower() if isinstance(value, bool) else value)
                    )
                    for key, value in row.items()
                }
            )
    path.with_name("barrier.json").write_text(
        json.dumps({"runs": reports}, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_summary(path.with_name("summary.md"), rows, reports, window_radius_mm=window_radius_mm)
    return path, rows, planned


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path, help="saved run directory or batch directory")
    parser.add_argument(
        "--starts",
        type=Path,
        help="CSV with unique start_id,run_dir; default: batch_dir/starts.csv if present",
    )
    parser.add_argument(
        "--evaluation", type=Path, help="YAML with explicit goal/tracking thresholds"
    )
    parser.add_argument(
        "--output", type=Path,
        help="output CSV; defaults to batch_dir/runs.csv or a new temporary directory with --recompute",
    )
    parser.add_argument(
        "--barrier-model", type=Path,
        help="trusted local model_b.pth for separate ideal/TCP sampled checks; default: NA",
    )
    parser.add_argument("--recompute", action="store_true", help="calculate metrics from verified raw trajectories")
    parser.add_argument("--window-radius-mm", type=float,
                        help="ideal goal radius for the MSE window; requires --recompute and a new output directory")
    args = parser.parse_args(argv)
    try:
        path, rows, planned = aggregate(
            args.batch_dir, starts=args.starts, evaluation=args.evaluation, output=args.output,
            barrier_model=args.barrier_model, recompute=args.recompute, window_radius_mm=args.window_radius_mm,
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(2, f"Aggregation error: {exc}\n")
    print(f"Saved table: {path.resolve()}")
    print(f"Saved summary: {path.with_name('summary.md').resolve()}")
    print(f"Saved Barrier checks: {path.with_name('barrier.json').resolve()}")
    source = (
        "single run"
        if (args.batch_dir / "manifest.json").is_file()
        else ("planned starts" if planned else "discovered runs")
    )
    print(f"Rows: {len(rows)}; source: {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
