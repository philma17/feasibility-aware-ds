"""Resolve experiment inputs, run the selected controller, and record its output."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import yaml

from evaluation.metrics import evaluate_saved_run
from evaluation.run_artifacts import sha256, write_json
from sim.ds import ROOT, demo_files, input_path, model_catalog
from sim.robot import MENAGERIE_DIR, ROBOTS
from sim.sim import build_case, replay_config
from sim.starts import StartUnavailable
from sim.controllers import pic, servo


CONTROLLERS = {"pic": (pic, pic.validate_config), "servo": (servo, None)}


def select_controller(cfg):
    """Return the selected controller module and its optional configuration validator."""
    try:
        return CONTROLLERS[cfg["controller"]]
    except KeyError as exc:
        raise ValueError(f"controller must be one of {', '.join(CONTROLLERS)}") from exc


def load_config(path, overrides=()):
    with Path(path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Experiment configuration must be a mapping")
    allowed = {
        "model",
        "model_catalog",
        "controller",
        "robot",
        "sim",
        "task",
        "plant",
        "pic",
        "disturbance",
        "spatial",
        "multistart",
        "pairing",
        "model_source",
        "time_rate",
        "ds_speed_factor",
    }
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, received {item!r}")
        key, text = item.split("=", 1)
        parts = key.split(".")
        current = cfg
        for part in parts[:-1]:
            if not isinstance(current, dict) or part not in current:
                raise ValueError(f"Unknown configuration key: {key}")
            current = current[part]
        if not isinstance(current, dict) or parts[-1] not in current:
            raise ValueError(f"Unknown configuration key: {key}")
        current[parts[-1]] = yaml.safe_load(text)
    _, validate = select_controller(cfg)
    if validate is not None:
        validate(cfg)
    for key in ("dt", "T_max"):
        value = cfg["sim"][key]
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"sim.{key} must be finite and positive")
    steps = cfg["sim"]["T_max"] / cfg["sim"]["dt"]
    if not np.isfinite(steps) or steps < 1 or not np.isclose(steps, round(steps)):
        raise ValueError("T_max must be a positive integer multiple of dt")
    if not isinstance(cfg["disturbance"]["enabled"], bool):
        raise ValueError("disturbance.enabled must be a boolean")
    # Keep legacy clear_every and pic.torque_limit.source in the YAML for provenance.
    return cfg


def input_hashes(cfg):
    package = model_catalog(cfg["model_catalog"])[cfg["model"]]
    if not package["selectable"]:
        raise ValueError(package["reason"])
    files = [
        package["checkpoint"],
        package["metadata"],
        *demo_files(package),
        cfg["model_catalog"],
        "asset_manifest.json",
        "vendor/S2NNDS/NNModels.py",
    ]
    spec = ROBOTS[cfg["robot"]]
    asset_dir = Path(MENAGERIE_DIR) / Path(spec.mjcf).parent
    files += [
        str(p.relative_to(ROOT))
        for p in sorted(asset_dir.rglob("*"))
        if p.is_file()
    ]
    expected = {
        a["path"]: a["sha256"]
        for a in json.loads((ROOT / "asset_manifest.json").read_text())["assets"]
    }
    hashes = {str(path): sha256(input_path(path)) for path in files}
    for path, digest in hashes.items():
        if path in expected and digest != expected[path]:
            raise ValueError(f"Asset hash mismatch: {path}")
    return hashes


def git_state():
    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": git("rev-parse", "HEAD"),
        "status": git("status", "--porcelain=v1", "--untracked-files=all"),
    }


def run_single(cfg, output, *, hashes=None, command_line=None, start_id=None,
               study=None, compute_tracking=True, start_row=None):
    """Build and simulate one experiment, then record raw and optional derived outputs."""
    output = Path(output)
    hashes = input_hashes(cfg) if hashes is None else hashes
    controller, _ = select_controller(cfg)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Output directory already exists: {output}", file=sys.stderr)
        return 2
    ignored = []
    if "clear_every" in cfg["sim"]:
        ignored.append("sim.clear_every")
    pic_config = cfg.get("pic")
    torque_config = pic_config.get("torque_limit") if isinstance(pic_config, dict) else None
    if isinstance(torque_config, dict) and "source" in torque_config:
        ignored.append("pic.torque_limit.source")
    if start_row is not None:
        start_id = start_row["start_id"] if start_id is None else start_id
        cfg = replay_config(cfg, start_row)
    start = cfg["task"].get("start")
    sampling_seed = ((cfg.get("multistart") or {}).get("seed", 0)
                     if isinstance(start, dict) and "index" in start else None)
    manifest = {
        "status": "incomplete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution": "dynamic MuJoCo simulation",
        "controller": cfg["controller"],
        "model": cfg["model"],
        "robot": cfg["robot"],
        "seed": cfg["sim"]["seed"],
        "seeds": {"ik": cfg["sim"]["seed"], "start_sampling": sampling_seed},
        "disturbance": cfg["disturbance"],
        "ignored_config_keys": ignored,
        "evaluation_schema": 2,
        "task_orientation": cfg["task"].get("orientation", "none"),
        "sigma_rotation_length_m": cfg["sim"].get("sigma_rotation_length_m", 1.0),
        "git": git_state(),
        "command_line": list(sys.argv if command_line is None else command_line),
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in ("numpy", "scipy", "mujoco", "torch", "pandas", "PyYAML", "pyLasaDataset")
        },
        "input_sha256": hashes,
        "implementation_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for folder in (ROOT / "sim", ROOT / "evaluation")
            for p in sorted(folder.rglob("*.py"))
        },
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
    }
    package = model_catalog(cfg["model_catalog"])[cfg["model"]]
    barrier = input_path(package["checkpoint"]).with_name("model_b.pth")
    if barrier.is_file():
        manifest["barrier_sha256"] = sha256(barrier)
    if study is not None:
        manifest["study"] = study
        manifest["seeds"]["start_sampling"] = study.get("sampling_seed")
    if start_id is not None:
        manifest["start_id"] = start_id
    for filename in ("main.py", "experiment.py"):
        manifest["implementation_sha256"][filename] = sha256(ROOT / filename)
    # Write provenance before IK so failed starts still leave an auditable record.
    write_json(output / "manifest.json", manifest)
    config_path = output / "config_resolved.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    manifest["artifact_sha256"] = {config_path.name: sha256(config_path)}
    started = time.perf_counter()
    stage = "start_ik"
    case = None

    def save_start(row):
        manifest["start_id"] = row["start_id"]
        manifest["seeds"]["ik"] = row["ik_seed"]
        row.setdefault("start_seed", sampling_seed)
        manifest["seeds"]["start_sampling"] = row["start_seed"]
        for key in ("member", "pair_id"):
            if row.get(key):
                manifest[key] = row[key]
        manifest["initialization_status"] = row["status"]
        manifest["sigma_min_start"] = row.get("sigma_min_start")
        write_json(output / "starts.json", [row])
        manifest["artifact_sha256"]["starts.json"] = sha256(output / "starts.json")

    try:
        if start_row is None and cfg["sim"].get("start_criterion") != "closest_to_reference":
            case = build_case(cfg)
        else:
            case = build_case(cfg, start_id=start_id, start_row=start_row)
            row = case.starts.record(case.q_start) if start_row is None else dict(start_row)
            save_start(row)
        manifest.update(
            robot_model={
                "mjcf": str(Path("vendor") / case.robot.spec.mjcf),
                "joint_names": list(case.robot.spec.joints),
                "fixed_joints_rad": dict(case.robot.spec.fixed_joints),
                "tcp_site": case.robot.spec.site,
                "arm_dof": case.robot.n,
                "q_min_rad": case.robot.q_min.tolist(),
                "q_max_rad": case.robot.q_max.tolist(),
                "qd_max_rad_s": case.robot.qd_max.tolist(),
                "acceleration_reference_rad_s2": case.robot.spec.acceleration_reference_rad_s2,
            },
            q_start_rad=case.q_start.tolist(),
            evaluation_inputs={
                "domain": case.scenario["domain"],
                "unsafe": case.unsafe.spec if case.unsafe is not None else None,
                "clearance_status": case.scenario["unsafe_info"],
                "torque_limit_Nm": case.tau_limit,
            },
            x0_n=case.x0_n.tolist(),
            plant=case.plant_report,
            scenario_warnings=case.scenario["warnings"],
            normalization={
                "ps_mm": case.sc.ps_mm,
                "vs_mm_s": case.sc.vs_mm_s,
                "rate_per_s": case.rate,
            },
            embedding={
                "origin": case.emb.origin.tolist(),
                "axes": case.emb.E.tolist(),
                "L_m": case.emb.L,
                "anchor_n": case.emb.anchor_n.tolist(),
            },
        )
        write_json(output / "manifest.json", manifest)
        print(
            f"Running {cfg['model']} / {cfg['robot']} / {cfg['controller']}: "
            f"{case.sp.T_max:g} s at dt={case.sp.dt:g} s",
            flush=True,
        )
        stage = "simulation"
        result = (pic.run(case, compute_tracking=False)
                  if controller is pic and not compute_tracking else controller.run(case))
        stage = "artifacts"
        trajectory = result["traj"]
        np.savez_compressed(output / "traj.npz", **trajectory)
        manifest["artifact_sha256"]["traj.npz"] = sha256(output / "traj.npz")
        manifest["initial_state"] = case.initial_state
        write_json(output / "controller.json", {k: v for k, v in result.items() if k != "traj"})
        manifest["artifact_sha256"]["controller.json"] = sha256(output / "controller.json")
        stage = "evaluation"
        if compute_tracking:
            metrics = evaluate_saved_run(output, manifest, tracking=result.get("tracking"))
            write_json(output / "metrics.json", metrics)
            manifest["artifact_sha256"]["metrics.json"] = sha256(output / "metrics.json")
        manifest.update(status="complete", wall_time_s=time.perf_counter() - started)
        write_json(output / "manifest.json", manifest)
    except StartUnavailable as exc:
        save_start(dict(exc.row))
        manifest.update(status=exc.row["status"], failure_stage="start_ik",
                        wall_time_s=time.perf_counter() - started)
        write_json(output / "manifest.json", manifest)
        print(f"{exc.row['start_id']}: {exc}", file=sys.stderr)
        return 1
    except BaseException as exc:
        if case is not None and case.initial_state is not None:
            manifest["initial_state"] = case.initial_state
        elif stage == "simulation":
            stage = "controller_initialization"
        manifest.update(
            status="incomplete" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "error",
            error=f"{type(exc).__name__}: {exc}",
            failure_stage=stage,
            wall_time_s=time.perf_counter() - started,
        )
        if stage == "start_ik":
            manifest.setdefault("initialization_status", "error")
        write_json(output / "manifest.json", manifest)
        (output / "error.txt").write_text(traceback.format_exc())
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        print(f"Run failed; details in {output / 'error.txt'}", file=sys.stderr)
        return 1
    print(f"Saved run: {output.resolve()}")
    if not compute_tracking:
        return 0
    print(f"Status: complete; success: {metrics.get('success', 'NA')}")
    for key, label in (("final_distance_m", "Final distance [m]"),
                       ("mse_sync_plane_m2", "Windowed task-space MSE [m^2]"),
                       ("dtw_mean_plane_m", "Task-space DTW [m]")):
        value = metrics.get(key)
        print(f"{label}: {value:.9g}" if value is not None else f"{label}: NA")
    return 0
