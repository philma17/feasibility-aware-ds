"""Run configured learned-DS experiments and save raw MuJoCo trajectories."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
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
from uuid import uuid4

import numpy as np
import yaml

from evaluation.metrics import summarize
from sim.ds import ROOT, input_path, model_catalog
from sim.sim import build_case
from sim.robot import MENAGERIE_DIR, ROBOTS


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


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
    if cfg["controller"] not in ("pic", "servo"):
        raise ValueError("controller must be pic or servo")
    for key in ("dt", "T_max"):
        value = cfg["sim"][key]
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"sim.{key} must be finite and positive")
    steps = cfg["sim"]["T_max"] / cfg["sim"]["dt"]
    if not np.isfinite(steps) or steps < 1 or not np.isclose(steps, round(steps)):
        raise ValueError("T_max must be a positive integer multiple of dt")
    for label, value in (
        ("pic.use_f", cfg["pic"]["use_f"]),
        ("pic.torque_limit.clip", cfg["pic"]["torque_limit"]["clip"]),
        ("pic.null_space.enabled", cfg["pic"]["null_space"]["enabled"]),
        ("disturbance.enabled", cfg["disturbance"]["enabled"]),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{label} must be a boolean")
    return cfg


def input_hashes(cfg):
    package = model_catalog(cfg["model_catalog"])[cfg["model"]]
    if not package["selectable"]:
        raise ValueError(package["reason"])
    files = [
        package["checkpoint"],
        package["metadata"],
        *package["demos"],
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


def list_models(catalog):
    for name, package in model_catalog(catalog).items():
        if not package["selectable"]:
            print(f"{name}: unsupported ({package['reason']})")
            continue
        paths = [package["checkpoint"], package["metadata"], *package["demos"]]
        missing = [p for p in paths if not input_path(p).is_file()]
        print(f"{name}: " + (f"missing local inputs: {missing}" if missing else "available"))


def run_single(cfg, output, *, hashes=None, command_line=None, start_id=None):
    """Execute the existing single-run path; batches call the same writer and controller."""
    from sim.controllers import pic, servo

    output = Path(output)
    hashes = input_hashes(cfg) if hashes is None else hashes
    controller = {"pic": pic, "servo": servo}[cfg["controller"]]
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Output directory already exists: {output}", file=sys.stderr)
        return 2
    manifest = {
        "status": "incomplete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution": "dynamic MuJoCo simulation",
        "controller": cfg["controller"],
        "model": cfg["model"],
        "robot": cfg["robot"],
        "seed": cfg["sim"]["seed"],
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
    if start_id is not None:
        manifest["start_id"] = start_id
    manifest["implementation_sha256"]["main.py"] = sha256(ROOT / "main.py")
    write_json(output / "manifest.json", manifest)
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    started = time.perf_counter()
    stage = "start_ik"
    try:
        case = build_case(cfg)
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
        result = controller.run(case)
        np.savez_compressed(output / "traj.npz", **result["traj"])
        write_json(output / "controller.json", {k: v for k, v in result.items() if k != "traj"})
        stage = "evaluation"
        metrics = summarize(result, case)
        write_json(output / "metrics.json", metrics)
        manifest.update(
            status="complete",
            wall_time_s=time.perf_counter() - started,
            artifact_sha256={
                name: sha256(output / name)
                for name in ("config_resolved.yaml", "traj.npz", "metrics.json", "controller.json")
            },
        )
        write_json(output / "manifest.json", manifest)
    except BaseException as exc:
        manifest.update(
            status="incomplete" if isinstance(exc, KeyboardInterrupt) else "error",
            error=f"{type(exc).__name__}: {exc}",
            failure_stage=stage,
            wall_time_s=time.perf_counter() - started,
        )
        write_json(output / "manifest.json", manifest)
        (output / "error.txt").write_text(traceback.format_exc())
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        print(f"Run failed; details in {output / 'error.txt'}", file=sys.stderr)
        return 1
    print(f"Saved run: {output.resolve()}")
    print(
        f"Status: complete; success: {metrics['success']}; final distance: {metrics['final_distance_m']:.9g} m"
    )
    print(f"MSE: {metrics['mse_sync_m2']:.9g} m^2; DTW: {metrics['dtw_mean_m']:.9g} m")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", help="experiment YAML path")
    parser.add_argument(
        "--output", type=Path, help="new output directory; existing paths are refused"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override an existing configuration key (repeatable)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        help="batch size; uniform plotting.init_range starts using multistart.seed",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify assets and build the initial state without stepping",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="list supported packages and local input availability",
    )
    args = parser.parse_args(argv)
    if args.runs is not None:
        if args.runs < 1:
            parser.error("--runs must be a positive integer")
        if args.check:
            parser.error("--runs and --check cannot be combined")
    if args.list_models:
        list_models("configs/models.yaml")
        return 0
    if not args.config:
        parser.error("a configuration is required unless --list-models is used")
    try:
        cfg = load_config(args.config, args.set)
        hashes = input_hashes(cfg)
        from sim.controllers import pic

        pic.PICParams.from_cfg(cfg["pic"])
        if args.check:
            case = build_case(cfg)
            print(f"Ready: {cfg['model']} / {cfg['robot']} / {cfg['controller']}")
            print(f"Start FK error: {case.ik.pos_err:.9g} m; no simulation steps executed")
            for warning in case.scenario["warnings"]:
                print(f"Scenario warning: {warning}")
            return 0
    except (ValueError, KeyError, TypeError, FileNotFoundError, NotImplementedError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "results" / f"{stamp}_{cfg['controller']}_{uuid4().hex[:8]}"
    command_line = list(sys.argv if argv is None else argv)
    if args.runs is not None:
        from sim.batch import run_batch

        try:
            return run_batch(
                cfg, output, args.runs, run_single, hashes=hashes, command_line=command_line
            )
        except (ValueError, KeyError, TypeError, FileNotFoundError, NotImplementedError) as exc:
            print(f"Batch configuration error: {exc}", file=sys.stderr)
            return 2
    return run_single(cfg, output, hashes=hashes, command_line=command_line)


if __name__ == "__main__":
    sys.exit(main())
