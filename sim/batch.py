"""Plan reproducible TCP starts, call the single-run writer, and aggregate every attempt."""

from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import yaml

from evaluation.run_artifacts import write_json
from sim.ds import MULTISTART_DEFAULT, load_scenario, sample_starts

START_FIELDS = (
    "start_id",
    "run_dir",
    "start_index",
    "model",
    "start_seed",
    "ik_seed",
    "x0_n",
    "tcp_start_m",
    "q_start_rad",
    "status",
    "error",
)


def start_config(cfg, count, index):
    """Replace the nominal single start with indexed sampling and the existing start IK."""
    child = deepcopy(cfg)
    # This count controls sampled TCP targets; sim.n_starts controls IK candidates.
    child["multistart"] = {**MULTISTART_DEFAULT, **(cfg.get("multistart") or {}), "n_starts": count}
    child["task"]["start"] = {"index": index}
    child["task"]["q_start_rad"] = None
    return child


def plan_starts(cfg, count):
    """Use exactly the loader's sampling and embedding; controller choice is irrelevant."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("runs must be a positive integer")
    child = start_config(cfg, count, 0)
    scenario = load_scenario(child)
    box = scenario["init_box"]
    if box is None:
        raise ValueError("Batch execution requires plotting.init_range in the model metadata")
    box = np.asarray(box, float)
    if (
        box.shape != (scenario["dim"], 2)
        or not np.isfinite(box).all()
        or np.any(box[:, 0] > box[:, 1])
    ):
        raise ValueError(
            "plotting.init_range must contain finite ordered bounds for every model axis"
        )
    seed = child["multistart"]["seed"]
    X0 = sample_starts(box, seed, count)
    P0 = scenario["emb"].to_world(X0)
    P0 = P0 + float(child["task"].get("out_of_plane_offset_m", 0)) * scenario["emb"].normal
    return [
        {
            "start_id": f"start_{i:03d}",
            "run_dir": f"start_{i:03d}",
            "start_index": i,
            "model": cfg["model"],
            "start_seed": seed,
            "ik_seed": cfg["sim"]["seed"],
            "x0_n": x.tolist(),
            "tcp_start_m": p.tolist(),
            "q_start_rad": None,
            "status": "planned",
            "error": None,
        }
        for i, (x, p) in enumerate(zip(X0, P0))
    ]


def write_starts(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=START_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        "NA"
                        if value is None
                        else (json.dumps(value) if isinstance(value, list) else value)
                    )
                    for key, value in row.items()
                }
            )


def refresh_start(output, row):
    path = output / row["run_dir"] / "manifest.json"
    if not path.is_file():
        row["status"] = "missing"
        return
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        row["status"] = saved["status"]
        row["q_start_rad"] = saved.get("q_start_rad")
        row["error"] = saved.get("error")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        row.update(status="invalid_manifest", error=str(exc))


def run_batch(cfg, output, count, run_single, *, hashes, command_line, aggregate_runs=None):
    """Keep failed and unattempted starts in the same plan and final table."""
    if aggregate_runs is None:
        from evaluation.aggregate import aggregate

        aggregate_runs = aggregate
    rows = plan_starts(cfg, count)
    output = Path(output)
    if output.exists():
        # No resume path: reusing a directory could mix different experiments.
        raise ValueError(f"Output directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    starts = output / "starts.csv"
    # Persist the full plan first so interrupted batches retain unattempted starts.
    write_starts(starts, rows)
    (output / "batch_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    report = {
        "status": "incomplete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": cfg["model"],
        "robot": cfg["robot"],
        "controller": cfg["controller"],
        "planned_runs": count,
        "command_line": list(command_line),
        "input_sha256": hashes,
        "sampling": "uniform plotting.init_range; source NumPy default_rng",
        "start_seed": rows[0]["start_seed"],
        "ik_seed": cfg["sim"]["seed"],
        "ik_n_starts": cfg["sim"]["n_starts"],
    }
    write_json(output / "batch.json", report)
    completed = False
    exit_code = 0
    try:
        for row in rows:
            child = start_config(cfg, count, row["start_index"])
            directory = output / row["run_dir"]
            print(f"Batch start {row['start_index'] + 1}/{count}: {row['start_id']}", flush=True)
            try:
                code = run_single(
                    child,
                    directory,
                    hashes=hashes,
                    command_line=command_line,
                    start_id=row["start_id"],
                )
            except Exception as exc:
                # Failures outside the single-run writer's exception handler still get a record.
                directory.mkdir(parents=True, exist_ok=True)
                if not (directory / "manifest.json").exists():
                    write_json(
                        directory / "manifest.json",
                        {
                            "status": "error",
                            "model": cfg["model"],
                            "robot": cfg["robot"],
                            "controller": cfg["controller"],
                            "start_id": row["start_id"],
                            "error": f"{type(exc).__name__}: {exc}",
                            "failure_stage": "run_setup",
                        },
                    )
                    (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                code = 1
            refresh_start(output, row)
            write_starts(starts, rows)
            if code != 0 or row["status"] != "complete":
                exit_code = 1
        completed = True
    finally:
        # Also runs after Ctrl-C: untouched plan entries become explicit missing rows.
        for row in rows:
            refresh_start(output, row)
        write_starts(starts, rows)
        csv_path, table, _ = aggregate_runs(output)
        report.update(
            status="complete" if completed else "incomplete",
            complete_runs=sum(row["status"] == "complete" for row in table),
            other_runs=sum(row["status"] != "complete" for row in table),
            starts_sha256=hashlib.sha256(starts.read_bytes()).hexdigest(),
        )
        write_json(output / "batch.json", report)
        print(f"Saved batch table: {csv_path.resolve()} ({len(table)} planned starts)", flush=True)
    return exit_code
