"""Plan reproducible TCP starts, call the single-run writer, and aggregate every attempt."""

from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from itertools import groupby
from operator import itemgetter
from pathlib import Path
import re
import traceback

import numpy as np
import yaml

from evaluation.run_artifacts import write_json
from sim.ds import MULTISTART_DEFAULT, load_scenario, sample_starts
from sim.pairs import comparison_configs, plan_pair

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
    "robot",
    "member",
    "pair_id",
    "initialization_status",
)


def read_starts(path, *, start_id=None, member=None):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("starts.json must contain a list")
    selected = []
    seen = set()
    for row in rows:
        identity = row["start_id"]
        if start_id is not None and identity != start_id or member is not None and row.get("member") != member:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identity):
            raise ValueError(f"Invalid start_id: {identity!r}")
        if identity in seen:
            raise ValueError(f"Ambiguous start_id {identity!r}. Select a pair member with --member")
        seen.add(identity)
        selected.append(row)
    if not selected:
        raise ValueError("No matching saved start")
    return selected


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
        row["initialization_status"] = saved.get("initialization_status")
        row["ik_seed"] = saved.get("seeds", {}).get("ik", row["ik_seed"])
        for key in ("pair_id", "member"):
            row[key] = saved.get(key, row.get(key))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        row.update(status="invalid_manifest", error=str(exc))


def record_setup_error(output, row, cfg, exc):
    directory = output / row["run_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        return
    manifest = {key: row[key] for key in ("model", "robot", "start_id", "member", "pair_id")}
    manifest.update(status="error", controller=cfg["controller"], failure_stage="run_setup",
                    error=f"{type(exc).__name__}: {exc}")
    write_json(directory / "manifest.json", manifest)
    (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")


def run_batch(cfg, output, count, run_single, *, hashes, command_line, aggregate_runs=None,
              comparison=None, lock=None, sigma_gap=0.001, saved_rows=None, hash_inputs=None):
    """Keep failed and unattempted starts in the same plan and final table."""
    if aggregate_runs is None:
        from evaluation.aggregate import aggregate

        aggregate_runs = aggregate
    configs = comparison_configs(cfg, comparison, lock) if comparison else {None: cfg}
    if comparison == "posture" and not (np.isfinite(sigma_gap) and sigma_gap > 0):
        raise ValueError("--sigma-gap must be finite and positive")
    if saved_rows is None:
        targets = plan_starts(cfg, count)
    else:
        if count > len(saved_rows):
            raise ValueError("Requested more runs than available saved starts")
        targets = [
            {"start_index": i, "start_id": r["start_id"], "run_dir": r["start_id"],
             "model": cfg["model"], "start_seed": r.get("start_seed"), "ik_seed": r["ik_seed"],
             "x0_n": r["x0_n"], "tcp_start_m": r["tcp_start_m"], "status": "planned",
             "q_start_rad": None, "error": None}
            for i, r in enumerate(saved_rows[:count])
        ]
    rows = [{**target, "robot": variant["robot"], "member": member,
             "pair_id": f"{comparison}_{target['start_id']}" if member else None,
             "run_dir": f"{target['start_id']}/{member}" if member else target["start_id"]}
            for target in targets for member, variant in configs.items()]
    robot_hashes = {cfg["robot"]: hashes}
    for variant in configs.values():
        if variant["robot"] not in robot_hashes:
            robot_hashes[variant["robot"]] = hash_inputs(variant)
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
        "planned_runs": len(rows),
        "command_line": list(command_line),
        "input_sha256": hashes,
        "sampling": "saved starts.json" if saved_rows is not None else "uniform plotting.init_range; source NumPy default_rng",
        "start_seed": rows[0]["start_seed"],
        "ik_seed": cfg["sim"]["seed"] if saved_rows is None else None,
        "ik_n_starts": cfg["sim"]["n_starts"] if saved_rows is None else 0,
    }
    if comparison:
        report.update(comparison=comparison, sigma_gap=sigma_gap if comparison == "posture" else None,
                      lock=lock)
    write_json(output / "batch.json", report)
    completed = False
    exit_code = 0
    try:
        for index, group in groupby(rows, key=itemgetter("start_index")):
            group = list(group)
            variants = {member: start_config(c, count, index) for member, c in configs.items()}
            start_id = group[0]["start_id"]
            print(f"Batch start {index + 1}/{count}: {start_id}", flush=True)
            try:
                pair = plan_pair(variants, start_id, comparison, sigma_gap) if comparison else {}
            except Exception as exc:
                for row in group:
                    record_setup_error(output, row, cfg, exc)
                exit_code = 1
                continue
            for row in group:
                member = row["member"]
                child = variants[member]
                replay = {}
                if comparison:
                    replay["start_row"] = {**pair[member], "start_seed": row["start_seed"]}
                elif saved_rows is not None:
                    replay["start_row"] = saved_rows[index]
                try:
                    code = run_single(child, output / row["run_dir"], hashes=robot_hashes[child["robot"]],
                                      command_line=command_line, start_id=start_id, **replay)
                except Exception as exc:
                    record_setup_error(output, row, cfg, exc)
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
        records = []
        for row in rows:
            path = output / row["run_dir"] / "starts.json"
            if path.is_file():
                records.extend(json.loads(path.read_text(encoding="utf-8")))
        if records:
            write_json(output / "starts.json", records)
        if comparison:
            valid = {member: {r["start_id"] for r in records if r["member"] == member
                              and r["status"] == "valid"} for member in configs}
            report["valid_start_ids"] = {k: sorted(ids) for k, ids in valid.items()}
            report["common_valid_start_ids"] = sorted(set.intersection(*valid.values()))
            report["valid_start_counts"] = {member: len(ids) for member, ids in valid.items()}
            report["common_valid_pairs"] = len(report["common_valid_start_ids"])
            print(f"Valid starts {report['valid_start_counts']}, common pairs {report['common_valid_pairs']}")
        write_json(output / "batch.json", report)
        print(f"Saved batch table: {csv_path.resolve()} ({len(table)} planned starts)", flush=True)
    return exit_code
