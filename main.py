"""Command-line entry point for learned-DS experiments."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from uuid import uuid4

import experiment
from experiment import input_hashes, load_config
from sim.ds import ROOT, input_path, model_catalog
from sim.sim import build_case
from sim.batch import read_starts, run_batch


def list_models(catalog):
    for name, package in model_catalog(catalog).items():
        if not package["selectable"]:
            print(f"{name}: unsupported ({package['reason']})")
            continue
        paths = [package["checkpoint"], package["metadata"], *package["demos"]]
        missing = [p for p in paths if not input_path(p).is_file()]
        print(f"{name}: " + (f"missing local inputs: {missing}" if missing else "available"))


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
    parser.add_argument("--start-list", type=Path, help="replay validated rows from starts.json")
    parser.add_argument("--start-id", help="select a saved start, e.g. start_000")
    parser.add_argument("--member", help="select one member of a saved pair")
    parser.add_argument("--pair", choices=("posture", "lock", "robot"), help="paired batch design")
    parser.add_argument("--sigma-gap", type=float, help="required absolute posture-pair sigma difference")
    parser.add_argument("--lock", nargs=2, metavar=("JOINT", "RAD"), help="prescribed FR3 lock angle")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="list supported packages and local input availability",
    )
    args = parser.parse_args(argv)
    if args.pair and (args.runs is None or args.start_list):
        parser.error("--pair requires --runs and generates a new start list")
    if bool(args.lock) != (args.pair == "lock"):
        parser.error("--pair lock requires --lock JOINT RAD")
    if args.sigma_gap is not None and args.pair != "posture":
        parser.error("--sigma-gap requires --pair posture")
    if (args.start_id or args.member) and not args.start_list:
        parser.error("--start-id and --member select rows from --start-list")
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
        rows = (read_starts(args.start_list, start_id=args.start_id, member=args.member)
                if args.start_list else None)
        if rows is not None and args.runs is None and len(rows) != 1:
            raise ValueError("Select a single row with --start-id or request a batch with --runs")
        replay = {"start_id": rows[0]["start_id"], "start_row": rows[0]} if rows else {}
        if args.check:
            case = build_case(cfg, **replay)
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
        from evaluation.aggregate import aggregate

        try:
            return run_batch(
                cfg,
                output,
                args.runs,
                experiment.run_single,
                hashes=hashes,
                command_line=command_line,
                aggregate_runs=aggregate,
                comparison=args.pair, lock=args.lock, saved_rows=rows, hash_inputs=input_hashes,
                sigma_gap=0.001 if args.sigma_gap is None else args.sigma_gap,
            )
        except (ValueError, KeyError, TypeError, FileNotFoundError, NotImplementedError) as exc:
            print(f"Batch configuration error: {exc}", file=sys.stderr)
            return 2
    return experiment.run_single(cfg, output, hashes=hashes, command_line=command_line, **replay)


if __name__ == "__main__":
    sys.exit(main())
