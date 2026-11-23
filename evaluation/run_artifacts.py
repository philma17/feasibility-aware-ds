"""Read and write the files that describe a saved simulation run."""

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


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
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_config(directory, manifest):
    """Use the resolved YAML, or the embedded configuration of an old TCP run."""
    path = Path(directory) / "config_resolved.yaml"
    expected = manifest.get("artifact_sha256", {}).get(path.name)
    if expected is not None or "config" not in manifest:
        if expected is not None and sha256(path) != expected:
            raise ValueError("Saved effective configuration hash does not match the run")
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return manifest["config"]


def read_trajectory(directory, manifest):
    path = Path(directory) / "traj.npz"
    expected = manifest.get("artifact_sha256", {}).get(path.name, manifest.get("trajectory_sha256"))
    if expected is None or sha256(path) != expected:
        raise ValueError("Missing or mismatched trajectory hash")
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}
