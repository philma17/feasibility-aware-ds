"""Evaluate trusted saved Barrier models on logged trajectories in their model domain."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from evaluation.run_artifacts import read_config, read_trajectory, sha256 as file_sha256


def recorded_model_hashes(run_dir, manifest):
    """Find producer hashes, falling back to the original TCP study input record."""
    if "input_sha256" in manifest:
        return manifest["input_sha256"], manifest.get("barrier_sha256")
    if "config" in manifest and "plan" in manifest:
        for parent in Path(run_dir).resolve().parents:
            record = parent / "provenance" / "inputs.json"
            if record.is_file():
                inputs = json.loads(record.read_text(encoding="utf-8"))
                return inputs["inputs"], inputs["barrier_sha256"]
    raise ValueError("Recorded model input hashes are unavailable")


def saved_model_inputs(run_dir, manifest):
    """Resolve catalog inputs only if they still match the recorded model hashes."""
    from sim.ds import input_path, model_catalog

    cfg = read_config(run_dir, manifest)
    hashes, _ = recorded_model_hashes(run_dir, manifest)
    catalog = cfg["model_catalog"]
    if file_sha256(input_path(catalog)) != hashes[catalog]:
        raise ValueError("Model catalog has changed since this run")
    package = model_catalog(catalog)[cfg["model"]]
    for key in ("metadata", "checkpoint"):
        path = package[key]
        if file_sha256(input_path(path)) != hashes[path]:
            raise ValueError(f"Model input has changed since this run: {path}")
    spec = json.loads(input_path(package["metadata"]).read_text(encoding="utf-8"))
    return cfg, package, spec


def model_to_world(points, embedding):
    """Use the run's effective embedding, as in LfD_sim/pic/plots.py::to_world."""
    return np.asarray(embedding["origin"]) + embedding["L_m"] * (
        (np.asarray(points) - np.asarray(embedding["anchor_n"])) @ np.asarray(embedding["axes"]).T
    )


def world_to_model(points, embedding):
    """Invert the logged planar or spatial embedding without changing normalization."""
    axes = np.asarray(embedding["axes"], float)
    scale = float(embedding["L_m"])
    if axes.shape not in ((3, 2), (3, 3)) or scale <= 0 or not np.isfinite(scale):
        raise ValueError("Expected a finite planar or spatial embedding with positive L_m")
    if not np.allclose(axes.T @ axes, np.eye(axes.shape[1]), rtol=0, atol=1e-10):
        raise ValueError("Embedding axes must be orthonormal")
    return (
        (np.asarray(points) - np.asarray(embedding["origin"])) @ axes / scale
        + np.asarray(embedding["anchor_n"])
    )


def domain_box(spec):
    """X is the closed model box from domain.range, independently of plotting.range."""
    box = np.asarray(spec["domain"]["range"], float)
    if (
        box.ndim != 2
        or box.shape[1] != 2
        or not np.isfinite(box).all()
        or np.any(box[:, 0] >= box[:, 1])
    ):
        raise ValueError("domain.range must contain finite increasing bounds")
    return box


def resolve_barrier_model(package, manifest, path=None, *, run_dir=None):
    """Require the Barrier checkpoint's sibling vector field to match the saved run."""
    from sim.ds import input_path

    path = Path(path) if path else input_path(package["checkpoint"]).with_name("model_b.pth")
    if not path.is_file():
        raise ValueError("Barrier model missing; supply the matching local --barrier-model PATH")
    hashes, barrier_hash = (recorded_model_hashes(run_dir, manifest) if run_dir is not None
                            else (manifest["input_sha256"], manifest.get("barrier_sha256")))
    if barrier_hash is None and "evaluation_schema" in manifest:
        raise ValueError("No Barrier checkpoint hash was recorded for this run")
    if barrier_hash is not None and file_sha256(path) != barrier_hash:
        raise ValueError("Barrier checkpoint hash does not match the archived input")
    if file_sha256(path.with_name("model_f.pth")) != hashes[package["checkpoint"]]:
        raise ValueError("Barrier package model_f.pth does not match this run")
    return path


@lru_cache(maxsize=4)
def _barrier_model(path, digest):
    # Including the digest in the key prevents reuse after a checkpoint changes.
    from sim.ds import load_torch_reference

    return load_torch_reference(path).cpu()


def barrier_values(path, points):
    """Reuse the source float32 network evaluation, in bounded-memory batches."""
    import torch

    points = np.asarray(points)
    model = _barrier_model(str(Path(path).resolve()), file_sha256(path))
    if points.ndim != 2 or points.shape[1] != model.input_size or not np.isfinite(points).all():
        raise ValueError("Barrier input dimension or finite values do not match model")
    values = []
    with torch.no_grad():
        for start in range(0, len(points), 32768):
            tensor = torch.as_tensor(points[start : start + 32768], dtype=torch.float32)
            values.append(model(tensor).numpy().reshape(-1))
    result = np.concatenate(values) if values else np.empty(0)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite barrier output")
    return result


def path_events(t, points, values, box):
    """Count strict B>0 only in closed X; box exits are a separate sampled event."""
    t, points, values = np.asarray(t), np.asarray(points), np.asarray(values)
    box = np.asarray(box)
    if (
        t.ndim != 1
        or not len(t)
        or points.shape != (len(t), len(box))
        or values.shape != t.shape
        or not np.isfinite(t).all()
        or not np.isfinite(points).all()
        or not np.isfinite(values).all()
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError("Barrier evaluation requires finite, synchronized trajectory samples")
    inside = np.all((points >= box[:, 0]) & (points <= box[:, 1]), axis=1)
    positive = inside & (values > 0)

    def first(mask):
        indices = np.flatnonzero(mask)
        return float(t[indices[0]]) if len(indices) else None

    return {
        "n_samples": len(t),
        "inside_samples": int(inside.sum()),
        "positive_samples_inside": int(positive.sum()),
        "first_positive_inside_s": first(positive),
        "box_exit_samples": int((~inside).sum()),
        "first_box_exit_s": first(~inside),
        "max_B_inside": float(values[inside].max()) if inside.any() else None,
        "max_B_outside": float(values[~inside].max()) if (~inside).any() else None,
    }


def evaluate_paths(t, measured, ideal, embedding, spec, barrier_model):
    """Evaluate ideal DS and measured TCP separately at the saved state timestamps."""
    box = domain_box(spec)
    result = {}
    for name, points in (("ideal", ideal), ("tcp", measured)):
        normalized = world_to_model(points, embedding)
        values = barrier_values(barrier_model, normalized)
        result[name] = path_events(t, normalized, values, box)
    return result


def evaluate_run(run_dir, barrier_model, start_id=None):
    """Return new evaluation results without modifying raw runs or their metrics.json."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    _, package, spec = saved_model_inputs(run_dir, manifest)
    result = {"start_id": start_id or manifest.get("start_id", run_dir.name)}
    if not spec.get("Barrier", False):
        return {**result, "status": "no_barrier_model"}
    path = resolve_barrier_model(package, manifest, barrier_model, run_dir=run_dir)
    arrays = read_trajectory(run_dir, manifest)
    paths = evaluate_paths(
        arrays["t"], arrays["P"], arrays["P_ideal"], manifest["embedding"], spec, path
    )
    hashes, _ = recorded_model_hashes(run_dir, manifest)
    return {
        **result,
        "status": "evaluated",
        "barrier_model": str(path.resolve()),
        "barrier_sha256": file_sha256(path),
        "metadata_sha256": hashes[package["metadata"]],
        "trajectory_sha256": manifest.get("artifact_sha256", {}).get("traj.npz", manifest.get("trajectory_sha256")),
        "domain_range": domain_box(spec).tolist(),
        "paths": paths,
    }
