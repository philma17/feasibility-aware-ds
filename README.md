# Feasibility-Aware Dynamical Systems for Imitation Learning

Evaluate whether learned dynamical systems can be tracked in MuJoCo while
respecting robot limits and avoiding obstacles. Supports FR3, fixed-joint and
mass variants, and UR5e with PIC torque control or position servo control.

## Quick start

Install [requirements.txt](requirements.txt) and place the checkpoints, metadata,
and demonstrations listed in [configs/models.yaml](configs/models.yaml) under
`data/`.

```bash
python -B main.py configs/five_obstacle_pic.yaml --check
python -B main.py configs/five_obstacle_pic.yaml --output results/example
python -B -m evaluation.aggregate results/example
python -B -m evaluation.plots results/example
```

Use a new output directory for each run. Add `--runs N` for a seeded batch or
`--set KEY=VALUE` to override a config value. Use the same experiment YAML for
each robot by setting `--set robot=FR3_fixed_J2_J3`, for example. The shared
configs use the stored joint start for the full FR3 joint layout and solve start
IK for other layouts. Batch starts always use IK.

To plot one saved run from a batch, pass the batch folder and its start ID:

```bash
python -B -m evaluation.plots results/my_batch --start-id start_001
```

Choose a completed start from `runs.csv`. Repeat `--start-id` to overlay runs.
The figure is saved to `results/my_batch/plots/trajectory_selected.png`.

For barrier plots, place the matching
`model_b.pth` alongside `model_f.pth` or pass `--barrier-model PATH`.
