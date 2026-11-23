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
`--set KEY=VALUE` to override a config value. For barrier plots, place the matching
`model_b.pth` alongside `model_f.pth` or pass `--barrier-model PATH`.
