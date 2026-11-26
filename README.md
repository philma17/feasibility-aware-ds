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
IK for other layouts. Fresh batch starts use IK. A saved nominal start can also be replayed by either entry point.

To plot one saved run from a batch, pass the batch folder and its start ID:

```bash
python -B -m evaluation.plots results/my_batch --start-id start_001
```

Choose a completed start from `runs.csv`. Repeat `--start-id` to overlay runs.
The figure is saved to `results/my_batch/plots/trajectory_selected.png`.

For barrier plots, place the matching
`model_b.pth` alongside `model_f.pth` or pass `--barrier-model PATH`.

## Start posture and orientation

The DS start `x0_n` comes from the demonstration, an explicit coordinate, or the
indexed sample from `multistart.seed`. The embedding maps it to the world TCP
target, including `out_of_plane_offset_m`. The legacy single-run path first
tries `task.q_start_rad` for the matching joint layout. It reports FK error
and accepts that branch even when it misses a 3D or 6D target. Otherwise
`multistart_ik` starts from `q_home` and draws the remaining seeds using
`sim.seed`. The original batch clears `q_start_rad` at each index.
`max_sigma_margin` still maximizes the original unscaled IK score among
converged candidates above the preferred margin, with its original fallback.
Both controllers set MuJoCo qpos to the chosen vector and qvel to zero.

| `task.orientation` | Start IK | PIC orientation feedback | Servo orientation feedback |
| --- | --- | --- | --- |
| `none` | TCP position | Free, logs rotation from initial frame | Free, logs rotation from initial frame |
| `damped` | TCP position | Damps all three world angular velocities, logs drift | Free, logs drift |
| `hold` | Position and full `R_des` | Restores `R_des` and damps angular velocity | Commands angular velocity towards `R_des` |
| `axis5` | Position and tool direction | Tilt stiffness `K_r` and tilt damping `D_tilt` | Commands two tilt rates |

`hold` defaults to the world rotation `diag(1,-1,-1)`. Start positions use
XYZ even for a legacy planar `position_mode: xy` rollout. The 5D mode requires
`position_mode: xyz`. Its two angular rows span the plane perpendicular to the
**current** world tool axis. A half-turn uses a deterministic perpendicular
correction axis. Rotation about the tool axis generates zero orientation
error and zero orientation-task damping. The independently configured PIC
null-space joint damping and the physical joint damping still apply.

In `configs/worm_axis5_pic.yaml`, the FR3 site is 0.107 m along link 7's +Z
with identity site rotation. The direction from wrist to virtual tip is site
+Z. On UR5e the same site +Z points along wrist 3's +Y toward the site at
(0, 0.1, 0) m. `tool_direction_world: inward_plane_normal` targets the negative
cross product of the embedding's orthonormal drawing axes, including tilted
planes. An explicit world unit 3-vector also works. The virtual tip uses the
existing attachment site. There is no pen surface contact model.

## Nominal starts and replay

Select `sim.start_criterion: closest_to_reference` to solve for the
Cartesian target even in a config that contains an older fixed joint start.
The reference in radians is the named MJCF `home` keyframe, verified against
the model joint names, order and limits.

| Model | Joint order | Home reference [rad] |
| --- | --- | --- |
| FR3 | fr3_joint1 … fr3_joint7 | 0, 0, 0, −1.57079, 0, 1.57079, −0.7853 |
| UR5e | shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3 (all suffixed `_joint`) | −1.5708, −1.5708, 1.5708, −1.5708, −1.5708, 0 |

Reduced arms use the `home` in their registered MJCF with the removed axes
omitted. Each start tries this reference first. The remaining uniform joint
seeds depend on `sim.seed` and `start_id`. Candidates must pass FK position
and angular tolerances, joint limits, model contact detection and supported
TCP obstacle membership tests (boundary included in the unsafe set). The
minimum sum of squared joint differences wins, in seed order on ties. Only
unlimited hinges wrap around. All registered FR3 and UR5e axes have stops.

Nominal single runs and batches write `starts.json` with the normalized and
world target, measured TCP, active and full angles, zero velocity, reference,
seed, status, model hash, site, locks, placement, task and collision-check
availability. Replay restores the row's normalized coordinate and validates
its FK, limits, contacts and task against the current config before stepping.
`starts.csv` remains the execution plan and `runs.csv` the results table.
An unreachable target keeps its ID and gets no trajectory or goal failure.

```bash
python -B main.py configs/worm_axis5_pic.yaml --output results/nominal
python -B main.py configs/worm_axis5_pic.yaml --runs 2 --output results/nominal-batch
python -B main.py configs/worm_axis5_pic.yaml --start-list results/nominal-batch/starts.json --start-id start_001 --output results/replay
python -B main.py configs/worm_axis5_pic.yaml --runs 2 --pair posture --sigma-gap 0.0001 --output results/postures
python -B main.py configs/worm_axis5_pic.yaml --runs 2 --pair lock --lock fr3_joint5 0.2 --output results/locks
python -B main.py configs/worm_axis5_pic.yaml --runs 2 --pair robot --output results/robots
```

Use a fresh output directory. For a four-step smoke run append
`--set sim.T_max=0.004`; `--set sim.n_starts=6` limits IK to six seeds
per target. To replay a pair member, use its `config_resolved.yaml` with
the batch's `--start-list`, `--start-id` and `--member` (for example
`lower_sigma` or `locked`). To replay several rows, use `--runs N`
instead of selecting one ID. A batch containing an unavailable start returns
exit code 1 after writing its tables.

Run the start and replay regression tests with
`python -B -m unittest discover -s tests -p test_starts.py`.

## Paired comparisons

- **Posture.** Both rollouts use one target, placement, task, velocity and
  rollout config. The second posture minimizes the task's smallest singular
  value among distinct valid candidates. Its absolute drop must meet
  `--sigma-gap` (default 0.001). Both measured values are recorded. If the
  search yields only a nominal member, it can run and the missing partner
  is recorded explicitly. For PIC hold/damped pairs the nominal posture
  supplies the inertia-derived damping gain for both members. A relative
  decrease alone assigns no absolute singularity class.
- **Prescribed lock.** The reduced FR3 model bakes the requested hinge angle
  into its rigid body transform before IK. The planner checks candidates in
  both models, inserts that exact angle into the free arm's full vector and
  validates the initialized rigid-body orientation. The two starts have the
  same seven physical angles, with six movable coordinates on the locked arm.
- **Robot.** FR3 and UR5e solve their own IK for each shared world target.
  `batch.json` lists the valid IDs by member and their intersection.

`traj.npz:sigma_min_task` covers all N+1 measured states. `sigma_min_start`
and `sigma_min_run` in the metrics and run table are its first and smallest
values. Only movable columns and controlled rows enter the Jacobian. For
5D/6D, translation uses metres and angular rows are multiplied by
`sim.sigma_rotation_length_m` (0.1 m/rad in the example, default 1).
Numerically deficient row rank, including fewer columns than required rows,
gives zero. Historical servo solver and PIC translation diagnostics retain
their own fields. The posture summary compares complete paired runs with
matching settings. Goal success retains the experiment's final TCP distance
criterion. The angular error is logged independently.
