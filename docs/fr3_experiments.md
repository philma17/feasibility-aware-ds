# FR3: separate joint-lock and link-density experiments

Work branch: `dof`. Initial worktree on 2026-11-17 was clean. The original
`fr3.xml`, existing fixed-joint XMLs, shared meshes and PIC controller are retained.

## Joint-lock series

| Model (`robot`) | Free joints, in array order | Fixed joints [rad] | Configuration |
| --- | --- | --- | --- |
| `FR3` | J1,J2,J3,J4,J5,J6,J7 | — | `configs/dof/five_obstacle_pic_7dof.yaml` |
| `FR3_fixed_J3_J5` | J1,J2,J4,J6,J7 | J3=J5=0 | `configs/dof/five_obstacle_pic_5dof_j3_j5.yaml` |
| `FR3_fixed_J3_J7` | J1,J2,J4,J5,J6 | J3=J7=0 | `configs/dof/five_obstacle_pic_5dof_j3_j7.yaml` |
| `FR3_fixed_J3_J5_J7` | J1,J2,J4,J6 | J3=J5=J7=0 | `configs/dof/five_obstacle_pic_4dof.yaml` |
| `FR3_fixed_J3_J5_J6_J7` | J1,J2,J4 | J3=J5=J7=0; J6=2.5307 | `configs/dof/five_obstacle_pic_3dof.yaml` |

The additional XML filenames are the lowercase model names, under
`vendor/franka_fr3/`. J6's midpoint is `(0.4398 + 4.6216)/2 = 2.5307 rad`.
For the 3-DoF arm its hinge rotation is composed with the original local body
rotation: `R_fixed = R_body R_axis(2.5307)`. The resulting `fr3_link6` quaternion
(MuJoCo w,x,y,z order) is
`[0.21264035414377619, 0.21264035414377619, -0.67437680846067749, 0.67437680846067749]`.
The hinge anchor is zero, so the relative position stays zero. The generator
also accounts for a nonzero anchor and joint reference angle.

Locked joints and their actuators are removed. Keyframes contain only the free
coordinates, with original joint names. All bodies, inertials, meshes and TCP
are retained. These models contain no equality constraints. The 5-DoF models
compare different lock locations at the same axis count; link masses are unchanged.

All configurations use start IK (`task.q_start_rad: null`). They draw identical
world TCP targets; each arm requires its own joint vector. Joint vectors of
different DoF or lock placements are therefore not equated. A failed start IK
is recorded by the existing batch runner and remains in the tables.

From the repository root, with the dependency environment activated:

```bash
python -B main.py configs/dof/five_obstacle_pic_7dof.yaml --runs 100 --output results/dof_fr3_7dof_100
python -B main.py configs/dof/five_obstacle_pic_5dof_j3_j5.yaml --runs 100 --output results/dof_fr3_5dof_j3_j5_100
python -B main.py configs/dof/five_obstacle_pic_5dof_j3_j7.yaml --runs 100 --output results/dof_fr3_5dof_j3_j7_100
python -B main.py configs/dof/five_obstacle_pic_4dof.yaml --runs 100 --output results/dof_fr3_4dof_100
python -B main.py configs/dof/five_obstacle_pic_3dof.yaml --runs 100 --output results/dof_fr3_3dof_100
```

## Link-density series (full 7 DoF only)

| Model (`robot`) | Changed link | Link mass [kg] | Added mass [kg] | Configuration |
| --- | --- | ---: | ---: | --- |
| `FR3` | — | — | 0 | `configs/mass/five_obstacle_pic_reference.yaml` |
| `FR3_link3_mass_150pct` | `fr3_link3` | 3.36735 | 1.12245 | `configs/mass/five_obstacle_pic_link3_mass_150pct.yaml` |
| `FR3_link3_mass_200pct` | `fr3_link3` | 4.4898 | 2.2449 | `configs/mass/five_obstacle_pic_link3_mass_200pct.yaml` |
| `FR3_link7_mass_150pct` | `fr3_link7` | 0.9407145 | 0.3135715 | `configs/mass/five_obstacle_pic_link7_mass_150pct.yaml` |
| `FR3_link7_mass_200pct` | `fr3_link7` | 1.254286 | 0.627143 | `configs/mass/five_obstacle_pic_link7_mass_200pct.yaml` |

The XML filenames are the lowercase model names under `vendor/franka_fr3/`.
Each scales exactly one link's `mass` and all three `diaginertia` entries by
1.5 or 2.0. Its inertial `pos` and `quat` stay unchanged, as do geometry, all
other links and all joints. All variants use the same `assets/` directory.
This models a uniform density change at unchanged geometry.

Both plant and nominal robot are constructed with `Robot(ROBOTS[name])` in
`sim/sim.py:build_case`. PIC's existing `gravity_comp: nominal` selects
`case.robot_nominal` in `sim/controllers/pic.py`; it therefore uses the selected
variant XML, including its density change. It does not select `fr3.xml`
independently. The local diagnostic checks the MJCF paths, compiled mass,
inertia, COM and gravity arrays, and gravity torques. All five nominal/plant
pairs match exactly, including zero maximum gravity-torque difference.
There is no need to switch to `gravity_comp: plant` for these configurations.

The shared settings are `plant.mass_scale: 1.0`, `plant.inertia_scale: 1.0`,
`plant.gravity_scale: 1.0`, no COM shift, `pic.gravity_comp: nominal`,
`time_rate: 1.0`, `ds_speed_factor: 1.0`, `task.L_m: 0.1`,
`plant.torque_limit_scale: 1.0`, and the unchanged PIC settings.
Moment limits in J1..J7 order are `[87, 87, 87, 87, 12, 12, 12] Nm`.

Single runs retain the exact same seven reference start angles. Batches use
the existing indexed TCP sampling and deterministic kinematic start IK:
`multistart.seed: 0`, `sim.seed: 0`, 32 IK candidates, `max_sigma_margin`,
minimum margin 5%. The mass variants explicitly use the unchanged FR3 model
for start IK with the same plant settings. This prevents MuJoCo's COM-based
Jacobian roundoff (observed differences around 1e-17 after density changes)
from producing slightly different IK angles. Mass and inertia are not IK
selection criteria. The common reference model and equal seeds retain exactly
the same initial joint configuration per indexed start across all mass variants.
Simulation and gravity compensation still use the selected density XML. The
tests check this explicitly; DS scaling, embedding and moment limits are also
unchanged.

```bash
python -B main.py configs/mass/five_obstacle_pic_reference.yaml --runs 100 --output results/mass_fr3_reference_100
python -B main.py configs/mass/five_obstacle_pic_link3_mass_150pct.yaml --runs 100 --output results/mass_fr3_link3_150pct_100
python -B main.py configs/mass/five_obstacle_pic_link3_mass_200pct.yaml --runs 100 --output results/mass_fr3_link3_200pct_100
python -B main.py configs/mass/five_obstacle_pic_link7_mass_150pct.yaml --runs 100 --output results/mass_fr3_link7_150pct_100
python -B main.py configs/mass/five_obstacle_pic_link7_mass_200pct.yaml --runs 100 --output results/mass_fr3_link7_200pct_100
```

These commands are prepared for later execution; the 100-run simulations were
not executed. Output directories must be new. The local environment can be
activated with `source /home/philipp/lfd_env/bin/activate`.

## Common-pose gravity comparison

The diagnostic calls the existing `sim.robot.gravity_torque` routine:
`g(q) = qfrc_bias` at zero joint velocity. These are compensation torques in
Nm, in original J1..J7 order; `Δg = g_variant - g_FR3`. The comparison pose is
the exact saved nominal Five Obstacle start:

```text
q [rad] = [0.0001392701517455879, 0.18456123940941815,
           0.00013888841695262956, -1.6143835872776873,
           0.000027150350159038066, 1.604234929676289, -0.7853]
```

Numbers below are rounded to nine decimals; numerical residuals below that
precision are shown as zero.

| Model | g(q) [Nm], J1..J7 |
| --- | --- |
| `FR3` | `[0, -32.180669400, 0.173847372, 19.577574855, 0.809731482, 1.509306601, 0.003452153]` |
| `FR3_link3_mass_150pct` | `[0, -33.201243775, 0.164118759, 19.577574855, 0.809731482, 1.509306601, 0.003452153]` |
| `FR3_link3_mass_200pct` | `[0, -34.221818150, 0.154390145, 19.577574855, 0.809731482, 1.509306601, 0.003452153]` |
| `FR3_link7_mass_150pct` | `[0, -34.119819254, 0.172252974, 21.088864591, 0.801030952, 1.812570771, 0.005178230]` |
| `FR3_link7_mass_200pct` | `[0, -36.058969108, 0.170658575, 22.600154327, 0.792330422, 2.115834942, 0.006904307]` |

| Model | Δg(q) [Nm], J1..J7 |
| --- | --- |
| `FR3` | `[0, 0, 0, 0, 0, 0, 0]` |
| `FR3_link3_mass_150pct` | `[0, -1.020574375, -0.009728614, 0, 0, 0, 0]` |
| `FR3_link3_mass_200pct` | `[0, -2.041148751, -0.019457227, 0, 0, 0, 0]` |
| `FR3_link7_mass_150pct` | `[0, -1.939149854, -0.001594399, 1.511289736, -0.008700530, 0.303264171, 0.001726077]` |
| `FR3_link7_mass_200pct` | `[0, -3.878299709, -0.003188797, 3.022579472, -0.017401060, 0.606528341, 0.003452153]` |

Recompute with the local `tools/` checkout, or pass `--q J1 J2 J3 J4 J5 J6 J7`
for another common pose.

```bash
python -B -m tools.compare_fr3_mass --output /tmp/fr3_mass_comparison.json
```

The JSON records the exact pose, actually added compiled mass, full-precision
torque vectors and deltas, selected XMLs and nominal/plant equality.

## Worm/PIC: the same nine models

The nine Worm configurations use `configs/worm_pic.yaml` as their baseline:
the full FR3, four joint-lock variants and four density variants. Joint locks
and density changes remain separate series. `configs/worm_pic.yaml` also serves
as the mass-series reference; `configs/dof/worm_pic_7dof.yaml` is identical to it.

| Model (`robot`) | Worm/PIC configuration |
| --- | --- |
| `FR3` | `configs/dof/worm_pic_7dof.yaml` |
| `FR3_fixed_J3_J5` | `configs/dof/worm_pic_5dof_j3_j5.yaml` |
| `FR3_fixed_J3_J7` | `configs/dof/worm_pic_5dof_j3_j7.yaml` |
| `FR3_fixed_J3_J5_J7` | `configs/dof/worm_pic_4dof.yaml` |
| `FR3_fixed_J3_J5_J6_J7` | `configs/dof/worm_pic_3dof.yaml` |
| `FR3_link3_mass_150pct` | `configs/mass/worm_pic_link3_mass_150pct.yaml` |
| `FR3_link3_mass_200pct` | `configs/mass/worm_pic_link3_mass_200pct.yaml` |
| `FR3_link7_mass_150pct` | `configs/mass/worm_pic_link7_mass_150pct.yaml` |
| `FR3_link7_mass_200pct` | `configs/mass/worm_pic_link7_mass_200pct.yaml` |

All settings retain the original Worm values: `time_rate: demo`,
`ds_speed_factor: 1.0`, embedding, explicit single-run TCP target, PIC gains,
moment limits by original axis, `dt: 0.001`, `T_max: 20.0`,
`multistart.seed: 0`, `sim.seed: 73`, 32 IK candidates and `max_sigma_margin`.
All 100 indexed TCP starts match the existing Worm/PIC and Worm/Servo batches.
They are Worm starts, rather than the Five Obstacle starts.

The reference and density variants also retain the exact original seven-angle
single-run joint vector. The four locked arms instead use `q_start_rad: null`
to solve that same TCP target with their own available axes and lock angles.
Batches select the same TCP targets per index for all nine models. Within
the mass series, the shared reference IK also guarantees identical initial
joint vectors per indexed start. Different locked arms need their own joint
vectors; a target that their IK cannot reach remains a recorded failed start.
Simulation and nominal gravity compensation use the selected density MJCF,
with `plant.mass_scale: 1.0` and the original PIC settings.

Commands for later 100-run execution (new output directories required):

```bash
python -B main.py configs/dof/worm_pic_7dof.yaml --runs 100 --output results/worm_pic_7dof_100
python -B main.py configs/dof/worm_pic_5dof_j3_j5.yaml --runs 100 --output results/worm_pic_5dof_j3_j5_100
python -B main.py configs/dof/worm_pic_5dof_j3_j7.yaml --runs 100 --output results/worm_pic_5dof_j3_j7_100
python -B main.py configs/dof/worm_pic_4dof.yaml --runs 100 --output results/worm_pic_4dof_100
python -B main.py configs/dof/worm_pic_3dof.yaml --runs 100 --output results/worm_pic_3dof_100
python -B main.py configs/mass/worm_pic_link3_mass_150pct.yaml --runs 100 --output results/worm_pic_link3_mass_150pct_100
python -B main.py configs/mass/worm_pic_link3_mass_200pct.yaml --runs 100 --output results/worm_pic_link3_mass_200pct_100
python -B main.py configs/mass/worm_pic_link7_mass_150pct.yaml --runs 100 --output results/worm_pic_link7_mass_150pct_100
python -B main.py configs/mass/worm_pic_link7_mass_200pct.yaml --runs 100 --output results/worm_pic_link7_mass_200pct_100
```

Verification on 2026-11-18: the configuration comparison passes for all nine
Worm configurations and all ten existing Five Obstacle configurations. Each
new Worm configuration completed a 4-ms single run and a two-start batch.
The saved initial joint vectors match the reference exactly for all four
mass variants, both for single runs and the two sampled batch starts.
These are function checks; the 100-run experiments have not been executed.
The validation report is saved at
`results/worm_configs_smoke_lakrkn3z/validation.json`.

## Local checks

```bash
python -B -m tools.generate_fr3_variants --check
python -B -m unittest tests.test_robot_models -v
python -B -m unittest discover -s tests -v
python -B main.py configs/dof/five_obstacle_pic_3dof.yaml --check
python -B main.py configs/mass/five_obstacle_pic_link7_mass_200pct.yaml --set sim.T_max=0.02 --output results/fr3_link7_mass_200pct_smoke
```

The model tests compile every arm, verify midpoint locks, removed joints and
actuators, shortened keyframes and original free-axis limits. Across 25 poses
per fixed variant, they compare TCP position/orientation, the reduced Jacobian,
gravity and reduced mass matrix with the full FR3 at the same locked angles.
An independent finite-difference check verifies the position Jacobian.
Density tests check the exact XML change scope, compiled added mass, doubled
gravity changes, and nominal compensation equality. Public-run tests exercise
PIC and Servo single runs and two-start batches for every registered model,
at 4 ms duration; these validate functionality, not 20-second task success.

Verification on 2026-11-17: all 61 tests passed in 67.774 s, including exact
cross-variant start-joint equality after introducing the shared reference IK.
All 49 FR3 asset sizes/hashes match the asset manifest, generated MJCFs match
their generator, and `git diff --check` passes. The 3-DoF `--check` gives a
start FK error of 7.45594674e-10 m. The additional 20-ms Link-7 +100% PIC run
completed at `results/fr3_link7_mass_200pct_smoke/`; its short duration does
not reach the task goal (`success: false`). The common-pose diagnostic is
saved at `results/fr3_mass_comparison.json`.
