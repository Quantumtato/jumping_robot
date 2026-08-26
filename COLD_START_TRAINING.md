# Cold-Start Training Playbook (reproduces the v8 policy)

How to train the jumping robot from **random weights** to the v8 level:
stable balance, reliable 4-8 cm hops, live velocity commands with correct
speed magnitude, ~0.4 falls/episode. This documents the `direct-pipeline`
stage, validated by run `2026-08-16_18-48-31_direct_pipeline_v8`.

Later stages (v9 `direct-continue` and beyond) are intentionally NOT covered
here until they're validated.

## TL;DR — the one command

```bash
cd ~/workspace/jumping_robot
# optional 3-iteration smoke test first (see "Smoke test" below), then:
nohup bash scripts/aws/run_then_stop_instance.sh \
  ~/.local/bin/uv run --with mjlab==1.5.3 \
  python -u mjlab_tasks/jumping_robot_balance/scripts/train.py \
  --stage direct-pipeline --num-envs 2048 --max-iterations 8000 \
  --run-name my_cold_start > ~/my_cold_start.log 2>&1 &
```

~7 hours wall clock on the usual EC2 GPU instance. The wrapper stops the
instance when training finishes. No checkpoint, no warm start, no manual
stage transitions — everything below happens inside this single run.

Requirements: repo at `~/workspace/jumping_robot`, `uv` installed,
`mjlab==1.5.3` (pinned — newer versions have not been checked against the
custom runners), wandb configured or offline.

## What the run does internally

Three scheduled phases, switched by `env.common_step_counter`
(32 steps per learning iteration; constants in `env_cfg.py`):

| Iterations | Phase | Commands | Jumps |
|---|---|---|---|
| 0 - 1500 | Balance | zero | none |
| 1500 - 1750 | Hop acclimation | zero | env-forced every 1-2 s |
| 1750 - 8000 | Locomotion | sampled, live | env-forced every 1-2 s (policy may request extras) |

Two curricula run inside phase 3:
- **Height ladder** (`jump_commands.py`): 4 -> 6 -> 8 cm, advances when the
  EMA of landing-recovery success clears 0.6 with a 60 s minimum dwell.
- **Speed caps** (`commands.py`): 0.15 -> 0.25 -> 0.40 m/s, advances when the
  EMA of hop-averaged tracking error (moving envs) drops below threshold.

## Expected milestones (reference: direct_pipeline_v8)

If a cold start deviates wildly from this table, something regressed.

| Iteration | What should happen | Reference values |
|---|---|---|
| 0 | Chaos | reward -98, falls ~51/ep |
| ~120 | Balance learned | falls ~2.5/ep, reward ~+35, ep len ~800 |
| 120 - 1500 | Balance plateau (normal — nothing new to learn) | reward oscillates 25-45 |
| 1500 | Hops start; brief fall spike | falls ~4/ep for ~50 iters, apex jumps to ~4.8 cm |
| ~1880 | Height curriculum -> 6 cm | console line "Jump curriculum advanced to level 1" |
| ~2070 | Height curriculum -> 8 cm (ceiling) | "advanced to level 2" |
| 1750 | Commands turn on | mean cmd speed ~0.09 m/s, reward jumps to ~150 |
| 2000 - 5500 | Long grind: tracking error slowly falls 0.135 -> 0.115 | falls decay 0.9 -> 0.4/ep, ep len -> ~1700 |
| ~5500 | Speed cap -> 0.25 m/s | "Speed curriculum advanced to level 1" (timing varies; v8 hit 5554) |
| 8000 (end) | | vel error ~0.17 m/s, falls ~0.4/ep, apex ~6.3 cm, reward ~175 |

Known limitation of the v8 endpoint: speed **magnitude** tracks but
directional aim is weak (flywheel flight authority is the bottleneck), and
the cap typically does not reach 0.40. That is the expected stopping point
of this playbook, not a failure.

## Smoke test before committing GPU-hours

```bash
cd ~/workspace/jumping_robot
~/.local/bin/uv run --with mjlab==1.5.3 \
  python -u mjlab_tasks/jumping_robot_balance/scripts/train.py \
  --stage direct-pipeline --num-envs 64 --max-iterations 3 \
  --run-name smoke > ~/smoke.log 2>&1; echo EXIT_$?
grep -n -e Traceback -e ValueError -e NaN ~/smoke.log
```

Pass = exit 0, 3 iterations complete, no errors. (`nan_guard` termination
stat reading `False`/`0.0` is normal.) Delete the smoke log dir and its
wandb run before launching the real thing so log listings stay clean.

## Monitoring a live run

```bash
grep -e "Learning iteration" -e "curriculum advanced" -e Traceback ~/my_cold_start.log | tail
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader   # expect >90%
```

Key wandb/console metrics: `Metrics/jump/apex_height`,
`Metrics/jump/curriculum_target_height`,
`Metrics/planar_velocity/average_velocity_error`,
`Metrics/planar_velocity/speed_curriculum_cap`,
`Episode_Termination/fell`, `Episode_Reward/directional_progress`.

Viewer on a checkpoint mid-run (any `model_N.pt` in the run's log dir):

```bash
cd ~/workspace/jumping_robot
nohup ~/.local/bin/uv run --with mjlab==1.5.3 \
  python -u mjlab_tasks/jumping_robot_balance/scripts/play.py \
  --agent trained --viewer viser --navigation \
  --checkpoint-file logs/rsl_rl/jumping_robot_balance/<run_dir>/model_<N>.pt \
  > ~/play.log 2>&1 &
# locally: ssh -N -L 8080:localhost:8080 ... then open http://localhost:8080
```

## Design invariants — the hard-won lessons baked into this stage

Each of these was learned from a failed run. Do not undo them casually.

1. **Jumps are env-forced for the entire run** (`phase_schedule_steps` with
   model-only step set to never). v4 gave the policy the option not to jump;
   it learned to never jump and scoot instead, and never escaped. The policy
   still has a jump-request channel (action 4, threshold 0.5) that can add
   jumps — it just can't withhold them.
2. **Velocity commands start with locomotion, not after it** (250-iteration
   acclimation only). v6/v7 trained hop-in-place for 2000 iterations before
   showing the first command; the hop-in-place attractor was then too deep
   to leave.
3. **The anti-scoot penalty has a 0.4 s post-landing grace window**
   (`grounded_planar_speed` in `rewards.py`). Without it (v7), every landing
   with legitimate forward momentum was taxed and the optimal policy was to
   not move.
4. **The tracking reward is a wide kernel (sigma 0.2 m/s) plus a linear
   directional-progress term.** A sharp kernel alone (v7, sigma 0.1) has no
   gradient until error is already small — a hopping-in-place policy earns
   nothing and learns nothing from it.
5. **Commands are observed in the robot's heading (yaw) frame**
   (`observations.py`). World-frame commands become unobservable to a
   yaw-blind actor once heading drifts (v2 lesson).
6. **The actor's inertial sensing is a 64-step IMU specific-force history**,
   accelerometer-realistic (reads ~0 in freefall, +1 g standing). Do not
   "helpfully" feed it true velocity — that breaks the hardware contract
   (see `HARDWARE_INTERFACE.md`).
7. **Observation terms are append-only across stages.** Warm-start runners
   copy trained input weights into the leading columns and zero-init the
   trailing ones; reordering or inserting mid-list silently corrupts every
   older checkpoint.
8. **Reset-time NaN guards** in `commands.py` (`dt <= 0` skip,
   `nan_to_num` on robot state at resample) are load-bearing; removing them
   crashes training with NaN observations within the first iterations.

## Where the knobs live

| File | Owns |
|---|---|
| `mjlab_tasks/jumping_robot_balance/env_cfg.py` | Phase boundaries, which curricula/stages are active, stage flags |
| `mdp/rewards.py` | All reward terms and weights (navigation block ~line 400) |
| `mdp/commands.py` | Velocity command sampling, speed curriculum, IMU + hop-averaged velocity computation |
| `mdp/jump_commands.py` | Jump trigger logic, height curriculum, jump metrics |
| `mdp/observations.py` | Actor/critic observation layouts (order matters!) |
| `rl/stage_transition_runner.py` | Warm-start/checkpoint-surgery runners |
| `scripts/train.py` | Stage names, CLI, hyperparameter overrides per stage |

## Why the older stages are not the path (short history)

- v1-v3 (`navigation` stage, warm-started from balance checkpoints): added
  IMU history, hop-averaged rewards, heading-frame commands — worked, but
  inherited whatever the balance checkpoint baked in.
- v4: model-commanded jumps too early -> exploration collapse (never jumped).
- v5: anti-scoot penalty added (no grace window yet).
- v6 (`full-pipeline`): first successful from-scratch run; proved the
  single-run curriculum but hop-in-place phase was too long and the height
  ladder too short.
- v7 (`pipeline-continue`): taller targets, but the ungated anti-scoot
  penalty + sharp tracking kernel meant velocity tracking never emerged.
- **v8 (`direct-pipeline`): the stage this playbook documents.**

The old stage registrations still exist in `task_registry.py` for checkpoint
compatibility; there is no reason to train them from scratch again.
