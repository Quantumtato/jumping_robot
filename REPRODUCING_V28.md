# Reproducing the v28 model from a cold start

The only lineage whose **deterministic** policy ever jumped, steered, and
idled on command is v19 → v21b → v22 (cold start) polished by v23 → v28
(warm continuation). The cold starts v29–v31 that modified this recipe all
produced policies that look alive in training metrics but are inert or
command-blind at inference (verified by headless eval, Aug 28 2026). This
document pins the recipe.

## The two-stage pipeline

### Stage 1 — cold start (`--stage direct-pipeline`, ~12,000 iters, ~7 h)

```bash
python mjlab_tasks/jumping_robot_balance/scripts/train.py \
  --stage direct-pipeline --num-envs 2048 --max-iterations 12000 \
  --run-name direct_pipeline_vXX
```

Clock-scheduled curriculum (proven timings, do not progress-gate):

| Iteration | Event |
|---|---|
| 0–1,500 | Balance only (forced hops off, commands zero) |
| 1,500 | Forced hop cadence begins |
| 1,750 | Velocity commands live; speed ladder 0.05 → 0.10 → 0.15 → 0.25 → 0.40 |
| 5,000–7,000 | Trigger handover: forced cadence anneals to zero; self-trigger bonus (300) keeps the jump channel alive |
| 7,000–9,000 | Push ramp to 0.5 scale |
| 9,000–12,000 | Consolidation |

Expected milestones: episode length recovers past ~1,000 by ~3,000 iters;
first speed promotion (0.05 → 0.10) usually before the handover; healthy
self-trigger rate (~10+/step-scale EMA) surviving after iteration 7,000.

### Stage 2 — warm polish (`--stage speed-continue`, from Stage 1 final)

What v23–v28 actually contributed (headless eval, v22 vs v28): **idle
discipline** (v22 hops ~21×/20 s at zero command; v28 idles) and **fewer
falls**. Mechanisms that mattered: hard action-std clamp (~0.40) on load,
zero entropy bonus, higher action-rate penalty, and gradual removal of
scaffold income (reward diet). Steering was already learned in Stage 1;
Stage 2 must not be asked to create skills, only to prune them.

## Reward forms Stage 1 depends on (v33 restorations)

These live in `mdp/rewards.py` / `mdp/commands.py` and were reverted to
era-correct forms on Aug 29 2026 after v29–v31 modified them in place:

- **`hop_displacement_along_command`**: raw signed meters along command,
  weight 10,000. No cap (v24), no tent (v29), no normalization (v31).
  A cold start needs an unaimed hop that drifts the right way to earn
  more than one that doesn't — that staircase IS the aim gradient.
- **`hop_displacement_perpendicular_l1`**: raw meters, weight −3,000.
- **`gated_*_velocity_tracking`**: absolute Gaussian kernel
  `exp(-err² / 0.04)` (σ = 0.2 m/s), income-gated 2 s after jump
  activity. The v30 command-relative kernel removes the partial credit
  noise-driven exploration climbs.
- **`takeoff_velocity_along_command`**: uncapped signed payout, weight 500.
- **Promotion floor**: `speed_curriculum_error_floor_m_s = 0.08`. The easy
  first promotion off the 0.05 rung is part of the recipe — it moves
  learning to rungs where displacement income differentiates aim.

Anti-exploit machinery belongs in Stage 2 (mechanical std clamp, diet),
not priced into Stage 1 rewards.

## Acceptance test — never trust training metrics alone

Training-time trigger/velocity metrics include exploration noise; v29–v31
looked healthy while their deterministic policies did nothing. Run the
headless battery (`/home/ubuntu/eval_policy.py`) on checkpoints during and
after training (deterministic mean, forced commands, 4 phases):

```bash
python eval_policy.py --task direct-pipeline --checkpoint model_XXXX.pt
```

Pass criteria (final Stage 1 checkpoint, 20 s phases):

- `cmd zero`: mean velocity ≈ 0 (idle hopping OK at this stage)
- `cmd ±0.15`: mean velocity within ~±0.05 m/s of command on-axis,
  cross-axis < 0.03 m/s, self-triggered jumps ≥ 15/env, falls < 1/env

The `v33_watcher.sh` pattern (eval every new checkpoint, appended to
`v33_eval.log`) makes this continuous during training.

## Known open questions

- The speed ladder itself may be unnecessary or harmful (learning "slow"
  may be harder than learning "go"); untested alternative: full command
  range from the start.
- Idle discipline might be teachable in Stage 1 with a stationary-hop
  penalty rather than requiring a Stage 2 diet.
