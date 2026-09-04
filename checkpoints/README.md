# Model checkpoints

RSL-RL PPO checkpoints for the speed-continue (wide-speed navigation) stage.
Load with `train.py --stage speed-continue --resume-from <file>` or evaluate
with `play.py` / `play_train_env.py --checkpoint-file <file>`.

## v40_model_9800_HARDWARE_CANDIDATE.pt  <-- LATEST / FROZEN FOR HARDWARE
Final v40 checkpoint (2026-09-03, iter 9,800 of 12,000; metrics flat from
~7,000 so effectively converged). Trained under the full hardware gauntlet:
foot friction randomization, full-scale pushes, 0-20 ms randomized actor
sensor latency, full-range 0-0.55 m/s commands. Deterministic eval vs v38
under identical conditions: falls per env halved (0.9-1.5 vs 1.75-3.06 per
20 s), symmetric steering in all four directions, near-stationary at zero
command. Known bias: realizes ~0.23 m/s for a 0.15 m/s command (consistent
~50% overshoot, all directions) -- calibrate command scaling at deployment.

## v39_model_800_v40_seed.pt
The checkpoint the v40 hardware-prep run was warm-started from
(2026-09-03, superseded by v40 above). Lineage: v28 -> v38 (ladderless 0-0.55 m/s speed commands,
8,000 iters) -> v39 (+ foot friction randomization sliding x0.6-1.4 /
torsional x0.5-1.5, + full-scale push ramp, + settled_velocity_error
metric; killed at iter 800 to add sensor latency). v40 continues from
this file with 0-20 ms randomized actor sensor latency.

## v38_final_model_7999.pt
Final checkpoint of the last COMPLETED run (v38, 2026-09-02): full-range
speed commands with no curriculum ladder, pushes at 0.5 scale, fixed
friction, no sensor latency. Settled tracking baseline before the
hardware-robustness additions.
