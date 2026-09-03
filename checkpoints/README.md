# Model checkpoints

RSL-RL PPO checkpoints for the speed-continue (wide-speed navigation) stage.
Load with `train.py --stage speed-continue --resume-from <file>` or evaluate
with `play.py` / `play_train_env.py --checkpoint-file <file>`.

## v39_model_800_v40_seed.pt  <-- LATEST
The checkpoint the v40 hardware-prep run was warm-started from
(2026-09-03). Lineage: v28 -> v38 (ladderless 0-0.55 m/s speed commands,
8,000 iters) -> v39 (+ foot friction randomization sliding x0.6-1.4 /
torsional x0.5-1.5, + full-scale push ramp, + settled_velocity_error
metric; killed at iter 800 to add sensor latency). v40 continues from
this file with 0-20 ms randomized actor sensor latency.

## v38_final_model_7999.pt
Final checkpoint of the last COMPLETED run (v38, 2026-09-02): full-range
speed commands with no curriculum ladder, pushes at 0.5 scale, fixed
friction, no sensor latency. Settled tracking baseline before the
hardware-robustness additions.
