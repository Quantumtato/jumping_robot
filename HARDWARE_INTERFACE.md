# Hardware Interface Checklist (as of v9 / direct-continue policy)

Everything the deployed actor network needs at runtime, and every piece of
sim-side machinery you must reimplement on the robot. Keep this in sync with
`mjlab_tasks/jumping_robot_balance/mdp/observations.py` — observation ORDER
and normalization must match training exactly or the policy sees garbage.

## Control loop basics

- **Rate: 100 Hz** (sim decimation x timestep). The policy expects fresh
  observations and issues actions every 10 ms.
- **Observation normalization is baked into the checkpoint** (running
  mean/std `obs_normalizer`). Export it with the network and apply it before
  inference — the raw normalizations listed below happen *before* that layer.
- All noise ranges below are what training injected; the real sensors just
  need to be at least that good.

## Actor observation vector (exact order)

| # | Term | Dim | What hardware must provide |
|---|------|-----|----------------------------|
| 1 | `projected_gravity` | 3 | Gravity direction in body frame from the IMU attitude filter (unit vector; trained with +/-0.01 noise). |
| 2 | `base_ang_vel` | 3 | Raw gyro rates, body frame, rad/s (+/-0.05 noise). |
| 3 | `flywheel_vel_norm` | 2 | Flywheel speeds (x wheel, y wheel) from encoders, divided by `MAX_FLYWHEEL_SPEED_RAD_S`. |
| 4 | `linear_pos_norm` | 1 | Leg linear position mapped to [-1, 1] over `[LINEAR_RANGE_MIN_M, LINEAR_RANGE_MAX_M]`. |
| 5 | `linear_vel_norm` | 1 | Leg linear velocity / `LINEAR_MAX_SPEED_M_S`. |
| 6 | `last_action` | 3 | The previous step's 3 motor actions (policy outputs 1-3), fed back. Bookkeeping only — no sensor. |
| 7 | `foot_contact` | 1 | Binary foot-ground contact (switch or current sensing), 0/1. |
| 8 | `jump_command` | 1 | Jump-active flag from the onboard jump state machine (see below). |
| 9 | `flight_phase_history` | 8 x 7 = 56 | 8-step (80 ms) ring buffer of: linear_pos_norm, linear_vel_norm, last 3 motor actions, foot_contact, jump-active flag. |
| 10 | `airborne_time` | 1 | Seconds since takeoff while airborne, clamped to 1.0 s max, 0 when grounded. From the jump state machine. |
| 11 | `planar_velocity_command` | 2 | The user velocity command rotated into the robot's **heading (yaw) frame**, m/s. See yaw note below. |
| 12 | `imu_specific_force_history` | 64 x 3 = 192 | 64-step (640 ms) ring buffer of body-frame accelerometer specific force, divided by `3g` (29.43 m/s^2), clipped to +/-4. Remember: in freefall this reads ~0, on the ground it reads ~+1g up — that's what the accelerometer physically measures, no gravity compensation. |
| 13 | `velocity_estimate` (NEW in v9) | 2 | Hop-averaged planar velocity estimate in the heading frame, divided by 0.40 m/s, clipped +/-4. See estimator spec below. |

History buffers must be zero-filled (or steady-state-filled) at power-on;
in sim they reset to repeated current values on episode reset.

## Actions (4 outputs)

1-2. Flywheel x / y targets — scaled per `mdp/actions.py` (speed/torque
   targets for the flywheel motors).
3. Leg linear position target — rate-limited in sim by `max_target_speed_m_s`
   and tracked by a PD (`kp_n_m`, `kd_n_s_m`, effort-clamped); replicate the
   same PD + rate limit on hardware or the sim-to-real gap will be large.
   **Gain scheduling (NEW in v10):** the leg PD switches gains with the jump
   state machine. While the jump-active flag is set: **kp = 4800 N/m,
   kd = 30 N·s/m**. Otherwise (stance, balancing, landing absorption —
   the flag clears at touchdown): **kp = 1600 N/m, kd = 100 N·s/m**.
   Effort clamp stays ±120 N in both modes. The hardware leg controller
   must implement the same two-mode switch, keyed off the same jump-active
   flag it already maintains for obs #8 — without it, push-off force is
   mostly cancelled by damping and the robot cannot jump past ~5 cm.
4. **Jump request** — fire a jump when raw output > **0.5**. Subject to the
   state-machine gating below. (During v8/v9 training the environment also
   force-triggers jumps every 1-2 s; see "jump cadence" note.)

## Things you must build/replicate on the robot

### 1. Yaw integrator (you have this)
Needed to rotate the world-frame velocity command into the heading frame
(obs #11) and for the velocity estimate frame (obs #13). Drift is tolerable:
nothing observes absolute yaw, only the command direction rotates with the
error — slow drift just slowly bends "forward."
**Shortcut:** if you command from a joystick in robot-relative terms, you can
skip world-frame entirely and feed the stick vector straight in as obs #11.

### 2. Velocity estimator (NEW for v9) — the "gyro/accel dead-reckoner"
Sim definition it must approximate: an exponential moving average of true
planar world velocity with **time constant tau = 1.0 s**, rotated into the
heading frame. Hardware recipe: rotate accelerometer specific force into the
world frame using the attitude estimate, add gravity back, integrate to
velocity, and run the same 1.0 s EMA; optionally re-anchor per landing
(zero-velocity-ish updates at stance) to kill drift. Training added +/-0.04
m/s uniform noise, so the estimator only needs to be that good over a
~1-2 s horizon.

### 3. Jump state machine
The env owns this in sim; the robot must own it in deployment. It provides
obs #8 and #10 and gates action #4:
- Jump-active flag: set on trigger, cleared on landing+stand (sim clears it
  when the jump routine finishes).
- Airborne time: stopwatch started at loss of contact, clamped at 1.0 s.
- Trigger gating: honor a jump request only when (a) no jump active,
  (b) foot in contact, (c) >= **0.25 s cooldown** since the last jump ended.
   - **Jump cadence (UPDATED in v14 — policy owns timing):** v14 anneals the
     env's forced triggers to zero during training, so the trained policy is
     responsible for initiating every jump via action #4. The hardware no
     longer needs the speed-dependent auto-trigger as a primary mechanism.
     Keep it only as a **watchdog fallback**: if a nonzero velocity command
     has been active for > ~2 s with no policy-initiated jump, fire one
     (mirrors the sim's anti-scoot gate). Keep the 0.25 s post-landing
     cooldown on policy jump requests.
     (Pre-v14 checkpoints still need the v12 speed-dependent auto-trigger:
     interval interpolating from (1.0, 2.0) s at zero command to (0.8, 1.2) s
     at the speed cap.)

### 4. Foot contact sensing
A clean binary signal. It feeds obs #7, the history buffer, the jump state
machine, and landing detection. Debounce it — sim contact is crisp.

## Constants to pull from `robot_cfg.py` / `commands.py` (don't hardcode stale values)

- `MAX_FLYWHEEL_SPEED_RAD_S`, `LINEAR_RANGE_MIN_M/MAX_M`,
  `LINEAR_MAX_SPEED_M_S` — normalization denominators.
- `PLANAR_VELOCITY_SCALE_M_S = 0.40` — command + velocity-estimate scale.
- `IMU_ACCEL_SCALE_M_S2 = 3g` — accelerometer scale.
- Control PD gains and effort clamps in `mdp/actions.py`; jump-window leg
  gains `DIRECT_CONTINUE_JUMP_KP_N_M` / `DIRECT_CONTINUE_JUMP_KD_N_S_M` in
  `env_cfg.py` (4800 / 30 as of v10).

## Known sim-to-real risks to watch

- **Flywheel authority / thermal duty (confirmed against hardware, v17):**
  the flywheel motor is a TQ IML 38x08 with **0.33 N·m PEAK** torque — the
  sim's `forcerange` is already at that peak, so there is no headroom to
  raise sim limits; more authority requires the planned v2 flywheel upgrade.
  Worse, trained policies rail the torque clamp ~95% of flight time and
  flight is ~25-35% of wall time, giving an RMS torque of roughly
  **0.33 × sqrt(0.30) ≈ 0.18 N·m sustained** during continuous hopping —
  likely above the motor's continuous rating (typical frameless continuous
  is ~1/3 of peak). Mitigations:
  - Hardware: run an i²t accumulator (∫torque² dt against the motor thermal
    time constant) and derate the torque clamp as it approaches the limit;
    brief peak bursts are fine, sustained hopping is not.
  - Training (future run): add a flywheel-effort cost so the policy spends
    torque only when attitude demands it, cutting RMS heat at equal peak.
  - Fill in the actual continuous torque rating from the IML 38x08
    datasheet to replace the 1/3-of-peak estimate above.
- **Accelerometer scale/bias**: obs #12 is the actor's only raw inertial
  velocity cue; a miscalibrated scale factor shifts everything. Calibrate
  against gravity at rest (+1 g up) and freefall (~0) — both cases appear
  constantly during hopping.
- **Latency**: 100 Hz training assumed observation-to-action within one tick.
  Measure end-to-end latency; if it exceeds ~10 ms consistently, we should
  retrain with an action-delay model.
- **Contact chatter** at landing can double-trigger the state machine —
  debounce, and rate-limit landing events to one per airborne phase.
- **Leg ballscrew rotor (NEW in v13, CAD values — replace via sysid):** the
  sim now models the leg motor's spinning parts explicitly: rotor inertia
  **1.863023e-5 kg·m²** (18.63023 kg·mm², from CAD), screw lead **40 mm/rev**
  (coupling 157.08 rad per meter of leg travel), coupled to the slide by an
  equality constraint. This gives both the reflected inertia on the leg
  (~0.46 kg equivalent) and the yaw reaction torque on the base during leg
  acceleration — the policy trained after v13 may actively use leg-driven
  yaw kicks for heading control, so the real screw direction must match the
  sim sign convention (or retrain with the sign flipped to match hardware).
  Run the actuator sysid chirp fit and update `LEG_ROTOR_INERTIA_KG_M2` in
  `robot_cfg.py` when hardware numbers exist.
- **Foot torsional friction (NEW in v13, estimated — sysid eventually):**
  foot contact is now `condim=4` with torsional coefficient **0.005 m**
  (≈ friction 1.0 × ~5 mm rubber contact-patch radius) so the grounded foot
  absorbs part of the screw reaction torque like real rubber does. If the
  real foot pivots more/less freely than this, yaw authority during stance
  will mismatch — measure by twisting the loaded foot on the real surface.
