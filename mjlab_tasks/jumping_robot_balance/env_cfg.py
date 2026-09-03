"""Environment configuration for jumping robot balance in mjlab."""

from __future__ import annotations

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from mjlab_tasks.jumping_robot_balance.mdp import (
    build_action_terms,
    build_disturbance_commands,
    build_free_hop_action_terms,
    build_height_action_terms,
    build_height_commands,
    build_height_robustness_events,
    build_jump_stage_one_action_terms,
    build_jump_commands,
    build_navigation_action_terms,
    build_observation_groups,
    NAVIGATION_TARGET_HEIGHTS_M,
    build_planar_velocity_commands,
    build_randomization_events,
    build_reward_terms,
    build_strong_robustness_events,
    build_termination_terms,
    build_warm_start_jump_events,
)
from mjlab_tasks.jumping_robot_balance.robot_cfg import (
    BASE_BODY_NAME,
    EPISODE_LENGTH_S,
    ROBOT_ENTITY_NAME,
    SIM_DECIMATION,
    SIM_TIMESTEP_S,
    make_robot_entity_cfg,
)

# Full-pipeline phase boundaries, in env.common_step_counter units (one unit
# per control step; PPO collects num_steps_per_env=32 of them per learning
# iteration -- see rl/ppo_cfg.py). Phase 1 (balance) runs until hop start,
# phase 2 (env-forced hops, zero velocity commands) until model-only, and
# phase 3 (model-triggered jumps, sampled velocity commands) afterward.
_PPO_STEPS_PER_ITERATION = 32
PIPELINE_HOP_START_STEP = 1_500 * _PPO_STEPS_PER_ITERATION
PIPELINE_MODEL_ONLY_STEP = 3_500 * _PPO_STEPS_PER_ITERATION

# Continuation schedule (resume from a trained full-pipeline checkpoint): no
# balance phase, a forced-hop phase long enough to climb the extended height
# ladder, then model-owned locomotion with sampled velocity commands.
CONTINUE_HOP_START_STEP = 0
CONTINUE_MODEL_ONLY_STEP = 1_500 * _PPO_STEPS_PER_ITERATION

# Direct-locomotion schedule (from scratch): balance, then a short hop
# acclimation with zero commands, then velocity commands sampled under a
# success-gated speed curriculum. Jumps stay env-forced for the whole run so
# the hop-in-place attractor never forms and the no-jump collapse (v4) is
# impossible; direction control is learned together with hopping (v8).
DIRECT_HOP_START_STEP = 1_500 * _PPO_STEPS_PER_ITERATION
DIRECT_COMMAND_START_STEP = 1_750 * _PPO_STEPS_PER_ITERATION
_NEVER_STEP = 10**9
DIRECT_SPEED_CURRICULUM_CAPS_M_S = (0.15, 0.25, 0.40)
# v16: start the ladder at a speed the settled gait (7 cm hops @ ~0.5 Hz)
# can genuinely track, so the relative gate can engage and ratchet. The
# 0.15 starting cap was never reachable for that gait, which starved the
# tracking gradient (v15 forensics).
V16_SPEED_CURRICULUM_CAPS_M_S = (0.05, 0.10, 0.15, 0.25, 0.40)
# Height ladder stays at the proven rungs; taller jumps come after direction
# control works, not alongside it.
DIRECT_TARGET_HEIGHTS_M = (0.04, 0.06, 0.08)

# Direct-continue (v9): resumes a trained direct-pipeline policy, adds the
# actor velocity-estimate observation, re-opens the tall height ladder
# (longer flight = more flywheel steering time per hop), and swaps the speed
# gate to a relative threshold that hop-in-place can't clear. Hops stay
# env-forced and commands are live from step zero.
DIRECT_CONTINUE_SPEED_GATE_RATIO = 0.6
# v9b: start the ladder at the 8 cm rung v8 already mastered and gate
# advancement on upright landings rather than stand-still recovery -- with
# commands live from step zero the robot never stands still, so the recovery
# gate can never fire (v9 stalled at 4 cm because of this).
# v12: cadence over height -- more, lower hops give more push-offs per
# second, and every push-off is a chance to change planar momentum. The
# 12.5/15 cm rungs proved reachable in v10/v11 but cost control bandwidth.
# Apex is now FOOT-tracked (baseline = grounded foot at trigger), which
# reads base rise + mid-air tuck. With a 15.2 cm leg stroke a typical tuck
# adds ~4-6 cm, so these rungs correspond to roughly 8-10 cm of base rise
# (the original cadence-over-height intent).
DIRECT_CONTINUE_TARGET_HEIGHTS_M = (0.12, 0.15)
# Forced-jump retrigger interval at zero command stays (1.0, 2.0) s (the
# full-pipeline default); at the current speed cap it shrinks to this.
DIRECT_CONTINUE_FAST_RESAMPLING_S = (0.8, 1.2)
# v14 trigger handover: forced jumps at full cadence for the first 1000
# iterations, intervals stretching up to 4x over the next 2000, then off --
# the policy's jump-request channel owns timing from iteration 3000 on.
V14_TRIGGER_ANNEAL_STEPS = (
    1_000 * _PPO_STEPS_PER_ITERATION,
    3_000 * _PPO_STEPS_PER_ITERATION,
)
# v19 cold start: the direct-pipeline stage now carries every fix proven on
# the continue lineage (velocity-estimate obs, jump-window PD gains, 80 deg
# fall angle, foot-tracked height ladder, v16 speed ladder + relative gate,
# v18 displacement rewards) plus two schedules of its own:
# - Push disturbances follow the pipeline phases: full strength while the
#   policy learns balance, OFF from hop start (a 0.25 m/s shove is 2.5x the
#   initial command cap and drowns the per-hop displacement gradient), then
#   linearly ramped back in to half strength late for deployment robustness.
# - The forced-trigger anneal is shifted to fit the cold-start timeline:
#   full cadence through balance and early locomotion, handover complete by
#   iteration 7000.
# v20: v19 stacked its two hardest transitions -- jump ownership (anneal
# 4000-6000) and returning pushes (5000-7000) -- and the policy quit jumping
# (0.03-0.16 self-triggers/ep at the end vs ~10 in the warm lineage).
# De-overlapped: ~3250 iterations of commanded locomotion practice before the
# anneal starts, and pushes return only after the handover completes.
V19_PUSH_RAMP_START_STEP = 7_000 * _PPO_STEPS_PER_ITERATION
V19_PUSH_RAMP_END_STEP = 9_000 * _PPO_STEPS_PER_ITERATION
V19_PUSH_RAMP_FINAL_SCALE = 0.5
V19_TRIGGER_ANNEAL_STEPS = (
    5_000 * _PPO_STEPS_PER_ITERATION,
    7_000 * _PPO_STEPS_PER_ITERATION,
)
# v10: stiff, lightly damped leg PD during the jump window only. The v9b
# diagnostic showed the balance-tuned kd=100 cancels up to 169 N of drive
# during push-off (clamp is 120 N) while the applied force never saturates;
# with kp=4800/kd=30 the 120 N clamp becomes the operative limit and a 15 cm
# apex needs only ~2.3 cm of stroke. Stance and landings (jump command clears
# at touchdown) keep the original balance-tuned gains.
DIRECT_CONTINUE_JUMP_KP_N_M = 4800.0
DIRECT_CONTINUE_JUMP_KD_N_S_M = 30.0
# v11: let the policy attempt recovery from extreme tilt instead of dying at
# the balance-era 30 deg. The -10k fall penalty moves with the termination
# (they must stay aligned; see rewards.py); between 30 and 80 deg the smooth
# tilt_error quadratic and the lost upright bonus supply the recovery
# gradient. The landing-upright curriculum gate keeps its own 30 deg
# criterion -- that is a success metric, not a termination.
DIRECT_CONTINUE_FALL_ANGLE_DEG = 80.0
# v22 speed-ladder continuation: resumes the v21b policy, which already owns
# its jump timing (the v21 self-trigger bonus survived the anneal handover at
# ~15 jumps/min). The whole run is spent climbing the speed ladder from the
# 0.10 m/s cap v21b earned:
# - No forced triggers at all; the handover already happened, so re-forcing
#   cadence would only mask the policy's own timing decisions.
# - Pushes hold constant at the half-strength deployment level v21b finished
#   with (no ramp), keeping the disturbance noise floor fixed while tracking
#   improves against it.
# - The ladder starts at 0.10 so no training time is spent re-earning it;
#   promotion still requires the relative gate (error EMA below
#   max(0.08, 0.6 x moving-command EMA)), i.e. genuinely better tracking.
# v23: v22 finished at the 0.15 cap (one promotion earned), so the
# continuation ladder starts there instead of re-earning 0.10.
V22_SPEED_CURRICULUM_CAPS_M_S = (0.10, 0.15, 0.25, 0.40, 0.55)  # v37: 0.55 rung for the wide-speed campaign
V22_TRIGGER_ANNEAL_STEPS = (0, 1)
V22_PUSH_SCALE_SCHEDULE = (0, 0, 1, 0.5)
# v39 hardware prep: ramp pushes from zero to FULL strength over the first
# 2,000 iterations (64,000 env steps at 32 steps/iter). The v38 checkpoint
# already tolerates 0.5-scale shoves, so the ramp mostly cushions the first
# few hundred iterations while the friction randomization settles in.
V39_PUSH_SCALE_SCHEDULE = (0, 0, 64000, 1.0)
FREE_HOP_MAX_SPEED_M_S = 0.15

# v29 annealed cold start: fresh weights, full direct-pipeline curriculum,
# but the shaping rewards (apex clearance, trigger bonus, takeoff impulse,
# gated velocity terms, ...) linearly fade to zero after the jump handover,
# leaving the diet-2.0 economics (capped hop displacement as the only
# tracking income). Pushes stay on the wall clock (they are robustness
# noise, not a skill gate). Plan ~12000 iterations.
V29_PUSH_RAMP_START_STEP = 9_500 * _PPO_STEPS_PER_ITERATION
V29_PUSH_RAMP_END_STEP = 11_500 * _PPO_STEPS_PER_ITERATION
# v31: the v29/v30 clock-scheduled transitions fired regardless of whether
# the run was ready (v30: hops started on a policy still falling every
# ~2.5 s; the scaffold fade ran while the ladder was stuck at rung one and
# bankrupted the gait). Phase transitions are now progress-gated:
# - Hops start when the balance phase has converged (EMA fraction of envs
#   tilted past 30 deg below this threshold), not on the clock alone.
# - Velocity commands follow the hop-start latch by a fixed offset.
# - The scaffold fade latch lives in the jump command term: handover
#   complete + healthy self-trigger rate + first earned speed promotion,
#   then a linear fade over the duration below.
# The ladder starts at 0.20 m/s: at the old 0.05 rung the tent income per
# hop was microscopic and standing still nearly satisfied the tracking
# kernel; 0.20 gives the gate real separation from the standstill error.
V31_BALANCE_GATE_UPSET_THRESHOLD = 0.02
V31_COMMAND_START_OFFSET_STEPS = 250 * _PPO_STEPS_PER_ITERATION
V31_SCAFFOLD_FADE_DURATION_STEPS = 2_000 * _PPO_STEPS_PER_ITERATION
V31_SPEED_CURRICULUM_CAPS_M_S = (0.20, 0.30, 0.40)
# v32: promotions gate on per-hop displacement accuracy -- the EMA of the
# tent fraction (achieved fraction of commanded hop distance) must clear
# this. The v31 velocity-error gate (error < 0.6 x command) was never
# passed by any lineage (best ratio ~1.03): a hopper is stationary between
# ballistic hops, so hop-averaged vector velocity error can't reach 60% of
# the command. 0.5 means hops consistently land within +/-50% of the
# commanded distance along the command. The push ramp is also gated now:
# it anchors on scaffold-fade completion instead of the wall clock (v31's
# clock ramp fired into an unpromoted run and ended its plateau).
V32_HOP_FRACTION_PROMOTION_THRESHOLD = 0.5


def _configure_scene_spec(spec: mujoco.MjSpec) -> None:
    spec.njmax = 128


def jumping_robot_balance_env_cfg(
    play: bool = False,
    height_control: bool = False,
    jump_stage_one: bool = False,
    jump_stage_two: bool = False,
    robust_balance: bool = False,
    strong_robust_balance: bool = False,
    warm_start_jump: bool = False,
    navigation: bool = False,
    full_pipeline: bool = False,
    pipeline_continue: bool = False,
    direct_pipeline: bool = False,
    direct_continue: bool = False,
    speed_continue: bool = False,
    free_hop_velocity: bool = False,
    annealed_pipeline: bool = False,
) -> ManagerBasedRlEnvCfg:
    # v29: identical machinery to direct-pipeline, plus a scheduled fade of
    # the scaffold rewards and a later push ramp (see V29_* constants).
    direct_pipeline = direct_pipeline or annealed_pipeline
    full_pipeline = (
        full_pipeline
        or pipeline_continue
        or direct_pipeline
        or direct_continue
        or speed_continue
    )
    # Stages sharing the proven direct-lineage machinery (velocity-estimate
    # obs, jump-window PD gains, 80 deg fall angle, relative speed gate).
    direct_lineage = direct_continue or direct_pipeline or speed_continue
    navigation = navigation or full_pipeline or free_hop_velocity
    if direct_continue or speed_continue:
        pipeline_phase_steps = (0, _NEVER_STEP)
        command_start_step = 0
    elif direct_pipeline:
        pipeline_phase_steps = (DIRECT_HOP_START_STEP, _NEVER_STEP)
        command_start_step = DIRECT_COMMAND_START_STEP
    elif pipeline_continue:
        pipeline_phase_steps = (CONTINUE_HOP_START_STEP, CONTINUE_MODEL_ONLY_STEP)
        command_start_step = CONTINUE_MODEL_ONLY_STEP
    else:
        pipeline_phase_steps = (PIPELINE_HOP_START_STEP, PIPELINE_MODEL_ONLY_STEP)
        command_start_step = PIPELINE_MODEL_ONLY_STEP
    jump_stage_two = jump_stage_two or warm_start_jump or navigation
    robust_balance = robust_balance or strong_robust_balance
    jump_stage_two = jump_stage_two or robust_balance
    jump_stage_one = jump_stage_one or jump_stage_two
    height_control = height_control or jump_stage_one
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={ROBOT_ENTITY_NAME: make_robot_entity_cfg()},
            num_envs=1,
            env_spacing=2.5,
            spec_fn=_configure_scene_spec,
        ),
        observations=build_observation_groups(
            height_control=height_control,
            jump_stage_one=jump_stage_one,
            jump_stage_two=jump_stage_two,
            sensor_noise=strong_robust_balance or navigation,
            navigation=navigation,
            actor_velocity_estimate=direct_lineage,
            # v40: 0-20 ms randomized actor-side sensor latency for hardware.
            sensor_latency=speed_continue,
        ),
        actions=(
            build_free_hop_action_terms()
            if free_hop_velocity
            else build_navigation_action_terms(
                jump_kp_n_m=(
                    DIRECT_CONTINUE_JUMP_KP_N_M
                    if (direct_lineage)
                    else None
                ),
                jump_kd_n_s_m=(
                    DIRECT_CONTINUE_JUMP_KD_N_S_M
                    if (direct_lineage)
                    else None
                ),
            )
            if navigation
            else (
                build_jump_stage_one_action_terms(
                    gate_with_jump_command=jump_stage_two,
                )
                if jump_stage_one
                else (
                    build_height_action_terms(play=play)
                    if height_control
                    else build_action_terms()
                )
            )
        ),
        events=(
            build_height_robustness_events(full_stroke=True)
            if robust_balance
            else build_randomization_events()
        ),
        rewards=build_reward_terms(
            height_control=height_control,
            jump_stage_one=jump_stage_one,
            jump_stage_two=jump_stage_two,
            robust_balance=robust_balance,
            warm_start_jump=warm_start_jump,
            navigation=navigation,
            fall_angle_deg=(
                DIRECT_CONTINUE_FALL_ANGLE_DEG
                if (direct_lineage)
                else 30.0
            ),
            relaxed_stance_tilt=direct_lineage,
            free_hop_velocity=free_hop_velocity,
            trigger_handover=direct_lineage,
            scaffold_fade=annealed_pipeline,
        ),
        terminations=build_termination_terms(
            fall_angle_deg=(
                DIRECT_CONTINUE_FALL_ANGLE_DEG
                if (direct_lineage)
                else 30.0
            ),
        ),
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name=ROBOT_ENTITY_NAME,
            body_name=BASE_BODY_NAME,
            distance=1.5,
            elevation=-20.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=SIM_TIMESTEP_S,
                iterations=10,
                ls_iterations=20,
            )
        ),
        decimation=SIM_DECIMATION,
        episode_length_s=EPISODE_LENGTH_S,
    )
    if strong_robust_balance:
        cfg.events = build_strong_robustness_events()
    elif warm_start_jump:
        cfg.events = build_warm_start_jump_events()
    elif navigation:
        cfg.events = build_strong_robustness_events(
            push_scale_schedule=(
                (
                    DIRECT_HOP_START_STEP,
                    (
                        V29_PUSH_RAMP_START_STEP
                        if annealed_pipeline
                        else V19_PUSH_RAMP_START_STEP
                    ),
                    (
                        V29_PUSH_RAMP_END_STEP
                        if annealed_pipeline
                        else V19_PUSH_RAMP_END_STEP
                    ),
                    V19_PUSH_RAMP_FINAL_SCALE,
                )
                if direct_pipeline
                else (V39_PUSH_SCALE_SCHEDULE if speed_continue else None)
            ),
            push_gate_on_scaffold_fade=annealed_pipeline,
            foot_friction_randomization=speed_continue,
        )

    commands = (
        build_height_commands(
            play=play or free_hop_velocity,
            full_stroke=robust_balance,
        )
        if height_control
        else {}
    )
    if jump_stage_two:
        commands.update(
            build_jump_commands(
                play=play,
                auto_trigger=not (robust_balance or navigation),
                repeat_auto_trigger=full_pipeline,
                auto_trigger_in_play=False,
                auto_trigger_min_planar_speed_m_s=None,
                resampling_time_range=(
                    (4.0, 6.0)
                    if warm_start_jump
                    else ((1.0, 2.0) if full_pipeline else (2.0, 4.0))
                ),
                success_gated_target_heights_m=(
                    DIRECT_CONTINUE_TARGET_HEIGHTS_M
                    if (direct_lineage)
                    else (NAVIGATION_TARGET_HEIGHTS_M if navigation else None)
                ),
                model_triggered=navigation and not full_pipeline,
                phase_schedule_steps=(
                    pipeline_phase_steps if full_pipeline else None
                ),
                # Continuations resume from a policy that already hops, so
                # each already-mastered rung only needs a short dwell time.
                curriculum_min_level_time_s=(
                    30.0
                    if (pipeline_continue or direct_continue or speed_continue)
                    else 60.0
                ),
                curriculum_gate_on_landing=direct_lineage,
                speed_scaled_resampling_range=(
                    DIRECT_CONTINUE_FAST_RESAMPLING_S
                    if (direct_lineage)
                    else None
                ),
                auto_trigger_anneal_steps=(
                    V14_TRIGGER_ANNEAL_STEPS
                    if direct_continue
                    else (
                        V19_TRIGGER_ANNEAL_STEPS
                        if direct_pipeline
                        else (
                            V22_TRIGGER_ANNEAL_STEPS
                            if speed_continue
                            else None
                        )
                    )
                ),
                balance_gate_upset_threshold=(
                    V31_BALANCE_GATE_UPSET_THRESHOLD
                    if annealed_pipeline
                    else None
                ),
                scaffold_fade_duration_steps=(
                    V31_SCAFFOLD_FADE_DURATION_STEPS
                    if annealed_pipeline
                    else None
                ),
            )
        )
    if navigation:
        commands.update(
            build_planar_velocity_commands(
                play=play,
                command_start_step=(
                    command_start_step if full_pipeline else None
                ),
                speed_curriculum_caps=(
                    V31_SPEED_CURRICULUM_CAPS_M_S
                    if annealed_pipeline
                    else (
                        # v38: no ladder for the wide-speed campaign.
                        # v37 sat on the 0.10 rung for 8,000 iterations
                        # because the 0.08 promotion gate sits below the
                        # gait's per-hop scatter floor (~0.10-0.15). A
                        # competent checkpoint doesn't need the gate;
                        # sample the full range instead.
                        None
                        if speed_continue
                        else (
                            V16_SPEED_CURRICULUM_CAPS_M_S
                            if direct_lineage
                            else None
                        )
                    )
                ),
                command_start_follows_jump_offset_steps=(
                    V31_COMMAND_START_OFFSET_STEPS
                    if annealed_pipeline
                    else None
                ),
                speed_gate_hop_fraction_threshold=(
                    V32_HOP_FRACTION_PROMOTION_THRESHOLD
                    if annealed_pipeline
                    else None
                ),
                speed_curriculum_relative_threshold=(
                    DIRECT_CONTINUE_SPEED_GATE_RATIO
                    if (direct_lineage)
                    else None
                ),
                stationary_probability=(
                    0.25
                    if (direct_lineage or free_hop_velocity)
                    else 0.35
                ),
                max_speed_m_s=(
                    FREE_HOP_MAX_SPEED_M_S
                    if free_hop_velocity
                    # v37: sample up to the new top rung during the
                    # wide-speed campaign; other stages keep 0.40.
                    else (0.55 if speed_continue else 0.40)
                ),
                apply_only_when_grounded=free_hop_velocity,
            )
        )
    if play:
        cfg.episode_length_s = 1e10
        cfg.observations["actor"].enable_corruption = False
        commands.update(build_disturbance_commands())
    cfg.commands = commands

    return cfg
