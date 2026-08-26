"""Task registration for the jumping robot balance task."""

from __future__ import annotations

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from mjlab_tasks.jumping_robot_balance.env_cfg import jumping_robot_balance_env_cfg
from mjlab_tasks.jumping_robot_balance.rl.ppo_cfg import (
    jumping_robot_balance_ppo_runner_cfg,
)
from mjlab_tasks.jumping_robot_balance.rl.stage_transition_runner import (
    JumpStageTwoRunner,
    JumpWarmStartRunner,
    NavigationWarmStartRunner,
    PipelineContinueRunner,
    SpeedContinueRunner,
    VelocityEstimateWarmStartRunner,
)

TASK_ID = "Mjlab-Balance-JumpingRobot-v0"
HEIGHT_TASK_ID = "Mjlab-Balance-Height-JumpingRobot-v0"
JUMP_STAGE_ONE_TASK_ID = "Mjlab-Jump-Stage1-JumpingRobot-v0"
JUMP_STAGE_TWO_TASK_ID = "Mjlab-Jump-Stage2-JumpingRobot-v0"
ROBUST_BALANCE_TASK_ID = "Mjlab-Robust-Balance-JumpingRobot-v0"
STRONG_ROBUST_BALANCE_TASK_ID = "Mjlab-Strong-Robust-Balance-JumpingRobot-v0"
WARM_START_JUMP_TASK_ID = "Mjlab-WarmStart-Jump-JumpingRobot-v0"
NAVIGATION_TASK_ID = "Mjlab-Navigation-JumpingRobot-v0"
FULL_PIPELINE_TASK_ID = "Mjlab-FullPipeline-JumpingRobot-v0"
PIPELINE_CONTINUE_TASK_ID = "Mjlab-PipelineContinue-JumpingRobot-v0"
DIRECT_PIPELINE_TASK_ID = "Mjlab-DirectPipeline-JumpingRobot-v0"
DIRECT_CONTINUE_TASK_ID = "Mjlab-DirectContinue-JumpingRobot-v0"
SPEED_CONTINUE_TASK_ID = "Mjlab-SpeedContinue-JumpingRobot-v0"
ANNEALED_PIPELINE_TASK_ID = "Mjlab-AnnealedPipeline-JumpingRobot-v0"
FREE_HOP_VELOCITY_TASK_ID = "Mjlab-FreeHopVelocity-JumpingRobot-v0"


def register_tasks() -> None:
    registered = list_tasks()
    if TASK_ID not in registered:
        register_mjlab_task(
            task_id=TASK_ID,
            env_cfg=jumping_robot_balance_env_cfg(),
            play_env_cfg=jumping_robot_balance_env_cfg(play=True),
            rl_cfg=jumping_robot_balance_ppo_runner_cfg(),
        )
    if HEIGHT_TASK_ID not in registered:
        register_mjlab_task(
            task_id=HEIGHT_TASK_ID,
            env_cfg=jumping_robot_balance_env_cfg(height_control=True),
            play_env_cfg=jumping_robot_balance_env_cfg(
                play=True,
                height_control=True,
            ),
            rl_cfg=jumping_robot_balance_ppo_runner_cfg(),
        )
    if JUMP_STAGE_ONE_TASK_ID not in registered:
        register_mjlab_task(
            task_id=JUMP_STAGE_ONE_TASK_ID,
            env_cfg=jumping_robot_balance_env_cfg(jump_stage_one=True),
            play_env_cfg=jumping_robot_balance_env_cfg(
                play=True,
                jump_stage_one=True,
            ),
            rl_cfg=jumping_robot_balance_ppo_runner_cfg(),
        )
    if JUMP_STAGE_TWO_TASK_ID not in registered:
        register_mjlab_task(
            task_id=JUMP_STAGE_TWO_TASK_ID,
            env_cfg=jumping_robot_balance_env_cfg(jump_stage_two=True),
            play_env_cfg=jumping_robot_balance_env_cfg(
                play=True,
                jump_stage_two=True,
            ),
            rl_cfg=jumping_robot_balance_ppo_runner_cfg(),
            runner_cls=JumpStageTwoRunner,
        )
    if ROBUST_BALANCE_TASK_ID not in registered:
        register_mjlab_task(
            task_id=ROBUST_BALANCE_TASK_ID,
            env_cfg=jumping_robot_balance_env_cfg(robust_balance=True),
            play_env_cfg=jumping_robot_balance_env_cfg(
                play=True,
                robust_balance=True,
            ),
            rl_cfg=jumping_robot_balance_ppo_runner_cfg(),
        )
    if STRONG_ROBUST_BALANCE_TASK_ID not in registered:
        register_mjlab_task(
            task_id=STRONG_ROBUST_BALANCE_TASK_ID,
            env_cfg=jumping_robot_balance_env_cfg(strong_robust_balance=True),
            play_env_cfg=jumping_robot_balance_env_cfg(
                play=True,
                strong_robust_balance=True,
            ),
            rl_cfg=jumping_robot_balance_ppo_runner_cfg(),
        )
    if WARM_START_JUMP_TASK_ID not in registered:
        register_mjlab_task(
            task_id=WARM_START_JUMP_TASK_ID,
            env_cfg=jumping_robot_balance_env_cfg(warm_start_jump=True),
            play_env_cfg=jumping_robot_balance_env_cfg(
                play=True,
                warm_start_jump=True,
            ),
            rl_cfg=jumping_robot_balance_ppo_runner_cfg(),
            runner_cls=JumpWarmStartRunner,
        )
    if NAVIGATION_TASK_ID not in registered:
        register_mjlab_task(
            task_id=NAVIGATION_TASK_ID,
            env_cfg=jumping_robot_balance_env_cfg(navigation=True),
            play_env_cfg=jumping_robot_balance_env_cfg(
                play=True,
                navigation=True,
            ),
            rl_cfg=jumping_robot_balance_ppo_runner_cfg(),
            runner_cls=NavigationWarmStartRunner,
        )
    if FULL_PIPELINE_TASK_ID not in registered:
        # Trains from random weights; phases are step-scheduled inside the
        # env, so no warm-start runner is needed.
        register_mjlab_task(
            task_id=FULL_PIPELINE_TASK_ID,
            env_cfg=jumping_robot_balance_env_cfg(full_pipeline=True),
            play_env_cfg=jumping_robot_balance_env_cfg(
                play=True,
                full_pipeline=True,
            ),
            rl_cfg=jumping_robot_balance_ppo_runner_cfg(),
        )
    if PIPELINE_CONTINUE_TASK_ID not in registered:
        # Resumes a trained full-pipeline checkpoint against the extended
        # height ladder and faster velocity commands: short forced-hop phase
        # to climb the new rungs, then model-owned locomotion.
        register_mjlab_task(
            task_id=PIPELINE_CONTINUE_TASK_ID,
            env_cfg=jumping_robot_balance_env_cfg(pipeline_continue=True),
            play_env_cfg=jumping_robot_balance_env_cfg(
                play=True,
                pipeline_continue=True,
            ),
            rl_cfg=jumping_robot_balance_ppo_runner_cfg(),
            runner_cls=PipelineContinueRunner,
        )
    if DIRECT_PIPELINE_TASK_ID not in registered:
        # v8: from scratch; balance, then forced hops with velocity commands
        # from the start of locomotion (no hop-in-place phase to unlearn).
        register_mjlab_task(
            task_id=DIRECT_PIPELINE_TASK_ID,
            env_cfg=jumping_robot_balance_env_cfg(direct_pipeline=True),
            play_env_cfg=jumping_robot_balance_env_cfg(
                play=True,
                direct_pipeline=True,
            ),
            rl_cfg=jumping_robot_balance_ppo_runner_cfg(),
        )
    if DIRECT_CONTINUE_TASK_ID not in registered:
        # v9: resumes a direct-pipeline checkpoint with the actor velocity
        # estimate appended, the tall height ladder re-opened, and the
        # relative speed gate.
        register_mjlab_task(
            task_id=DIRECT_CONTINUE_TASK_ID,
            env_cfg=jumping_robot_balance_env_cfg(direct_continue=True),
            play_env_cfg=jumping_robot_balance_env_cfg(
                play=True,
                direct_continue=True,
            ),
            rl_cfg=jumping_robot_balance_ppo_runner_cfg(),
            runner_cls=VelocityEstimateWarmStartRunner,
        )
    if SPEED_CONTINUE_TASK_ID not in registered:
        # v22: resumes a trained direct-pipeline checkpoint whose policy
        # already owns jump timing (post-handover); no forced triggers,
        # constant half-strength pushes, and the speed ladder starting at
        # the previously earned 0.10 m/s cap. Same observation and action
        # layout as direct-pipeline. v25/v25b clamped the resumed action
        # std after every update (v23/v23b/v24 inflated noise until hops
        # turned chaotic). v27: with the v26 reward diet removing all
        # per-jump income, noise should no longer be self-funding, so the
        # clamp is lifted to restore full exploration -- the plain continue
        # runner loads the checkpoint std (~0.24) unmodified.
        register_mjlab_task(
            task_id=SPEED_CONTINUE_TASK_ID,
            env_cfg=jumping_robot_balance_env_cfg(speed_continue=True),
            play_env_cfg=jumping_robot_balance_env_cfg(
                play=True,
                speed_continue=True,
            ),
            rl_cfg=jumping_robot_balance_ppo_runner_cfg(),
            runner_cls=SpeedContinueRunner,
        )
    if ANNEALED_PIPELINE_TASK_ID not in registered:
        # v29: from random weights on the full direct-pipeline curriculum,
        # with the shaping rewards fading to zero after the jump handover
        # so the run graduates onto the diet-2.0 economics (capped hop
        # displacement as sole tracking income) without a checkpoint
        # handoff -- the from-scratch answer to the overfit-to-old-rewards
        # concern in the warm-start lineage.
        register_mjlab_task(
            task_id=ANNEALED_PIPELINE_TASK_ID,
            env_cfg=jumping_robot_balance_env_cfg(annealed_pipeline=True),
            play_env_cfg=jumping_robot_balance_env_cfg(
                play=True,
                annealed_pipeline=True,
            ),
            rl_cfg=jumping_robot_balance_ppo_runner_cfg(),
        )
    if FREE_HOP_VELOCITY_TASK_ID not in registered:
        register_mjlab_task(
            task_id=FREE_HOP_VELOCITY_TASK_ID,
            env_cfg=jumping_robot_balance_env_cfg(free_hop_velocity=True),
            play_env_cfg=jumping_robot_balance_env_cfg(
                play=True,
                free_hop_velocity=True,
            ),
            rl_cfg=jumping_robot_balance_ppo_runner_cfg(),
            runner_cls=NavigationWarmStartRunner,
        )
