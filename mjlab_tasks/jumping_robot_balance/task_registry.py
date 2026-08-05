"""Task registration for the jumping robot balance task."""

from __future__ import annotations

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from mjlab_tasks.jumping_robot_balance.env_cfg import jumping_robot_balance_env_cfg
from mjlab_tasks.jumping_robot_balance.rl.ppo_cfg import (
    jumping_robot_balance_ppo_runner_cfg,
)

TASK_ID = "Mjlab-Balance-JumpingRobot-v0"
HEIGHT_TASK_ID = "Mjlab-Balance-Height-JumpingRobot-v0"
JUMP_STAGE_ONE_TASK_ID = "Mjlab-Jump-Stage1-JumpingRobot-v0"


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
