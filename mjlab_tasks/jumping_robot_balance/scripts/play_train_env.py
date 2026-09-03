"""Viser viewer running the TRAINING environment (not the play config).

Registers a "train-view" variant of the speed-continue task whose
play_env_cfg is the actual training cfg: random command resampling,
sensor noise, scheduled pushes, finite episodes, and real terminations.
What you see is what the fleet trains on (deterministic policy mean;
training additionally samples exploration noise around it).

Usage:
    python play_train_env.py --checkpoint-file <model.pt> [--num-envs 4]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _add_repo_root_to_pythonpath() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def main() -> None:
    _add_repo_root_to_pythonpath()
    from mjlab.scripts.play import PlayConfig, run_play
    from mjlab.tasks.registry import list_tasks, register_mjlab_task

    from mjlab_tasks.jumping_robot_balance.env_cfg import (
        jumping_robot_balance_env_cfg,
    )
    from mjlab_tasks.jumping_robot_balance.rl.ppo_cfg import (
        jumping_robot_balance_ppo_runner_cfg,
    )
    from mjlab_tasks.jumping_robot_balance.rl.stage_transition_runner import (
        SpeedContinueRunner,
    )
    from mjlab_tasks.jumping_robot_balance.task_registry import register_tasks

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    register_tasks()

    task_id = "Mjlab-SpeedContinue-TrainView-JumpingRobot-v0"
    if task_id not in list_tasks():
        train_cfg = jumping_robot_balance_env_cfg(speed_continue=True)
        register_mjlab_task(
            task_id=task_id,
            env_cfg=train_cfg,
            # The whole point: the viewer gets the training cfg, not the
            # play cfg. Random commands, noise, pushes, resets included.
            play_env_cfg=jumping_robot_balance_env_cfg(speed_continue=True),
            rl_cfg=jumping_robot_balance_ppo_runner_cfg(),
            runner_cls=SpeedContinueRunner,
        )

    run_play(
        task_id,
        PlayConfig(
            agent="trained",
            checkpoint_file=args.checkpoint_file,
            num_envs=args.num_envs,
            device=args.device,
            viewer="viser",
        ),
    )


if __name__ == "__main__":
    main()
