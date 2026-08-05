"""Train helper for the jumping robot balance task."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import replace
from pathlib import Path

import torch


def _add_repo_root_to_pythonpath() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def main() -> None:
    _add_repo_root_to_pythonpath()

    from mjlab.scripts.train import TrainConfig, launch_training

    from mjlab_tasks.jumping_robot_balance.task_registry import (
        HEIGHT_TASK_ID,
        JUMP_STAGE_ONE_TASK_ID,
        TASK_ID,
        register_tasks,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-root", default="logs/rsl_rl")
    parser.add_argument(
        "--stage",
        choices=("nominal", "robust", "height", "jump-stage-1"),
        default="nominal",
        help="Training curriculum stage.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Local checkpoint path under the experiment log directory. "
            "Checkpoints must match the selected stage's action and observation "
            "interface."
        ),
    )
    args = parser.parse_args()

    register_tasks()

    task_id = {
        "height": HEIGHT_TASK_ID,
        "jump-stage-1": JUMP_STAGE_ONE_TASK_ID,
    }.get(args.stage, TASK_ID)
    cfg = TrainConfig.from_task(task_id)
    cfg.env.scene.num_envs = args.num_envs
    cfg.agent.max_iterations = args.max_iterations
    cfg.agent.seed = args.seed
    cfg.env.seed = args.seed
    cfg = replace(cfg, log_root=args.log_root)

    if args.stage == "robust":
        from mjlab_tasks.jumping_robot_balance.mdp import build_robustness_events

        cfg.env.events = build_robustness_events()
        cfg.agent.algorithm.learning_rate = 1.0e-4
        cfg.agent.algorithm.entropy_coef = 5.0e-4
        cfg.agent.run_name = "robust"
    elif args.stage == "height":
        from mjlab_tasks.jumping_robot_balance.mdp import (
            build_height_robustness_events,
        )

        cfg.env.events = build_height_robustness_events()
        cfg.agent.algorithm.learning_rate = 5.0e-5
        cfg.agent.algorithm.schedule = "fixed"
        cfg.agent.algorithm.entropy_coef = 5.0e-4
        cfg.agent.run_name = "height"
    elif args.stage == "jump-stage-1":
        from mjlab_tasks.jumping_robot_balance.mdp import (
            build_height_robustness_events,
        )

        cfg.env.events = build_height_robustness_events()
        cfg.agent.algorithm.learning_rate = 5.0e-5
        cfg.agent.algorithm.schedule = "fixed"
        cfg.agent.algorithm.entropy_coef = 5.0e-4
        cfg.agent.run_name = "jump_stage_1"

    if args.resume_from is not None:
        resume_path = Path(args.resume_from).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        experiment_root = (
            Path(args.log_root) / cfg.agent.experiment_name
        ).resolve()
        try:
            relative_path = resume_path.relative_to(experiment_root)
        except ValueError as exc:
            raise ValueError(
                f"Resume checkpoint must be under {experiment_root}: {resume_path}"
            ) from exc
        if len(relative_path.parts) != 2:
            raise ValueError(
                "Resume checkpoint must be directly inside a training run directory."
            )
        cfg.agent.resume = True
        cfg.agent.load_run = f"^{re.escape(relative_path.parent.name)}$"
        cfg.agent.load_checkpoint = f"^{re.escape(relative_path.name)}$"

    if not torch.cuda.is_available():
        print("[WARN] No CUDA GPU detected; forcing CPU mode for mjlab training launch.")
        cfg = replace(cfg, gpu_ids=None)

    launch_training(task_id, cfg)


if __name__ == "__main__":
    main()
