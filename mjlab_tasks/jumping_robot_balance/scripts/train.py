"""Train helper for the jumping robot balance task."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import torch


def _add_repo_root_to_pythonpath() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def main() -> None:
    _add_repo_root_to_pythonpath()

    from mjlab.scripts.train import TrainConfig, launch_training

    from mjlab_tasks.jumping_robot_balance.task_registry import TASK_ID, register_tasks

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-root", default="logs/rsl_rl")
    args = parser.parse_args()

    register_tasks()

    cfg = TrainConfig.from_task(TASK_ID)
    cfg.env.scene.num_envs = args.num_envs
    cfg.agent.max_iterations = args.max_iterations
    cfg.agent.seed = args.seed
    cfg.env.seed = args.seed
    cfg = replace(cfg, log_root=args.log_root)

    if not torch.cuda.is_available():
        print("[WARN] No CUDA GPU detected; forcing CPU mode for mjlab training launch.")
        cfg = replace(cfg, gpu_ids=None)

    launch_training(TASK_ID, cfg)


if __name__ == "__main__":
    main()
