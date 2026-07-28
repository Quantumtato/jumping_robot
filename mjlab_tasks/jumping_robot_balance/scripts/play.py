"""Play helper for the jumping robot balance task."""

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

    from mjlab_tasks.jumping_robot_balance.task_registry import TASK_ID, register_tasks

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-file", default=None)
    parser.add_argument("--wandb-run-path", default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--viewer", choices=("auto", "native", "viser"), default="auto")
    args = parser.parse_args()

    register_tasks()

    cfg = PlayConfig(
        checkpoint_file=args.checkpoint_file,
        wandb_run_path=args.wandb_run_path,
        num_envs=args.num_envs,
        viewer=args.viewer,
    )
    run_play(TASK_ID, cfg)


if __name__ == "__main__":
    main()
