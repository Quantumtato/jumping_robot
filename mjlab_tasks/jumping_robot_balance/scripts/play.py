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

    from mjlab_tasks.jumping_robot_balance.task_registry import (
        DIRECT_CONTINUE_TASK_ID,
        DIRECT_PIPELINE_TASK_ID,
        FULL_PIPELINE_TASK_ID,
        FREE_HOP_VELOCITY_TASK_ID,
        HEIGHT_TASK_ID,
        JUMP_STAGE_ONE_TASK_ID,
        JUMP_STAGE_TWO_TASK_ID,
        NAVIGATION_TASK_ID,
        PIPELINE_CONTINUE_TASK_ID,
        ROBUST_BALANCE_TASK_ID,
        STRONG_ROBUST_BALANCE_TASK_ID,
        TASK_ID,
        WARM_START_JUMP_TASK_ID,
        register_tasks,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=("zero", "random", "trained"), default="zero")
    parser.add_argument("--checkpoint-file", default=None)
    parser.add_argument("--wandb-run-path", default=None)
    parser.add_argument(
        "--wandb-checkpoint-name",
        default=None,
        help=(
            "Checkpoint filename within the W&B run, for example "
            "model_300.pt. Defaults to the latest uploaded checkpoint."
        ),
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--viewer", choices=("auto", "native", "viser"), default="auto")
    parser.add_argument(
        "--viewer-stride",
        type=int,
        default=1,
        help=(
            "Deprecated and ignored. Use the Viser Speed controls to increase "
            "playback speed without dropping viewer updates."
        ),
    )
    parser.add_argument("--no-terminations", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--height-control",
        action="store_true",
        help="Enable the trained linear-position action for height-stage checkpoints.",
    )
    mode.add_argument(
        "--jump-stage-1",
        action="store_true",
        help="Use the three-action jump Stage 1 task.",
    )
    mode.add_argument(
        "--robust-balance",
        action="store_true",
        help="Use the history-equipped robust balance task without automatic jumps.",
    )
    mode.add_argument(
        "--strong-robust-balance",
        action="store_true",
        help="Use the robust balance task with the strong physics randomization.",
    )
    mode.add_argument(
        "--warmstart-jump",
        action="store_true",
        help="Use the small-jump and landing-recovery curriculum.",
    )
    mode.add_argument(
        "--navigation",
        action="store_true",
        help="Use repeated hopping with world-frame planar velocity commands.",
    )
    mode.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Use the full pipeline task playback configuration.",
    )
    mode.add_argument(
        "--pipeline-continue",
        action="store_true",
        help="Use the continued full-pipeline playback configuration.",
    )
    mode.add_argument(
        "--direct-pipeline",
        action="store_true",
        help="Use the direct-pipeline locomotion playback configuration.",
    )
    mode.add_argument(
        "--direct-continue",
        action="store_true",
        help="Use the direct-continue playback configuration for v9 models.",
    )
    mode.add_argument(
        "--free-hop-velocity",
        action="store_true",
        help="Use autonomous free-hop velocity-control playback.",
    )
    mode.add_argument(
        "--jump-stage-2",
        action="store_true",
        help="Use the vertical-hop Stage 2 task and one-shot jump command.",
    )
    args = parser.parse_args()

    register_tasks()

    cfg = PlayConfig(
        agent=args.agent,
        checkpoint_file=args.checkpoint_file,
        wandb_run_path=args.wandb_run_path,
        wandb_checkpoint_name=args.wandb_checkpoint_name,
        num_envs=args.num_envs,
        device=args.device,
        viewer=args.viewer,
        no_terminations=args.no_terminations,
    )
    if args.free_hop_velocity:
        task_id = FREE_HOP_VELOCITY_TASK_ID
    elif args.direct_continue:
        task_id = DIRECT_CONTINUE_TASK_ID
    elif args.direct_pipeline:
        task_id = DIRECT_PIPELINE_TASK_ID
    elif args.pipeline_continue:
        task_id = PIPELINE_CONTINUE_TASK_ID
    elif args.full_pipeline:
        task_id = FULL_PIPELINE_TASK_ID
    elif args.navigation:
        task_id = NAVIGATION_TASK_ID
    elif args.warmstart_jump:
        task_id = WARM_START_JUMP_TASK_ID
    elif args.strong_robust_balance:
        task_id = STRONG_ROBUST_BALANCE_TASK_ID
    elif args.robust_balance:
        task_id = ROBUST_BALANCE_TASK_ID
    elif args.jump_stage_2:
        task_id = JUMP_STAGE_TWO_TASK_ID
    elif args.jump_stage_1:
        task_id = JUMP_STAGE_ONE_TASK_ID
    elif args.height_control:
        task_id = HEIGHT_TASK_ID
    else:
        task_id = TASK_ID

    if args.viewer_stride > 1:
        print(
            "[WARN]: --viewer-stride is ignored because it only drops visual "
            "updates. Use the Viser Speed controls to fast-forward playback."
        )
    run_play(task_id, cfg)


if __name__ == "__main__":
    main()
