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
        ANNEALED_PIPELINE_TASK_ID,
        DIRECT_CONTINUE_TASK_ID,
        DIRECT_PIPELINE_TASK_ID,
        SPEED_CONTINUE_TASK_ID,
        FULL_PIPELINE_TASK_ID,
        FREE_HOP_VELOCITY_TASK_ID,
        PIPELINE_CONTINUE_TASK_ID,
        HEIGHT_TASK_ID,
        JUMP_STAGE_ONE_TASK_ID,
        JUMP_STAGE_TWO_TASK_ID,
        NAVIGATION_TASK_ID,
        ROBUST_BALANCE_TASK_ID,
        STRONG_ROBUST_BALANCE_TASK_ID,
        TASK_ID,
        WARM_START_JUMP_TASK_ID,
        register_tasks,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-root", default="logs/rsl_rl")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Distinctive run label used for the log directory and W&B run.",
    )
    parser.add_argument(
        "--stage",
        choices=(
            "nominal",
            "robust",
            "height",
            "robust-balance",
            "strong-robust-balance",
            "warmstart-jump",
            "navigation",
            "jump-stage-1",
            "jump-stage-2",
            "full-pipeline",
            "pipeline-continue",
            "direct-pipeline",
            "direct-continue",
            "speed-continue",
            "annealed-pipeline",
            "free-hop-velocity",
        ),
        default="nominal",
        help="Training curriculum stage.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Local checkpoint path under the experiment log directory. "
            "Stage 2 can warm-start from a Stage 1 checkpoint; other stages must "
            "match the selected action and observation interface."
        ),
    )
    args = parser.parse_args()

    register_tasks()

    task_id = {
        "height": HEIGHT_TASK_ID,
        "jump-stage-1": JUMP_STAGE_ONE_TASK_ID,
        "jump-stage-2": JUMP_STAGE_TWO_TASK_ID,
        "robust-balance": ROBUST_BALANCE_TASK_ID,
        "strong-robust-balance": STRONG_ROBUST_BALANCE_TASK_ID,
        "warmstart-jump": WARM_START_JUMP_TASK_ID,
        "navigation": NAVIGATION_TASK_ID,
        "full-pipeline": FULL_PIPELINE_TASK_ID,
        "pipeline-continue": PIPELINE_CONTINUE_TASK_ID,
        "direct-pipeline": DIRECT_PIPELINE_TASK_ID,
        "direct-continue": DIRECT_CONTINUE_TASK_ID,
        "speed-continue": SPEED_CONTINUE_TASK_ID,
        "annealed-pipeline": ANNEALED_PIPELINE_TASK_ID,
        "free-hop-velocity": FREE_HOP_VELOCITY_TASK_ID,
    }.get(args.stage, TASK_ID)
    cfg = TrainConfig.from_task(task_id)
    cfg.env.scene.num_envs = args.num_envs
    cfg.agent.max_iterations = args.max_iterations
    cfg.agent.seed = args.seed
    cfg.env.seed = args.seed
    cfg = replace(cfg, log_root=args.log_root)
    cfg.agent.wandb_tags = (*cfg.agent.wandb_tags, args.stage)

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
    elif args.stage in (
        "robust-balance",
        "strong-robust-balance",
        "warmstart-jump",
        "navigation",
        "jump-stage-2",
    ):
        from mjlab_tasks.jumping_robot_balance.mdp import (
            build_jump_stage_two_events,
        )

        if args.stage in ("robust-balance", "strong-robust-balance"):
            from mjlab_tasks.jumping_robot_balance.mdp import (
                build_height_robustness_events,
                build_strong_robustness_events,
            )

            cfg.env.events = (
                build_strong_robustness_events()
                if args.stage == "strong-robust-balance"
                else build_height_robustness_events(full_stroke=True)
            )
            cfg.agent.algorithm.learning_rate = 1.0e-4
            cfg.agent.algorithm.entropy_coef = 5.0e-4
            cfg.agent.run_name = (
                "strong_robust_balance_100hz"
                if args.stage == "strong-robust-balance"
                else "robust_balance_100hz"
            )
        elif args.stage == "warmstart-jump":
            from mjlab_tasks.jumping_robot_balance.mdp import (
                build_warm_start_jump_events,
            )

            cfg.env.events = build_warm_start_jump_events()
            cfg.agent.algorithm.learning_rate = 2.5e-5
            cfg.agent.algorithm.entropy_coef = 5.0e-4
            cfg.agent.run_name = "warmstart_small_jumps"
        elif args.stage == "navigation":
            from mjlab_tasks.jumping_robot_balance.mdp import (
                build_strong_robustness_events,
            )

            cfg.env.events = build_strong_robustness_events()
            cfg.agent.algorithm.learning_rate = 2.5e-5
            cfg.agent.algorithm.entropy_coef = 5.0e-4
            cfg.agent.run_name = "navigation_hops"
        else:
            cfg.env.events = build_jump_stage_two_events()
            cfg.agent.algorithm.learning_rate = 2.5e-5
            cfg.agent.algorithm.entropy_coef = 2.0e-4
            cfg.agent.run_name = "jump_stage_2"
        cfg.agent.algorithm.schedule = "fixed"
    elif args.stage == "full-pipeline":
        # From-scratch training: keep the default adaptive learning rate.
        # Events come from the env cfg (strong robustness set).
        cfg.agent.run_name = "full_pipeline"
    elif args.stage == "pipeline-continue":
        # Fine-tunes a trained full-pipeline checkpoint on the extended
        # height ladder; adaptive LR self-tunes from the resumed policy.
        cfg.agent.run_name = "pipeline_continue"
    elif args.stage == "direct-pipeline":
        # From-scratch training with directional locomotion learned alongside
        # hopping; keep the default adaptive learning rate.
        cfg.agent.run_name = "direct_pipeline"
    elif args.stage == "direct-continue":
        # Fine-tunes a direct-pipeline checkpoint with the new velocity
        # estimate input and the taller height ladder.
        cfg.agent.run_name = "direct_continue"
    elif args.stage == "speed-continue":
        # Speed-ladder continuation of a post-handover direct-pipeline
        # checkpoint; adaptive LR self-tunes from the resumed policy.
        # v28: zero entropy again. v27 showed the v26 diet still left
        # hop-count-scaling income (the gated velocity terms), so 2e-4
        # kept ratcheting the std (0.25 -> 0.62). Diet 2.0 removes that
        # income; exploration comes from the checkpoint's clamped std.
        cfg.agent.algorithm.entropy_coef = 0.0
        cfg.agent.run_name = "speed_continue"
    elif args.stage == "annealed-pipeline":
        # v29: from scratch with the scaffold-reward fade. Default adaptive
        # LR and default entropy -- a cold start needs real exploration,
        # and once the scaffolds fade there is no per-jump income left for
        # entropy-driven noise to farm, so the std ratchet has no fuel.
        cfg.agent.run_name = "annealed_pipeline"
    elif args.stage == "free-hop-velocity":
        cfg.agent.algorithm.learning_rate = 2.5e-5
        cfg.agent.algorithm.schedule = "fixed"
        cfg.agent.algorithm.entropy_coef = 5.0e-4
        cfg.agent.run_name = "free_hop_velocity_from_robust_balance"

    if args.run_name is not None:
        cfg.agent.run_name = args.run_name
        cfg.agent.wandb_tags = (*cfg.agent.wandb_tags, args.run_name)

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
    elif args.stage in ("navigation", "free-hop-velocity"):
        raise ValueError(
            f"{args.stage} must warm-start from the known-good robust balance "
            "checkpoint via --resume-from."
        )
    elif args.stage == "pipeline-continue":
        raise ValueError(
            "Pipeline continuation must resume from a trained full-pipeline "
            "checkpoint via --resume-from."
        )
    elif args.stage == "direct-continue":
        raise ValueError(
            "Direct continuation must resume from a trained direct-pipeline "
            "checkpoint via --resume-from."
        )
    elif args.stage == "speed-continue":
        raise ValueError(
            "Speed continuation must resume from a trained direct-pipeline "
            "checkpoint via --resume-from."
        )

    if not torch.cuda.is_available():
        print("[WARN] No CUDA GPU detected; forcing CPU mode for mjlab training launch.")
        cfg = replace(cfg, gpu_ids=None)

    launch_training(task_id, cfg)


if __name__ == "__main__":
    main()
