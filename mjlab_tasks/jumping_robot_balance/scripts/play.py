"""Play helper for the jumping robot balance task."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import sys
import time

import torch


def _add_repo_root_to_pythonpath() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _build_strided_viewer_classes():
    from mjlab.viewer.native import NativeMujocoViewer
    from mjlab.viewer.viser import ViserPlayViewer

    class _ViewerStrideMixin:
        def __init__(self, *args, viewer_stride: int = 1, **kwargs):
            super().__init__(*args, **kwargs)
            self._viewer_stride = max(1, int(viewer_stride))
            self._last_rendered_step = -1
            self._force_next_render = True

        def reset_environment(self) -> None:
            super().reset_environment()
            self._force_next_render = True

        def _single_step(self) -> None:
            super()._single_step()
            self._force_next_render = True

        def tick(self) -> bool:
            now = time.perf_counter()
            dt = now - self._last_tick_time
            self._last_tick_time = now

            self._process_actions()

            if self._is_paused:
                self._forward_paused()
            else:
                self._step_physics(dt)

            self._time_until_next_render -= dt
            if self._time_until_next_render > 0:
                return False

            self._time_until_next_render += self.frame_time
            if self._time_until_next_render < -self.frame_time:
                self._time_until_next_render = 0.0

            if not self._is_paused and not self._force_next_render:
                if (
                    self._step_count - self._last_rendered_step
                    < self._viewer_stride
                ):
                    return False

            self.sync_env_to_viewer()
            self._last_rendered_step = self._step_count
            self._force_next_render = False
            self._stats_frames += 1
            return True

    class StridedNativeMujocoViewer(_ViewerStrideMixin, NativeMujocoViewer):
        pass

    class StridedViserPlayViewer(_ViewerStrideMixin, ViserPlayViewer):
        pass

    return StridedNativeMujocoViewer, StridedViserPlayViewer


def main() -> None:
    _add_repo_root_to_pythonpath()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.scripts.play import PlayConfig, get_wandb_checkpoint_path
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    from mjlab_tasks.jumping_robot_balance.task_registry import (
        HEIGHT_TASK_ID,
        JUMP_STAGE_ONE_TASK_ID,
        JUMP_STAGE_TWO_TASK_ID,
        TASK_ID,
        register_tasks,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=("zero", "random", "trained"), default="zero")
    parser.add_argument("--checkpoint-file", default=None)
    parser.add_argument("--wandb-run-path", default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--viewer", choices=("auto", "native", "viser"), default="auto")
    parser.add_argument(
        "--viewer-stride",
        type=int,
        default=1,
        help=(
            "Render the viewer every N environment steps while running. "
            "Keeps policy/physics stepping unchanged."
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
        help="Use the five-action jump Stage 1 task.",
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
        num_envs=args.num_envs,
        device=args.device,
        viewer=args.viewer,
        no_terminations=args.no_terminations,
    )
    if args.jump_stage_2:
        task_id = JUMP_STAGE_TWO_TASK_ID
    elif args.jump_stage_1:
        task_id = JUMP_STAGE_ONE_TASK_ID
    elif args.height_control:
        task_id = HEIGHT_TASK_ID
    else:
        task_id = TASK_ID

    if args.viewer_stride <= 1:
        from mjlab.scripts.play import run_play

        run_play(task_id, cfg)
        return

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    dummy_mode = cfg.agent in {"zero", "random"}
    trained_mode = not dummy_mode
    if cfg.no_terminations:
        env_cfg.terminations = {}
    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs

    resume_path: Path | None = None
    if trained_mode:
        log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
        if cfg.checkpoint_file is not None:
            resume_path = Path(cfg.checkpoint_file)
            if not resume_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        else:
            if cfg.wandb_run_path is None:
                raise ValueError(
                    "`wandb_run_path` is required when `checkpoint_file` is not provided."
                )
            resume_path, _ = get_wandb_checkpoint_path(
                log_root_path, Path(cfg.wandb_run_path), None
            )

    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(
            cfg=env_cfg,
            device=device,
        ),
        clip_actions=agent_cfg.clip_actions,
    )

    if dummy_mode:
        action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
        if cfg.agent == "zero":

            class PolicyZero:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return torch.zeros(action_shape, device=env.unwrapped.device)

            policy = PolicyZero()
        else:

            class PolicyRandom:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

            policy = PolicyRandom()
    else:
        assert resume_path is not None
        runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(
            str(resume_path),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)

    has_display = bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    if cfg.viewer == "auto":
        resolved_viewer = "native" if has_display else "viser"
    else:
        resolved_viewer = cfg.viewer
    StridedNativeMujocoViewer, StridedViserPlayViewer = (
        _build_strided_viewer_classes()
    )
    try:
        if resolved_viewer == "native":
            StridedNativeMujocoViewer(
                env,
                policy,
                viewer_stride=args.viewer_stride,
            ).run()
        elif resolved_viewer == "viser":
            StridedViserPlayViewer(
                env,
                policy,
                viewer_stride=args.viewer_stride,
            ).run()
        else:
            raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
