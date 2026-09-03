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


def _install_multirun_checkpoint_browser(checkpoint_file: str) -> None:
    """Add a Run dropdown next to the viewer's Checkpoint dropdown.

    mjlab's local CheckpointManager only lists *.pt files in the loaded
    checkpoint's own directory. This wraps ViserPlayViewer with a "Run"
    dropdown (inserted just above the existing "Checkpoint" dropdown in the
    Checkpoints tab). Picking a run repopulates the checkpoint dropdown with
    that run's checkpoints and hot-swaps to its latest one. Runs from older
    lineages with a different observation space fail to load safely: the
    previous policy keeps running and the Run dropdown snaps back.
    """
    import time as _time

    import mjlab.scripts.play as _mjplay
    from mjlab.viewer.base import ViewerAction
    from mjlab.viewer.viser.viewer import format_time_ago

    resume_path = Path(checkpoint_file).resolve()
    runs_root = resume_path.parent.parent
    base_viewer = _mjplay.ViserPlayViewer
    state = {"run": resume_path.parent.name}

    def _step_of(f: Path) -> int:
        try:
            return int(f.stem.split("_")[1])
        except (IndexError, ValueError):
            return 0

    def _list_runs() -> list[str]:
        runs = sorted(
            d.name
            for d in runs_root.iterdir()
            if d.is_dir() and next(d.glob("model_*.pt"), None) is not None
        )[-30:]
        if state["run"] not in runs:
            runs.append(state["run"])
        return runs

    class _MultiRunViserViewer(base_viewer):  # type: ignore[misc,valid-type]
        def __init__(self, env, policy, checkpoint_manager=None):
            self._run_dropdown = None
            self._run_guard = False
            mgr = checkpoint_manager
            if mgr is not None:
                orig_load = mgr.load_checkpoint

                def fetch_available() -> list[tuple[str, str]]:
                    now = _time.time()
                    self._refresh_run_options()
                    models = sorted(
                        (runs_root / state["run"]).glob("model_*.pt"),
                        key=_step_of,
                    )
                    return [
                        (f.name, format_time_ago(int(now - f.stat().st_mtime)))
                        for f in models
                    ]

                def load_checkpoint(name: str):
                    # Manager's original loader resolves relative to the
                    # resume checkpoint's directory; hop up to the run root.
                    previous_run = getattr(mgr, "_loaded_run", state["run"])
                    previous_name = getattr(
                        mgr, "_loaded_name", resume_path.name
                    )
                    try:
                        policy = orig_load(f"../{state['run']}/{name}")
                        mgr._loaded_run = state["run"]
                        mgr._loaded_name = name
                        return policy
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"[WARN] Could not load {state['run']}/{name} "
                            f"({exc}); keeping the current policy. Runs from "
                            "older lineages may have a different observation "
                            "space."
                        )
                        state["run"] = previous_run
                        self._set_run_dropdown(previous_run)
                        mgr.current_name = previous_name
                        return orig_load(
                            f"../{previous_run}/{previous_name}"
                        )

                mgr._loaded_run = state["run"]
                mgr._loaded_name = resume_path.name
                mgr.fetch_available = fetch_available
                mgr.load_checkpoint = load_checkpoint
            super().__init__(env, policy, checkpoint_manager=mgr)

        def _set_run_dropdown(self, run_name: str) -> None:
            if self._run_dropdown is None:
                return
            self._run_guard = True
            try:
                if run_name not in self._run_dropdown.options:
                    self._run_dropdown.options = (
                        *self._run_dropdown.options,
                        run_name,
                    )
                self._run_dropdown.value = run_name
            finally:
                self._run_guard = False

        def _refresh_run_options(self) -> None:
            if self._run_dropdown is None:
                return
            runs = _list_runs()
            if list(self._run_dropdown.options) != runs:
                self._run_guard = True
                try:
                    self._run_dropdown.options = runs
                    self._run_dropdown.value = state["run"]
                finally:
                    self._run_guard = False

        def setup(self) -> None:
            # Intercept creation of the "Checkpoint" dropdown so the "Run"
            # dropdown lands directly above it, inside the same tab.
            gui = self._server.gui
            orig_add_dropdown = gui.add_dropdown

            def add_dropdown(label, *args, **kwargs):
                if label == "Checkpoint" and self._run_dropdown is None:
                    runs = _list_runs()
                    self._run_dropdown = orig_add_dropdown(
                        "Run", options=runs, initial_value=state["run"]
                    )

                    @self._run_dropdown.on_update
                    def _(_) -> None:
                        if self._run_guard:
                            return
                        selected = self._run_dropdown.value
                        if selected != state["run"]:
                            state["run"] = selected
                            # Checkpoint names repeat across runs (e.g.
                            # model_1000.pt); blank current_name so the
                            # viewer's same-name check can't skip the load.
                            if self._ckpt_mgr is not None:
                                self._ckpt_mgr.current_name = ""
                            self._actions.append(
                                (ViewerAction.FETCH_CHECKPOINT, "latest")
                            )

                return orig_add_dropdown(label, *args, **kwargs)

            gui.add_dropdown = add_dropdown
            try:
                super().setup()
            finally:
                del gui.add_dropdown

    _mjplay.ViserPlayViewer = _MultiRunViserViewer


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

    if args.checkpoint_file and args.agent == "trained":
        _install_multirun_checkpoint_browser(args.checkpoint_file)

    if args.viewer_stride > 1:
        print(
            "[WARN]: --viewer-stride is ignored because it only drops visual "
            "updates. Use the Viser Speed controls to fast-forward playback."
        )
    run_play(task_id, cfg)


if __name__ == "__main__":
    main()
