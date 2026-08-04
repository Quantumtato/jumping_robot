"""Command generators for commanded-height balance training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING, Any

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from mjlab_tasks.jumping_robot_balance.mdp.height_curriculum import (
    HEIGHT_RANGE_SCHEDULE,
    scheduled_height_half_width,
)
from mjlab_tasks.jumping_robot_balance.robot_cfg import (
    LINEAR_RANGE_CENTER_M,
    LINEAR_RANGE_HALF_WIDTH_M,
)

if TYPE_CHECKING:
    import viser

    from mjlab.envs import ManagerBasedRlEnv

HEIGHT_COMMAND_NAME = "height"


class HeightCommand(CommandTerm):
    cfg: HeightCommandCfg

    def __init__(self, cfg: HeightCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command = torch.full(
            (self.num_envs, 1),
            LINEAR_RANGE_CENTER_M,
            device=self.device,
        )
        self._pending: SimpleQueue[tuple[int, float]] = SimpleQueue()

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if self.cfg.play:
            self._command[env_ids] = LINEAR_RANGE_CENTER_M
            return
        half_width = scheduled_height_half_width(
            self._env.common_step_counter,
            self.cfg.range_schedule,
        )
        sampled = torch.empty(
            (len(env_ids), 1),
            device=self.device,
        ).uniform_(-half_width, half_width)
        self._command[env_ids] = sampled + LINEAR_RANGE_CENTER_M

    def _update_command(self) -> None:
        while True:
            try:
                env_idx, command = self._pending.get_nowait()
            except Empty:
                break
            self._command[env_idx, 0] = command

    def create_gui(
        self,
        name: str,
        server: viser.ViserServer,
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        del name
        with server.gui.add_folder("Height command"):
            slider = server.gui.add_slider(
                "Height offset (mm)",
                min=-1000.0 * LINEAR_RANGE_HALF_WIDTH_M,
                max=1000.0 * LINEAR_RANGE_HALF_WIDTH_M,
                step=1.0,
                initial_value=0.0,
            )

            @slider.on_update
            def _(event) -> None:
                command = LINEAR_RANGE_CENTER_M + float(event.target.value) / 1000.0
                self._pending.put((get_env_idx(), command))
                if on_change is not None:
                    on_change()
                if request_action is not None:
                    request_action("SINGLE_STEP")


@dataclass(kw_only=True)
class HeightCommandCfg(CommandTermCfg):
    play: bool = False
    range_schedule: tuple[tuple[int, float], ...] = HEIGHT_RANGE_SCHEDULE
    resampling_time_range: tuple[float, float] = (4.0, 8.0)

    def build(self, env: ManagerBasedRlEnv) -> HeightCommand:
        return HeightCommand(self, env)


def build_height_commands(play: bool = False) -> dict[str, CommandTermCfg]:
    return {
        HEIGHT_COMMAND_NAME: HeightCommandCfg(
            play=play,
            resampling_time_range=(1.0e9, 1.0e9) if play else (4.0, 8.0),
        )
    }
