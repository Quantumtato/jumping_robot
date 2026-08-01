"""Interactive playback disturbances for the balance task."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING, Any

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from mjlab_tasks.jumping_robot_balance.robot_cfg import ROBOT_ENTITY_NAME

if TYPE_CHECKING:
    import viser

    from mjlab.envs import ManagerBasedRlEnv


class DisturbanceCommand(CommandTerm):
    cfg: DisturbanceCommandCfg

    def __init__(self, cfg: DisturbanceCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._robot: Entity = env.scene[cfg.entity_name]
        self._command = torch.empty((self.num_envs, 0), device=self.device)
        self._pending: SimpleQueue[
            tuple[int, tuple[float, float, float], tuple[float, float, float]]
        ] = SimpleQueue()

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        del env_ids

    def _update_command(self) -> None:
        while True:
            try:
                env_idx, linear_delta, angular_delta = self._pending.get_nowait()
            except Empty:
                break
            env_ids = torch.tensor([env_idx], dtype=torch.int64, device=self.device)
            velocity = self._robot.data.root_link_vel_w[env_ids].clone()
            velocity[:, :3] += torch.tensor(
                linear_delta,
                dtype=velocity.dtype,
                device=self.device,
            )
            velocity[:, 3:] += torch.tensor(
                angular_delta,
                dtype=velocity.dtype,
                device=self.device,
            )
            self._robot.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)

    def create_gui(
        self,
        name: str,
        server: viser.ViserServer,
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        del name, on_change

        def queue_kick(
            linear: tuple[float, float, float],
            angular: tuple[float, float, float],
        ) -> None:
            self._pending.put((get_env_idx(), linear, angular))
            if request_action is not None:
                request_action("SINGLE_STEP")

        with server.gui.add_folder("Disturbances"):
            server.gui.add_html(
                "<small>Additive world-frame velocity kicks. "
                "Adjust the strength, then click a direction.</small>"
            )
            linear_strength = server.gui.add_slider(
                "Linear kick (m/s)",
                min=0.05,
                max=1.0,
                step=0.05,
                initial_value=0.2,
            )
            linear_buttons = server.gui.add_button_group(
                "Linear direction",
                options=["+X", "-X", "+Y", "-Y"],
            )

            @linear_buttons.on_click
            def _(event) -> None:
                strength = float(linear_strength.value)
                direction = {
                    "+X": (strength, 0.0, 0.0),
                    "-X": (-strength, 0.0, 0.0),
                    "+Y": (0.0, strength, 0.0),
                    "-Y": (0.0, -strength, 0.0),
                }[event.target.value]
                queue_kick(direction, (0.0, 0.0, 0.0))

            angular_strength = server.gui.add_slider(
                "Angular kick (rad/s)",
                min=0.05,
                max=2.0,
                step=0.05,
                initial_value=0.3,
            )
            angular_buttons = server.gui.add_button_group(
                "Angular direction",
                options=["+Roll", "-Roll", "+Pitch", "-Pitch"],
            )

            @angular_buttons.on_click
            def _(event) -> None:
                strength = float(angular_strength.value)
                direction = {
                    "+Roll": (strength, 0.0, 0.0),
                    "-Roll": (-strength, 0.0, 0.0),
                    "+Pitch": (0.0, strength, 0.0),
                    "-Pitch": (0.0, -strength, 0.0),
                }[event.target.value]
                queue_kick((0.0, 0.0, 0.0), direction)


@dataclass(kw_only=True)
class DisturbanceCommandCfg(CommandTermCfg):
    entity_name: str = ROBOT_ENTITY_NAME
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env: ManagerBasedRlEnv) -> DisturbanceCommand:
        return DisturbanceCommand(self, env)


def build_disturbance_commands() -> dict[str, CommandTermCfg]:
    return {"disturbance": DisturbanceCommandCfg()}
