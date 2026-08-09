"""Binary one-shot jump command without a generated motion phase."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING, Any

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_tasks.jumping_robot_balance.mdp.contact import foot_ground_contact
from mjlab_tasks.jumping_robot_balance.robot_cfg import (
    FLYWHEEL_X_JOINT,
    FLYWHEEL_Y_JOINT,
    MAX_FLYWHEEL_SPEED_RAD_S,
    ROBOT_ENTITY_NAME,
)

if TYPE_CHECKING:
    import viser

    from mjlab.envs import ManagerBasedRlEnv

JUMP_COMMAND_NAME = "jump"
_ROBOT_CFG = SceneEntityCfg(ROBOT_ENTITY_NAME)
_WHEEL_PHASE_NAMES = (
    "pre_jump",
    "jump_grounded",
    "flight",
    "post_landing",
)


class JumpCommand(CommandTerm):
    """Hold a binary jump request for a fixed attempt window."""

    cfg: JumpCommandCfg

    def __init__(self, cfg: JumpCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._robot: Entity = env.scene[_ROBOT_CFG.name]
        flywheel_joint_ids, flywheel_joint_names = self._robot.find_joints(
            (FLYWHEEL_X_JOINT, FLYWHEEL_Y_JOINT),
            preserve_order=True,
        )
        if len(flywheel_joint_ids) != 2:
            raise ValueError(
                "Expected two flywheel joints, found "
                f"{flywheel_joint_names}."
            )
        self._flywheel_joint_ids = torch.tensor(
            flywheel_joint_ids,
            dtype=torch.long,
            device=self.device,
        )
        self._command = torch.zeros((self.num_envs, 1), device=self.device)
        self.active_time = torch.zeros(self.num_envs, device=self.device)
        self.baseline_height = torch.zeros(self.num_envs, device=self.device)
        self.apex_height = torch.zeros(self.num_envs, device=self.device)
        self.apex_progress_delta = torch.zeros(self.num_envs, device=self.device)
        self.has_triggered = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.was_airborne = torch.zeros_like(self.has_triggered)
        self.has_landed = torch.zeros_like(self.has_triggered)
        self.landing_event = torch.zeros(self.num_envs, device=self.device)
        self.airborne_time = torch.zeros(self.num_envs, device=self.device)
        self.landing_recovery_time = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.landing_recovery_complete = torch.zeros_like(self.has_triggered)
        self.landing_recovery_success_event = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.landing_impact_speed = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self._previous_root_vertical_velocity = (
            self._robot.data.root_link_lin_vel_w[:, 2].clone()
        )
        phase_count = len(_WHEEL_PHASE_NAMES)
        self._wheel_sample_count = torch.zeros(
            self.num_envs,
            phase_count,
            device=self.device,
        )
        self._target_speed_abs_sum = torch.zeros(
            self.num_envs,
            phase_count,
            2,
            device=self.device,
        )
        self._target_speed_abs_peak = torch.zeros_like(self._target_speed_abs_sum)
        self._target_speed_saturation_count = torch.zeros_like(
            self._target_speed_abs_sum
        )
        self._speed_abs_sum = torch.zeros_like(self._target_speed_abs_sum)
        self._speed_abs_peak = torch.zeros_like(self._target_speed_abs_sum)
        self._speed_saturation_count = torch.zeros_like(self._target_speed_abs_sum)
        self._pending: SimpleQueue[int] = SimpleQueue()
        self.metrics["apex_height"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["landing_impact_speed"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["landed"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["landing_recovery_success"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["landing_recovery_time"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        for phase_name in _WHEEL_PHASE_NAMES:
            for wheel_name in ("x", "y"):
                self.metrics[
                    f"flywheel_{phase_name}_{wheel_name}_mean_abs_target_speed"
                ] = torch.zeros(self.num_envs, device=self.device)
                self.metrics[
                    f"flywheel_{phase_name}_{wheel_name}_peak_abs_target_speed"
                ] = torch.zeros(self.num_envs, device=self.device)
                self.metrics[
                    f"flywheel_{phase_name}_{wheel_name}_target_speed_saturation_fraction"
                ] = torch.zeros(self.num_envs, device=self.device)
                self.metrics[
                    f"flywheel_{phase_name}_{wheel_name}_mean_abs_speed"
                ] = torch.zeros(self.num_envs, device=self.device)
                self.metrics[
                    f"flywheel_{phase_name}_{wheel_name}_peak_abs_speed"
                ] = torch.zeros(self.num_envs, device=self.device)
                self.metrics[
                    f"flywheel_{phase_name}_{wheel_name}_speed_saturation_fraction"
                ] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        self.metrics["apex_height"][:] = self.apex_height
        self.metrics["landing_impact_speed"][:] = self.landing_impact_speed
        self.metrics["landed"][:] = self.has_landed.float()
        self.metrics["landing_recovery_success"][:] = (
            self.landing_recovery_complete.float()
        )
        self.metrics["landing_recovery_time"][:] = self.landing_recovery_time
        sample_count = self._wheel_sample_count.clamp_min(1.0).unsqueeze(-1)
        target_speed_mean = self._target_speed_abs_sum / sample_count
        target_speed_saturation = self._target_speed_saturation_count / sample_count
        speed_mean = self._speed_abs_sum / sample_count
        speed_saturation = self._speed_saturation_count / sample_count
        for phase_id, phase_name in enumerate(_WHEEL_PHASE_NAMES):
            for wheel_id, wheel_name in enumerate(("x", "y")):
                self.metrics[
                    f"flywheel_{phase_name}_{wheel_name}_mean_abs_target_speed"
                ][:] = target_speed_mean[:, phase_id, wheel_id]
                self.metrics[
                    f"flywheel_{phase_name}_{wheel_name}_peak_abs_target_speed"
                ][:] = self._target_speed_abs_peak[:, phase_id, wheel_id]
                self.metrics[
                    f"flywheel_{phase_name}_{wheel_name}_target_speed_saturation_fraction"
                ][:] = target_speed_saturation[:, phase_id, wheel_id]
                self.metrics[
                    f"flywheel_{phase_name}_{wheel_name}_mean_abs_speed"
                ][:] = speed_mean[:, phase_id, wheel_id]
                self.metrics[
                    f"flywheel_{phase_name}_{wheel_name}_peak_abs_speed"
                ][:] = self._speed_abs_peak[:, phase_id, wheel_id]
                self.metrics[
                    f"flywheel_{phase_name}_{wheel_name}_speed_saturation_fraction"
                ][:] = speed_saturation[:, phase_id, wheel_id]

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._command[env_ids, 0] = 0.0
        self.active_time[env_ids] = 0.0
        self.baseline_height[env_ids] = 0.0
        self.apex_height[env_ids] = 0.0
        self.apex_progress_delta[env_ids] = 0.0
        self.has_triggered[env_ids] = False
        self.was_airborne[env_ids] = False
        self.has_landed[env_ids] = False
        self.landing_event[env_ids] = 0.0
        self.airborne_time[env_ids] = 0.0
        self.landing_recovery_time[env_ids] = 0.0
        self.landing_recovery_complete[env_ids] = False
        self.landing_recovery_success_event[env_ids] = 0.0
        self.landing_impact_speed[env_ids] = 0.0
        self._wheel_sample_count[env_ids] = 0.0
        self._target_speed_abs_sum[env_ids] = 0.0
        self._target_speed_abs_peak[env_ids] = 0.0
        self._target_speed_saturation_count[env_ids] = 0.0
        self._speed_abs_sum[env_ids] = 0.0
        self._speed_abs_peak[env_ids] = 0.0
        self._speed_saturation_count[env_ids] = 0.0
        self._previous_root_vertical_velocity[env_ids] = (
            self._robot.data.root_link_lin_vel_w[env_ids, 2]
        )
        if self.cfg.play:
            self.time_left[env_ids] = math.inf

    def _update_command(self, dt: float) -> None:
        self.apex_progress_delta.zero_()
        self.landing_event.zero_()
        self.landing_recovery_success_event.zero_()
        while True:
            try:
                env_idx = self._pending.get_nowait()
            except Empty:
                break
            self._trigger(
                torch.tensor([env_idx], dtype=torch.long, device=self.device)
            )

        inactive = self._command[:, 0] < 0.5
        auto_trigger = inactive & ~self.has_triggered & (self.time_left <= 0.0)
        if self.cfg.auto_trigger and not self.cfg.play and torch.any(auto_trigger):
            self._trigger(auto_trigger.nonzero().flatten())

        contact = foot_ground_contact(self._env)[:, 0] > 0.5
        self.airborne_time[contact] = 0.0
        self.airborne_time[~contact] = torch.clamp(
            self.airborne_time[~contact] + dt,
            max=self.cfg.airborne_time_observation_max_s,
        )
        tracking_landing = self.has_triggered & ~self.has_landed
        self.was_airborne |= tracking_landing & ~contact
        touchdown = tracking_landing & self.was_airborne & contact
        self.landing_event[touchdown] = 1.0
        self.landing_impact_speed[touchdown] = torch.clamp(
            -self._previous_root_vertical_velocity[touchdown],
            min=0.0,
        )
        self.has_landed[touchdown] = True
        # Return control to the standing-height objective as soon as flight ends.
        self._finish_jump(touchdown.nonzero().flatten())

        recovering = self.has_landed & ~self.landing_recovery_complete
        horizontal_gravity = torch.linalg.vector_norm(
            self._robot.data.projected_gravity_b[:, :2],
            dim=1,
        )
        angular_speed = torch.linalg.vector_norm(
            self._robot.data.root_link_ang_vel_b,
            dim=1,
        )
        upright_and_quiet = (
            horizontal_gravity
            <= math.sin(math.radians(self.cfg.landing_recovery_max_tilt_deg))
        ) & (angular_speed <= self.cfg.landing_recovery_max_ang_vel_rad_s)
        stable_recovery = recovering & contact & upright_and_quiet
        self.landing_recovery_time[stable_recovery] += dt
        self.landing_recovery_time[recovering & ~stable_recovery] = 0.0
        recovered = stable_recovery & (
            self.landing_recovery_time
            >= self.cfg.landing_recovery_duration_s
        )
        self.landing_recovery_complete[recovered] = True
        self.landing_recovery_success_event[recovered] = 1.0

        active = self._command[:, 0] > 0.5
        height_gain = torch.clamp(
            self._robot.data.root_link_pos_w[:, 2] - self.baseline_height,
            min=0.0,
        )
        new_apex = torch.maximum(self.apex_height, height_gain)
        self.apex_progress_delta[active] = (
            new_apex[active] - self.apex_height[active]
        )
        self.apex_height[active] = new_apex[active]

        expired = active & (self.active_time >= self.cfg.command_duration_s)
        self._finish_jump(expired.nonzero().flatten())
        self._previous_root_vertical_velocity[:] = (
            self._robot.data.root_link_lin_vel_w[:, 2]
        )

    def _accumulate_wheel_metrics(self) -> None:
        contact = foot_ground_contact(self._env)[:, 0] > 0.5
        phase = torch.full(
            (self.num_envs,),
            2,
            dtype=torch.long,
            device=self.device,
        )
        phase[~self.has_triggered] = 0
        phase[(self._command[:, 0] > 0.5) & contact] = 1
        phase[self.has_landed] = 3

        target_speed = torch.clamp(
            self._env.action_manager.action[:, :2],
            min=-1.0,
            max=1.0,
        ).abs() * MAX_FLYWHEEL_SPEED_RAD_S
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._wheel_sample_count[env_ids, phase] += 1.0
        self._target_speed_abs_sum[env_ids, phase] += target_speed
        self._target_speed_abs_peak[env_ids, phase] = torch.maximum(
            self._target_speed_abs_peak[env_ids, phase],
            target_speed,
        )
        self._target_speed_saturation_count[env_ids, phase] += (
            target_speed >= 0.99 * MAX_FLYWHEEL_SPEED_RAD_S
        )
        speed = self._robot.data.joint_vel[:, self._flywheel_joint_ids].abs()
        self._speed_abs_sum[env_ids, phase] += speed
        self._speed_abs_peak[env_ids, phase] = torch.maximum(
            self._speed_abs_peak[env_ids, phase],
            speed,
        )
        self._speed_saturation_count[env_ids, phase] += (
            speed >= 0.99 * MAX_FLYWHEEL_SPEED_RAD_S
        )

    def compute(self, dt: float) -> None:
        self._accumulate_wheel_metrics()
        active = self._command[:, 0] > 0.5
        self.time_left[~active] -= dt
        self.active_time[active] += dt
        self._update_command(dt)
        self._update_metrics()

    def _trigger(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        inactive = self._command[env_ids, 0] < 0.5
        env_ids = env_ids[inactive]
        if len(env_ids) == 0:
            return
        self._command[env_ids, 0] = 1.0
        self.active_time[env_ids] = 0.0
        self.has_triggered[env_ids] = True
        self.was_airborne[env_ids] = False
        self.has_landed[env_ids] = False
        self.landing_event[env_ids] = 0.0
        self.airborne_time[env_ids] = 0.0
        self.landing_recovery_time[env_ids] = 0.0
        self.landing_recovery_complete[env_ids] = False
        self.landing_recovery_success_event[env_ids] = 0.0
        self.landing_impact_speed[env_ids] = 0.0
        self._previous_root_vertical_velocity[env_ids] = (
            self._robot.data.root_link_lin_vel_w[env_ids, 2]
        )
        self.baseline_height[env_ids] = self._robot.data.root_link_pos_w[
            env_ids, 2
        ]
        self.apex_height[env_ids] = 0.0
        self.command_counter[env_ids] += 1

    def _finish_jump(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self._command[env_ids, 0] = 0.0
        self.active_time[env_ids] = 0.0
        self.time_left[env_ids] = math.inf

    def create_gui(
        self,
        name: str,
        server: viser.ViserServer,
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        del name
        with server.gui.add_folder("Jump command"):
            server.gui.add_html(
                "<small>Sends one binary jump request. No motion phase or "
                "recovery state is generated.</small>"
            )
            button = server.gui.add_button("Trigger vertical jump")

            @button.on_click
            def _(_) -> None:
                self._pending.put(get_env_idx())
                if on_change is not None:
                    on_change()
                if request_action is not None:
                    request_action("SINGLE_STEP")


@dataclass(kw_only=True)
class JumpCommandCfg(CommandTermCfg):
    play: bool = False
    auto_trigger: bool = True
    command_duration_s: float = 1.5
    resampling_time_range: tuple[float, float] = (2.0, 4.0)
    landing_recovery_duration_s: float = 0.5
    landing_recovery_max_tilt_deg: float = 12.0
    landing_recovery_max_ang_vel_rad_s: float = 3.0
    airborne_time_observation_max_s: float = 1.0

    def build(self, env: ManagerBasedRlEnv) -> JumpCommand:
        return JumpCommand(self, env)


def build_jump_commands(
    play: bool = False,
    auto_trigger: bool = True,
) -> dict[str, CommandTermCfg]:
    return {
        JUMP_COMMAND_NAME: JumpCommandCfg(
            play=play,
            auto_trigger=auto_trigger,
            resampling_time_range=(1.0e9, 1.0e9) if play else (2.0, 4.0),
        )
    }
