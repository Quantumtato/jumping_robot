"""One-shot jump command and contact-driven phase state."""

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
from mjlab_tasks.jumping_robot_balance.mdp.jump_curriculum import (
    JUMP_HEIGHT_SCHEDULE,
    scheduled_jump_height,
)
from mjlab_tasks.jumping_robot_balance.robot_cfg import ROBOT_ENTITY_NAME

if TYPE_CHECKING:
    import viser

    from mjlab.envs import ManagerBasedRlEnv

JUMP_COMMAND_NAME = "jump"

PHASE_BALANCE = 0
PHASE_TAKEOFF = 1
PHASE_FLIGHT = 2
PHASE_RECOVERY = 3

_ROBOT_CFG = SceneEntityCfg(ROBOT_ENTITY_NAME)


class JumpCommand(CommandTerm):
    """Latch a one-shot jump until contact-based landing recovery completes."""

    cfg: JumpCommandCfg

    def __init__(self, cfg: JumpCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._robot: Entity = env.scene[_ROBOT_CFG.name]
        self._command = torch.zeros((self.num_envs, 4), device=self.device)
        self.phase = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.phase_time = torch.zeros(self.num_envs, device=self.device)
        self.stable_time = torch.zeros(self.num_envs, device=self.device)
        self.airborne_time = torch.zeros(self.num_envs, device=self.device)
        self.baseline_height = torch.zeros(self.num_envs, device=self.device)
        self.target_height = torch.full(
            (self.num_envs,),
            self.cfg.height_schedule[0][1],
            device=self.device,
        )
        self.apex_height = torch.zeros(self.num_envs, device=self.device)
        self.reached_target = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.apex_progress_delta = torch.zeros(self.num_envs, device=self.device)
        self.target_reached_event = torch.zeros(self.num_envs, device=self.device)
        self.landing_quality_event = torch.zeros(self.num_envs, device=self.device)
        self.landing_impact_event = torch.zeros(self.num_envs, device=self.device)
        self.completed_event = torch.zeros(self.num_envs, device=self.device)
        self.missed_event = torch.zeros(self.num_envs, device=self.device)
        self._pending: SimpleQueue[int] = SimpleQueue()
        self.metrics["completed_jumps"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["target_reached"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._clear_jump(env_ids, schedule_next=False)
        self.reached_target[env_ids] = False
        if self.cfg.play:
            self.time_left[env_ids] = math.inf

    def _update_command(self) -> None:
        self._clear_events()
        while True:
            try:
                env_idx = self._pending.get_nowait()
            except Empty:
                break
            self._trigger(
                torch.tensor([env_idx], dtype=torch.long, device=self.device)
            )

        inactive = self.phase == PHASE_BALANCE
        auto_trigger = inactive & (self.time_left <= 0.0)
        if not self.cfg.play and torch.any(auto_trigger):
            self._trigger(auto_trigger.nonzero().flatten())

        contact = foot_ground_contact(self._env)[:, 0] > 0.5
        self._update_takeoff(contact)
        self._update_flight(contact)
        self._update_recovery(contact)
        self._update_command_tensor()

    def compute(self, dt: float) -> None:
        self._update_metrics()
        inactive = self.phase == PHASE_BALANCE
        self.time_left[inactive] -= dt
        self.phase_time[~inactive] += dt
        self._update_command()

    def _trigger(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        inactive = self.phase[env_ids] == PHASE_BALANCE
        env_ids = env_ids[inactive]
        if len(env_ids) == 0:
            return
        self.phase[env_ids] = PHASE_TAKEOFF
        self.phase_time[env_ids] = 0.0
        self.stable_time[env_ids] = 0.0
        self.airborne_time[env_ids] = 0.0
        self.baseline_height[env_ids] = self._robot.data.root_link_pos_w[env_ids, 2]
        self.target_height[env_ids] = scheduled_jump_height(
            self._env.common_step_counter,
            self.cfg.height_schedule,
        )
        self.apex_height[env_ids] = 0.0
        self.reached_target[env_ids] = False
        self.command_counter[env_ids] += 1

    def _update_takeoff(self, contact: torch.Tensor) -> None:
        takeoff = self.phase == PHASE_TAKEOFF
        self.airborne_time[takeoff & ~contact] += self._env.step_dt
        self.airborne_time[takeoff & contact] = 0.0
        lifted = takeoff & (self.airborne_time >= self.cfg.liftoff_debounce_s)
        self.phase[lifted] = PHASE_FLIGHT
        self.phase_time[lifted] = 0.0

        missed = takeoff & (self.phase_time >= self.cfg.takeoff_timeout_s)
        self.missed_event[missed] = 1.0
        self._clear_jump(missed.nonzero().flatten())

    def _update_flight(self, contact: torch.Tensor) -> None:
        flight = self.phase == PHASE_FLIGHT
        height_gain = torch.clamp(
            self._robot.data.root_link_pos_w[:, 2] - self.baseline_height,
            min=0.0,
        )
        normalized_height = torch.clamp(
            height_gain / self.target_height,
            min=0.0,
            max=1.0,
        )
        previous_progress = torch.clamp(
            self.apex_height / self.target_height,
            min=0.0,
            max=1.0,
        )
        self.apex_progress_delta[flight] = torch.clamp(
            normalized_height[flight] - previous_progress[flight],
            min=0.0,
        )
        new_apex = flight & (height_gain > self.apex_height)
        self.apex_height[new_apex] = height_gain[new_apex]
        reached = flight & (previous_progress < 1.0) & (normalized_height >= 1.0)
        self.target_reached_event[reached] = 1.0
        self.reached_target[reached] = True
        self.metrics["target_reached"][reached] += 1.0

        landed = flight & contact
        if torch.any(landed):
            vertical_speed = torch.abs(self._robot.data.root_link_lin_vel_w[:, 2])
            angular_speed = torch.linalg.vector_norm(
                self._robot.data.root_link_ang_vel_b,
                dim=1,
            )
            self.landing_quality_event[landed] = torch.exp(
                -torch.square(vertical_speed[landed] / 0.5)
                - torch.square(angular_speed[landed] / 1.0)
            )
            self.landing_impact_event[landed] = torch.clamp(
                vertical_speed[landed] / 2.0,
                max=1.0,
            )
            self.phase[landed] = PHASE_RECOVERY
            self.phase_time[landed] = 0.0
            self.stable_time[landed] = 0.0

        timed_out = flight & ~landed & (
            self.phase_time >= self.cfg.flight_timeout_s
        )
        self.missed_event[timed_out] = 1.0
        self._clear_jump(timed_out.nonzero().flatten())

    def _update_recovery(self, contact: torch.Tensor) -> None:
        recovery = self.phase == PHASE_RECOVERY
        projected_gravity = self._robot.data.projected_gravity_b
        tilt_l2 = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
        angular_speed = torch.linalg.vector_norm(
            self._robot.data.root_link_ang_vel_b,
            dim=1,
        )
        vertical_speed = torch.abs(self._robot.data.root_link_lin_vel_w[:, 2])
        stable = (
            recovery
            & contact
            & (tilt_l2 <= math.sin(math.radians(10.0)) ** 2)
            & (angular_speed <= 1.0)
            & (vertical_speed <= 0.2)
        )
        self.stable_time[stable] += self._env.step_dt
        self.stable_time[recovery & ~stable] = 0.0

        completed = recovery & (self.stable_time >= self.cfg.recovery_time_s)
        successful = completed & self.reached_target
        unsuccessful = completed & ~self.reached_target
        self.completed_event[successful] = 1.0
        self.missed_event[unsuccessful] = 1.0
        self.metrics["completed_jumps"][successful] += 1.0
        self._clear_jump(completed.nonzero().flatten())

        timed_out = recovery & (self.phase_time >= self.cfg.recovery_timeout_s)
        self.missed_event[timed_out] = 1.0
        self._clear_jump(timed_out.nonzero().flatten())

    def _clear_jump(
        self,
        env_ids: torch.Tensor,
        schedule_next: bool = True,
    ) -> None:
        if len(env_ids) == 0:
            return
        self.phase[env_ids] = PHASE_BALANCE
        self.phase_time[env_ids] = 0.0
        self.stable_time[env_ids] = 0.0
        self.airborne_time[env_ids] = 0.0
        if schedule_next:
            if self.cfg.play:
                self.time_left[env_ids] = math.inf
            else:
                self.time_left[env_ids].uniform_(*self.cfg.trigger_interval_s)
        self._update_command_tensor()

    def _clear_events(self) -> None:
        self.apex_progress_delta.zero_()
        self.target_reached_event.zero_()
        self.landing_quality_event.zero_()
        self.landing_impact_event.zero_()
        self.completed_event.zero_()
        self.missed_event.zero_()

    def _update_command_tensor(self) -> None:
        self._command[:, 0] = (self.phase != PHASE_BALANCE).float()
        self._command[:, 1:3] = 0.0
        self._command[:, 3] = self.phase.float() / float(PHASE_RECOVERY)

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
                "<small>Stage 2 uses a vertical one-shot command. The command "
                "stays latched through landing recovery.</small>"
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
    height_schedule: tuple[tuple[int, float], ...] = JUMP_HEIGHT_SCHEDULE
    trigger_interval_s: tuple[float, float] = (2.0, 4.0)
    liftoff_debounce_s: float = 0.010
    takeoff_timeout_s: float = 1.0
    flight_timeout_s: float = 1.5
    recovery_time_s: float = 0.5
    recovery_timeout_s: float = 2.0
    resampling_time_range: tuple[float, float] = (2.0, 4.0)

    def build(self, env: ManagerBasedRlEnv) -> JumpCommand:
        return JumpCommand(self, env)


def build_jump_commands(play: bool = False) -> dict[str, CommandTermCfg]:
    return {
        JUMP_COMMAND_NAME: JumpCommandCfg(
            play=play,
            resampling_time_range=(1.0e9, 1.0e9) if play else (2.0, 4.0),
        )
    }
