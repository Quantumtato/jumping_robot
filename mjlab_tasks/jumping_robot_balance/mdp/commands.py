"""Command generators for commanded-height balance training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING, Any

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab_tasks.jumping_robot_balance.mdp.height_curriculum import (
    HEIGHT_RANGE_SCHEDULE,
    scheduled_height_half_width,
)
from mjlab_tasks.jumping_robot_balance.mdp.contact import (
    foot_ground_contact,
    foot_height_w,
)
from mjlab_tasks.jumping_robot_balance.robot_cfg import (
    LINEAR_RANGE_HALF_WIDTH_M,
    LINEAR_RANGE_CENTER_M,
    LINEAR_RANGE_MAX_M,
    LINEAR_RANGE_MIN_M,
    ROBOT_ENTITY_NAME,
)

if TYPE_CHECKING:
    import viser

    from mjlab.envs import ManagerBasedRlEnv

HEIGHT_COMMAND_NAME = "height"
PLANAR_VELOCITY_COMMAND_NAME = "planar_velocity"
PLANAR_VELOCITY_SCALE_M_S = 0.40
_GRAVITY_M_S2 = 9.81
IMU_ACCEL_SCALE_M_S2 = 3.0 * _GRAVITY_M_S2


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
        endpoint_mask = torch.rand(
            (len(env_ids), 1),
            device=self.device,
        ) < self.cfg.endpoint_sample_probability
        endpoint = torch.where(
            torch.rand((len(env_ids), 1), device=self.device) < 0.5,
            -half_width,
            half_width,
        )
        sampled = torch.where(endpoint_mask, endpoint, sampled)
        self._command[env_ids] = sampled + LINEAR_RANGE_CENTER_M

    def _update_command(self) -> None:
        while True:
            try:
                env_idx, command = self._pending.get_nowait()
            except Empty:
                break
            self._command[env_idx, 0] = min(
                max(command, LINEAR_RANGE_MIN_M),
                LINEAR_RANGE_MAX_M,
            )

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
            travel_mm = 1000.0 * (LINEAR_RANGE_MAX_M - LINEAR_RANGE_MIN_M)
            server.gui.add_html(
                "<small>Absolute actuator extension: 0 mm is minimum travel "
                f"and {travel_mm:.0f} mm is maximum travel.</small>"
            )
            slider = server.gui.add_slider(
                "Linear extension (mm)",
                min=0.0,
                max=travel_mm,
                step=1.0,
                initial_value=1000.0
                * (LINEAR_RANGE_CENTER_M - LINEAR_RANGE_MIN_M),
            )

            @slider.on_update
            def _(event) -> None:
                command = LINEAR_RANGE_MIN_M + float(event.target.value) / 1000.0
                self._pending.put((get_env_idx(), command))
                if on_change is not None:
                    on_change()
                if request_action is not None:
                    request_action("SINGLE_STEP")


@dataclass(kw_only=True)
class HeightCommandCfg(CommandTermCfg):
    play: bool = False
    range_schedule: tuple[tuple[int, float], ...] = HEIGHT_RANGE_SCHEDULE
    endpoint_sample_probability: float = 0.2
    resampling_time_range: tuple[float, float] = (4.0, 8.0)

    def build(self, env: ManagerBasedRlEnv) -> HeightCommand:
        return HeightCommand(self, env)


def build_height_commands(
    play: bool = False,
    full_stroke: bool = False,
) -> dict[str, CommandTermCfg]:
    return {
        HEIGHT_COMMAND_NAME: HeightCommandCfg(
            play=play,
            range_schedule=(
                ((0, LINEAR_RANGE_HALF_WIDTH_M),)
                if full_stroke
                else HEIGHT_RANGE_SCHEDULE
            ),
            resampling_time_range=(1.0e9, 1.0e9) if play else (4.0, 8.0),
        )
    }


class PlanarVelocityCommand(CommandTerm):
    """World-frame horizontal velocity request for hopping navigation."""

    cfg: "PlanarVelocityCommandCfg"

    def __init__(self, cfg: "PlanarVelocityCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._robot: Entity = env.scene[ROBOT_ENTITY_NAME]
        self._command = torch.zeros((self.num_envs, 2), device=self.device)
        self._pending_command = torch.zeros_like(self._command)
        self._has_pending_command = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._pending: SimpleQueue[tuple[int, float, float]] = SimpleQueue()
        self._gravity_w = torch.tensor(
            (0.0, 0.0, -_GRAVITY_M_S2),
            device=self.device,
        )
        self._previous_root_velocity = (
            self._robot.data.root_link_lin_vel_w.clone()
        )
        self.imu_specific_force_b = torch.zeros(
            (self.num_envs, 3),
            device=self.device,
        )
        self.average_planar_velocity = (
            self._robot.data.root_link_lin_vel_w[:, :2].clone()
        )
        contact = foot_ground_contact(env)[:, 0] > 0.5
        self._was_grounded = contact
        self.takeoff_event = torch.zeros(self.num_envs, device=self.device)
        self.touchdown_event = torch.zeros(self.num_envs, device=self.device)
        self.touchdown_impact_speed = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.grounded_foot_height = foot_height_w(env).clone()
        self.metrics["command_speed"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["measured_speed"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["velocity_error"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["average_velocity_error"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["takeoff"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["touchdown"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["airborne"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["touchdown_impact_speed"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self._speed_level = 0
        self._speed_level_time_s = 0.0
        # Start pessimistic so the first level only advances on real evidence.
        self._tracking_error_ema = torch.full(
            (),
            2.0 * cfg.speed_curriculum_error_threshold_m_s,
            device=self.device,
        )
        self._moving_command_speed_ema = torch.full(
            (),
            cfg.max_speed_m_s,
            device=self.device,
        )
        if cfg.speed_curriculum_caps is not None:
            self.metrics["speed_curriculum_cap"] = torch.zeros(
                self.num_envs,
                device=self.device,
            )

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def current_max_speed(self) -> float:
        caps = self.cfg.speed_curriculum_caps
        if caps is None:
            return self.cfg.max_speed_m_s
        return caps[self._speed_level]

    def compute(self, dt: float) -> None:
        self._update_imu_and_average(dt)
        self._update_speed_curriculum(dt)
        super().compute(dt)

    def _speed_gate_threshold_m_s(self) -> float:
        """Error the moving-env EMA must beat before the cap advances.

        The relative gate scales with the mean commanded speed, so a policy
        that hops in place (error equal to the command magnitude) can never
        pass it. v17: an absolute floor keeps the threshold from shrinking
        below the physical noise floor at low caps, where the relative gate
        became mathematically unpassable (v16 stalled at the 0.05 cap).
        """
        if self.cfg.speed_curriculum_relative_threshold is None:
            return self.cfg.speed_curriculum_error_threshold_m_s
        return max(
            self.cfg.speed_curriculum_error_floor_m_s,
            self.cfg.speed_curriculum_relative_threshold
            * float(self._moving_command_speed_ema),
        )

    def _update_speed_curriculum(self, dt: float) -> None:
        """Raise the command speed cap once tracking error stays low."""
        caps = self.cfg.speed_curriculum_caps
        if caps is None or self.cfg.play or dt <= 0.0:
            return
        # v20: promotions can only be earned against live commands. With
        # commands zeroed before command_start_step, tracking error is ~0 and
        # the gate would promote on pure dwell time (v19 reached the 0.25 cap
        # during the balance phase without ever tracking a command).
        if (
            self.cfg.command_start_step is not None
            and self._env.common_step_counter < self.cfg.command_start_step
        ):
            return
        moving = (
            torch.linalg.vector_norm(self._command, dim=1) > 0.01
        )
        if moving.any():
            error = torch.linalg.vector_norm(
                self.average_planar_velocity - self._command,
                dim=1,
            )[moving].mean()
            command_speed = torch.linalg.vector_norm(
                self._command,
                dim=1,
            )[moving].mean()
            blend = dt / (dt + self.cfg.speed_curriculum_ema_tau_s)
            self._tracking_error_ema += blend * (
                error - self._tracking_error_ema
            )
            self._moving_command_speed_ema += blend * (
                command_speed - self._moving_command_speed_ema
            )
        self._speed_level_time_s += dt
        if (
            self._speed_level >= len(caps) - 1
            or self._speed_level_time_s
            < self.cfg.speed_curriculum_min_level_time_s
            or self._tracking_error_ema > self._speed_gate_threshold_m_s()
        ):
            return
        self._speed_level += 1
        self._speed_level_time_s = 0.0
        # Reset pessimistic against the new, faster command distribution.
        self._tracking_error_ema.fill_(caps[self._speed_level])
        print(
            f"[INFO]: Speed curriculum advanced to level {self._speed_level} "
            f"(command cap {caps[self._speed_level]:.2f} m/s)."
        )

    def _update_imu_and_average(self, dt: float) -> None:
        """Finite-difference IMU specific force and hop-averaged velocity."""
        if dt <= 0.0:
            # The env calls command compute with dt=0 during reset; skip the
            # finite difference to avoid a 0/0 NaN.
            return
        velocity = torch.nan_to_num(self._robot.data.root_link_lin_vel_w)
        quat = torch.nan_to_num(self._robot.data.root_link_quat_w)
        contact = foot_ground_contact(self._env)[:, 0] > 0.5
        self.takeoff_event[:] = (self._was_grounded & ~contact).float()
        self.touchdown_event[:] = (~self._was_grounded & contact).float()
        self.touchdown_impact_speed[:] = (
            torch.clamp(-self._previous_root_velocity[:, 2], min=0.0)
            * self.touchdown_event
        )
        foot_height = foot_height_w(self._env)
        self.grounded_foot_height[contact] = foot_height[contact]
        self._was_grounded[:] = contact
        acceleration_w = (velocity - self._previous_root_velocity) / dt
        self._previous_root_velocity[:] = velocity
        self.imu_specific_force_b[:] = quat_apply_inverse(
            quat,
            acceleration_w - self._gravity_w,
        )
        blend = dt / (dt + self.cfg.velocity_average_tau_s)
        self.average_planar_velocity += blend * (
            velocity[:, :2] - self.average_planar_velocity
        )

    def _update_metrics(self) -> None:
        measured_velocity = self._robot.data.root_link_lin_vel_w[:, :2]
        self.metrics["command_speed"][:] = torch.linalg.vector_norm(
            self._command,
            dim=1,
        )
        self.metrics["measured_speed"][:] = torch.linalg.vector_norm(
            measured_velocity,
            dim=1,
        )
        self.metrics["velocity_error"][:] = torch.linalg.vector_norm(
            measured_velocity - self._command,
            dim=1,
        )
        self.metrics["average_velocity_error"][:] = torch.linalg.vector_norm(
            self.average_planar_velocity - self._command,
            dim=1,
        )
        self.metrics["takeoff"][:] = self.takeoff_event
        self.metrics["touchdown"][:] = self.touchdown_event
        self.metrics["airborne"][:] = 1.0 - foot_ground_contact(self._env)[:, 0]
        self.metrics["touchdown_impact_speed"][:] = self.touchdown_impact_speed
        if self.cfg.speed_curriculum_caps is not None:
            self.metrics["speed_curriculum_cap"].fill_(
                self.current_max_speed()
            )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Robot state can still be NaN while the reset pipeline runs, so
        # sanitize here; the per-step update rewrites everything next step.
        velocity = torch.nan_to_num(
            self._robot.data.root_link_lin_vel_w[env_ids]
        )
        self._previous_root_velocity[env_ids] = velocity
        self.average_planar_velocity[env_ids] = velocity[:, :2]
        self.imu_specific_force_b[env_ids] = 0.0
        if self.cfg.play:
            # v14 player fix: episode resets must not wipe knob input. The
            # command buffer already holds whatever the GUI last requested.
            return
        if (
            self.cfg.command_start_step is not None
            and self._env.common_step_counter < self.cfg.command_start_step
        ):
            # Earlier pipeline phases train balance and hopping in place.
            self._command[env_ids] = 0.0
            return
        max_speed = self.current_max_speed()
        sampled = torch.empty((len(env_ids), 2), device=self.device).uniform_(
            -max_speed,
            max_speed,
        )
        stationary = (
            torch.rand((len(env_ids), 1), device=self.device)
            < self.cfg.stationary_probability
        )
        sampled = torch.where(
            stationary,
            torch.zeros_like(sampled),
            sampled,
        )
        self._set_or_queue_command(env_ids, sampled)

    def _set_or_queue_command(
        self,
        env_ids: torch.Tensor,
        command: torch.Tensor,
    ) -> None:
        # v14 player fix: knob input applies immediately in play mode; the
        # touchdown queue otherwise delays commands indefinitely for a policy
        # that rarely takes off.
        if self.cfg.play or not self.cfg.apply_only_when_grounded:
            self._command[env_ids] = command
            self._has_pending_command[env_ids] = False
            return
        grounded = foot_ground_contact(self._env)[env_ids, 0] > 0.5
        grounded_ids = env_ids[grounded]
        airborne_ids = env_ids[~grounded]
        self._command[grounded_ids] = command[grounded]
        self._has_pending_command[grounded_ids] = False
        self._pending_command[airborne_ids] = command[~grounded]
        self._has_pending_command[airborne_ids] = True

    def _update_command(self) -> None:
        while True:
            try:
                env_idx, vx, vy = self._pending.get_nowait()
            except Empty:
                break
            command = torch.tensor(
                (vx, vy),
                device=self.device,
            ).clamp(-self.cfg.max_speed_m_s, self.cfg.max_speed_m_s)
            self._set_or_queue_command(
                torch.tensor([env_idx], device=self.device),
                command.unsqueeze(0),
            )
        if self.cfg.apply_only_when_grounded:
            grounded = foot_ground_contact(self._env)[:, 0] > 0.5
            apply_pending = self._has_pending_command & grounded
            self._command[apply_pending] = self._pending_command[apply_pending]
            self._has_pending_command[apply_pending] = False

    def create_gui(
        self,
        name: str,
        server: viser.ViserServer,
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        del name
        with server.gui.add_folder("Navigation command"):
            server.gui.add_html(
                "<small>World-frame velocity request. +X is forward and +Y is "
                "left in the terrain frame. A nonzero request starts the "
                "navigation hop routine.</small>"
            )
            forward = server.gui.add_slider(
                "World X velocity (m/s)",
                min=-self.cfg.max_speed_m_s,
                max=self.cfg.max_speed_m_s,
                step=0.01,
                initial_value=0.0,
            )
            lateral = server.gui.add_slider(
                "World Y velocity (m/s)",
                min=-self.cfg.max_speed_m_s,
                max=self.cfg.max_speed_m_s,
                step=0.01,
                initial_value=0.0,
            )

            def update_command() -> None:
                self._pending.put(
                    (
                        get_env_idx(),
                        float(forward.value),
                        float(lateral.value),
                    )
                )
                if on_change is not None:
                    on_change()
                if request_action is not None:
                    request_action("SINGLE_STEP")

            @forward.on_update
            def _(_) -> None:
                update_command()

            @lateral.on_update
            def _(_) -> None:
                update_command()


@dataclass(kw_only=True)
class PlanarVelocityCommandCfg(CommandTermCfg):
    play: bool = False
    max_speed_m_s: float = PLANAR_VELOCITY_SCALE_M_S
    stationary_probability: float = 0.35
    apply_only_when_grounded: bool = False
    resampling_time_range: tuple[float, float] = (3.0, 6.0)
    # Time constant that averages planar velocity across roughly one hop
    # cycle so tracking rewards ignore intra-hop velocity oscillation.
    velocity_average_tau_s: float = 1.0
    # Sample zero commands until this common_step_counter value (full-pipeline
    # training keeps commands at zero through the balance and hop phases).
    command_start_step: int | None = None
    # Success-gated speed curriculum: commands are sampled within the current
    # cap, which advances through these levels once the EMA of hop-averaged
    # tracking error (over moving envs) stays under the threshold.
    speed_curriculum_caps: tuple[float, ...] | None = None
    speed_curriculum_error_threshold_m_s: float = 0.12
    # When set, the gate becomes relative: error EMA must drop below this
    # fraction of the mean commanded speed (moving envs), so hopping in place
    # (error == command) can never clear it.
    speed_curriculum_relative_threshold: float | None = None
    # v17: absolute floor for the relative promotion threshold, so it can
    # never shrink below the hop-cycle noise floor at low command caps.
    speed_curriculum_error_floor_m_s: float = 0.08
    speed_curriculum_ema_tau_s: float = 30.0
    speed_curriculum_min_level_time_s: float = 60.0

    def build(self, env: ManagerBasedRlEnv) -> PlanarVelocityCommand:
        return PlanarVelocityCommand(self, env)


def build_planar_velocity_commands(
    play: bool = False,
    command_start_step: int | None = None,
    speed_curriculum_caps: tuple[float, ...] | None = None,
    speed_curriculum_relative_threshold: float | None = None,
    stationary_probability: float = 0.35,
    max_speed_m_s: float = PLANAR_VELOCITY_SCALE_M_S,
    apply_only_when_grounded: bool = False,
) -> dict[str, CommandTermCfg]:
    return {
        PLANAR_VELOCITY_COMMAND_NAME: PlanarVelocityCommandCfg(
            play=play,
            max_speed_m_s=max_speed_m_s,
            resampling_time_range=(1.0e9, 1.0e9) if play else (3.0, 6.0),
            command_start_step=command_start_step,
            speed_curriculum_caps=speed_curriculum_caps,
            speed_curriculum_relative_threshold=speed_curriculum_relative_threshold,
            stationary_probability=stationary_probability,
            apply_only_when_grounded=apply_only_when_grounded,
        )
    }
