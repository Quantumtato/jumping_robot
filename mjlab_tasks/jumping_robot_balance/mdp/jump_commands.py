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

from mjlab_tasks.jumping_robot_balance.mdp.contact import (
    foot_ground_contact,
    foot_height_w,
)
from mjlab_tasks.jumping_robot_balance.mdp.commands import (
    PLANAR_VELOCITY_COMMAND_NAME,
)
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
NAVIGATION_TARGET_HEIGHTS_M: tuple[float, ...] = (
    0.04,
    0.06,
    0.08,
    0.10,
    0.125,
    0.15,
)
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
        # 1.0 on the single step the foot leaves the ground during a jump;
        # the v16 takeoff-impulse reward reads it.
        self.takeoff_event = torch.zeros(self.num_envs, device=self.device)
        # 1.0 on the step a POLICY jump request fires (v21). Rewarded
        # directly: forced cadence preempts any income advantage from
        # self-triggering, so without a bonus the marginal self-trigger is
        # pure fall risk and the policy prunes the channel (v19/v20 collapse).
        self.model_trigger_event = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        # Root xy at takeoff and the takeoff-to-touchdown displacement,
        # exposed on the touchdown step for the v18 per-hop displacement
        # reward (prices the whole hop, not just the launch).
        self._takeoff_pos_xy = torch.zeros(
            (self.num_envs, 2),
            device=self.device,
        )
        self.hop_displacement_xy = torch.zeros(
            (self.num_envs, 2),
            device=self.device,
        )
        # Wall-clock interval between consecutive touchdowns, exposed on the
        # touchdown step (v24). The displacement reward caps its payout at
        # command_speed * hop_period so neither bigger nor more frequent hops
        # can out-earn accurate tracking (the v23/v23b std inflation was PPO
        # funding trigger-channel noise from uncapped displacement income).
        self.hop_period_s = torch.zeros(self.num_envs, device=self.device)
        self._time_since_touchdown_s = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        # v31 progress gates. Hop start waits for the balance phase to
        # actually converge (fraction of envs tilted past the gate angle,
        # EMA, below threshold) instead of firing on the clock -- v30
        # started hops on a policy that was still falling every ~2.5 s.
        # The scaffold fade waits for the trigger handover to complete,
        # a healthy self-trigger rate, and an earned speed promotion --
        # v30 faded on the clock while the ladder was stuck at rung one
        # and bankrupted the gait.
        self.hops_enabled_since_step: int | None = None
        self._balance_upset_ema = 1.0
        self._fade_start_step: int | None = None
        # v32 promotion currency: EMA of the per-hop tent fraction (see the
        # touchdown handler). Starts at 0 (pessimistic); the speed
        # curriculum reads it through hop_fraction_ema and resets it on
        # promotion via reset_hop_fraction_ema().
        self._hop_fraction_ema = 0.0
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
        self.landing_upright_event = torch.zeros(
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
        self._previous_root_velocity_w = (
            self._robot.data.root_link_lin_vel_w.clone()
        )
        # Base |dv|/dt during the post-touchdown window; zero elsewhere.
        # Rewards penalize its excess over a threshold: actual landing
        # violence, immune to the touch-down-early gaming that a
        # speed-at-first-contact penalty invites.
        self.landing_impact_accel = torch.zeros(
            self.num_envs,
            device=self.device,
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
        self._model_jump_request = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.time_since_jump_end = torch.full(
            (self.num_envs,),
            cfg.trigger_cooldown_s,
            device=self.device,
        )
        self._curriculum_level = 0
        self._level_time_s = 0.0
        self._steps_since_curriculum_check = 0
        self._trigger_count_this_step = 0
        self._trigger_rate_ema = torch.zeros((), device=self.device)
        self._recovery_rate_ema = torch.zeros((), device=self.device)
        self._model_trigger_count_this_step = 0
        self._auto_trigger_count_this_step = 0
        self._model_trigger_rate_ema = torch.zeros((), device=self.device)
        self._auto_trigger_rate_ema = torch.zeros((), device=self.device)
        self.metrics["balance_upset_ema"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["hop_fraction_ema"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["scaffold_fade_progress"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["model_trigger_rate"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["auto_trigger_rate"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        if cfg.success_gated_target_heights_m is not None:
            self.metrics["curriculum_target_height"] = torch.zeros(
                self.num_envs,
                device=self.device,
            )
            self.metrics["curriculum_recovery_ratio"] = torch.zeros(
                self.num_envs,
                device=self.device,
            )
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

    def request_jump(self, request_mask: torch.Tensor) -> None:
        """Record a policy-issued jump request; honored on the next update."""
        self._model_jump_request |= request_mask

    def _update_phase_gates(self) -> None:
        """Latch the hop-start and scaffold-fade progress gates (v31)."""
        if self.cfg.play or self.cfg.phase_schedule_steps is None:
            return
        step = self._env.common_step_counter
        hop_start_step, _ = self.cfg.phase_schedule_steps
        if self.hops_enabled_since_step is None:
            if self.cfg.balance_gate_upset_threshold is None:
                if step >= hop_start_step:
                    self.hops_enabled_since_step = step
            else:
                upset = (
                    (
                        torch.linalg.vector_norm(
                            self._robot.data.projected_gravity_b[:, :2],
                            dim=1,
                        )
                        > math.sin(
                            math.radians(self.cfg.balance_gate_tilt_deg)
                        )
                    )
                    .float()
                    .mean()
                    .item()
                )
                # tau ~2000 policy steps (~60 iterations); starting from the
                # pessimistic 1.0, the latch needs a few hundred iterations
                # of consistently upright envs before it can fire.
                blend = 0.0005
                self._balance_upset_ema += blend * (
                    upset - self._balance_upset_ema
                )
                ready = (
                    self._balance_upset_ema
                    < self.cfg.balance_gate_upset_threshold
                )
                overdue = step >= (
                    hop_start_step + self.cfg.balance_gate_max_delay_steps
                )
                if step >= hop_start_step and (ready or overdue):
                    self.hops_enabled_since_step = step
                    print(
                        f"[INFO] Hop phase enabled at step {step} (upset "
                        f"EMA {self._balance_upset_ema:.4f}"
                        f"{', fallback latch' if overdue and not ready else ''})."
                    )
        if (
            self.cfg.scaffold_fade_duration_steps is not None
            and self._fade_start_step is None
            and self.hops_enabled_since_step is not None
            and self._trigger_anneal_progress() >= 1.0
            and float(self._model_trigger_rate_ema)
            >= self.cfg.scaffold_fade_min_trigger_rate
        ):
            planar = self._env.command_manager.get_term(
                PLANAR_VELOCITY_COMMAND_NAME
            )
            if getattr(planar, "_speed_level", 0) >= 1:
                self._fade_start_step = step
                print(f"[INFO] Scaffold fade started at step {step}.")

    @property
    def hop_fraction_ema(self) -> float:
        """EMA of the achieved fraction of commanded hop distance (v32)."""
        return self._hop_fraction_ema

    def reset_hop_fraction_ema(self) -> None:
        """Pessimistic reset after a speed promotion (new command scale)."""
        self._hop_fraction_ema = 0.0

    @property
    def scaffold_fade_progress(self) -> float:
        """0 before the fade latch fires, 1 once scaffolds are fully gone."""
        if (
            self.cfg.scaffold_fade_duration_steps is None
            or self._fade_start_step is None
        ):
            return 0.0
        return min(
            1.0,
            (self._env.common_step_counter - self._fade_start_step)
            / max(1, self.cfg.scaffold_fade_duration_steps),
        )

    def _model_trigger_enabled(self) -> bool:
        if self.cfg.phase_schedule_steps is None:
            return self.cfg.model_triggered
        if self.cfg.play:
            return True
        return self.hops_enabled_since_step is not None

    def _trigger_anneal_progress(self) -> float:
        """0 before the anneal starts, 1 when forced triggers are fully off."""
        anneal = self.cfg.auto_trigger_anneal_steps
        if anneal is None:
            return 0.0
        start, end = anneal
        step = self._env.common_step_counter
        if step <= start:
            return 0.0
        return min(1.0, (step - start) / max(1, end - start))

    def _auto_trigger_enabled(self) -> bool:
        # v14 handover: forced triggers anneal away entirely; the policy's
        # own jump-request channel takes over timing.
        if (
            self.cfg.auto_trigger_anneal_steps is not None
            and not self.cfg.play
            and self._trigger_anneal_progress() >= 1.0
        ):
            return False
        if self.cfg.phase_schedule_steps is None:
            return self.cfg.auto_trigger
        if self.cfg.play:
            return False
        _, model_only_step = self.cfg.phase_schedule_steps
        if self.hops_enabled_since_step is None:
            return False
        return self._env.common_step_counter < model_only_step

    @property
    def current_target_height(self) -> float:
        heights = self.cfg.success_gated_target_heights_m
        if heights is None:
            raise ValueError(
                "Success-gated jump curriculum heights are not configured."
            )
        return heights[self._curriculum_level]

    def _update_curriculum(self, dt: float) -> None:
        """Advance the jump-height target when landing recovery is reliable."""
        heights = self.cfg.success_gated_target_heights_m
        if heights is None or self.cfg.play:
            return
        blend = dt / (dt + self.cfg.curriculum_ema_tau_s)
        self._trigger_rate_ema += blend * (
            float(self._trigger_count_this_step) - self._trigger_rate_ema
        )
        self._model_trigger_rate_ema += blend * (
            float(self._model_trigger_count_this_step)
            - self._model_trigger_rate_ema
        )
        self._auto_trigger_rate_ema += blend * (
            float(self._auto_trigger_count_this_step)
            - self._auto_trigger_rate_ema
        )
        # Under continuous locomotion the robot never stands still, so the
        # stand-still recovery gate can never fire (v9 lesson); gate on
        # upright foot touchdowns instead when configured.
        success_event = (
            self.landing_upright_event
            if self.cfg.curriculum_gate_on_landing
            else self.landing_recovery_success_event
        )
        self._recovery_rate_ema += blend * (
            success_event.sum() - self._recovery_rate_ema
        )
        self._level_time_s += dt
        self._steps_since_curriculum_check += 1
        if self._steps_since_curriculum_check < 50:
            return
        self._steps_since_curriculum_check = 0
        if (
            self._curriculum_level >= len(heights) - 1
            or self._level_time_s < self.cfg.curriculum_min_level_time_s
        ):
            return
        ratio = (
            self._recovery_rate_ema
            / self._trigger_rate_ema.clamp_min(1.0e-6)
        ).item()
        if ratio >= self.cfg.curriculum_success_threshold:
            self._curriculum_level += 1
            self._level_time_s = 0.0
            self._trigger_rate_ema.zero_()
            self._recovery_rate_ema.zero_()
            print(
                "[INFO] Jump curriculum advanced to level "
                f"{self._curriculum_level} (target height "
                f"{heights[self._curriculum_level]:.3f} m)."
            )

    def _update_metrics(self) -> None:
        heights = self.cfg.success_gated_target_heights_m
        if heights is not None:
            self.metrics["curriculum_target_height"].fill_(
                heights[self._curriculum_level]
            )
            self.metrics["curriculum_recovery_ratio"][:] = (
                self._recovery_rate_ema
                / self._trigger_rate_ema.clamp_min(1.0e-6)
            )
        self.metrics["apex_height"][:] = self.apex_height
        self.metrics["balance_upset_ema"].fill_(self._balance_upset_ema)
        self.metrics["hop_fraction_ema"].fill_(self._hop_fraction_ema)
        self.metrics["scaffold_fade_progress"].fill_(
            self.scaffold_fade_progress
        )
        self.metrics["model_trigger_rate"].fill_(
            float(self._model_trigger_rate_ema)
        )
        self.metrics["auto_trigger_rate"].fill_(
            float(self._auto_trigger_rate_ema)
        )
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
        self.takeoff_event[env_ids] = 0.0
        self.model_trigger_event[env_ids] = 0.0
        self._takeoff_pos_xy[env_ids] = 0.0
        self.hop_displacement_xy[env_ids] = 0.0
        self.hop_period_s[env_ids] = 0.0
        self._time_since_touchdown_s[env_ids] = 0.0
        self.airborne_time[env_ids] = 0.0
        self.landing_recovery_time[env_ids] = 0.0
        self.landing_recovery_complete[env_ids] = False
        self.landing_recovery_success_event[env_ids] = 0.0
        self.landing_upright_event[env_ids] = 0.0
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
        self._previous_root_velocity_w[env_ids] = (
            self._robot.data.root_link_lin_vel_w[env_ids]
        )
        self.landing_impact_accel[env_ids] = 0.0
        self._model_jump_request[env_ids] = False
        self.time_since_jump_end[env_ids] = self.cfg.trigger_cooldown_s
        if self.cfg.play:
            self.time_left[env_ids] = math.inf

    def _update_command(self, dt: float) -> None:
        self._update_phase_gates()
        self.apex_progress_delta.zero_()
        self.landing_event.zero_()
        self.hop_displacement_xy.zero_()
        self.hop_period_s.zero_()
        self._time_since_touchdown_s += dt
        self.model_trigger_event.zero_()
        self.landing_recovery_success_event.zero_()
        self.landing_upright_event.zero_()
        self._trigger_count_this_step = 0
        while True:
            try:
                env_idx = self._pending.get_nowait()
            except Empty:
                break
            self._trigger(
                torch.tensor([env_idx], dtype=torch.long, device=self.device)
            )

        self.time_since_jump_end += dt
        contact = foot_ground_contact(self._env)[:, 0] > 0.5
        inactive = self._command[:, 0] < 0.5
        current_velocity_w = self._robot.data.root_link_lin_vel_w
        if dt > 0.0:
            base_accel = (
                torch.linalg.vector_norm(
                    current_velocity_w - self._previous_root_velocity_w,
                    dim=1,
                )
                / dt
            )
        else:
            base_accel = torch.zeros(self.num_envs, device=self.device)
        if self._model_trigger_enabled():
            eligible = (
                inactive
                & contact
                & (self.time_since_jump_end >= self.cfg.trigger_cooldown_s)
            )
            model_trigger = self._model_jump_request & eligible
            self._model_trigger_count_this_step = int(model_trigger.sum())
            self.model_trigger_event[model_trigger] = 1.0
            self._trigger(model_trigger.nonzero().flatten())
        else:
            self._model_trigger_count_this_step = 0
        self._model_jump_request[:] = False

        can_auto_trigger = ~self.has_triggered | self.cfg.repeat_auto_trigger
        requested_hop = torch.ones(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        if self.cfg.auto_trigger_min_planar_speed_m_s is not None:
            planar_velocity = self._env.command_manager.get_command(
                PLANAR_VELOCITY_COMMAND_NAME
            )
            requested_hop = torch.linalg.vector_norm(
                planar_velocity,
                dim=1,
            ) >= self.cfg.auto_trigger_min_planar_speed_m_s
        trigger_due = self.time_left <= 0.0
        if self.cfg.play and self.cfg.auto_trigger_in_play:
            trigger_due |= ~self.has_triggered
        auto_trigger = (
            inactive
            & can_auto_trigger
            & requested_hop
            & trigger_due
        )
        self._auto_trigger_count_this_step = 0
        if (
            self._auto_trigger_enabled()
            and (not self.cfg.play or self.cfg.auto_trigger_in_play)
            and torch.any(auto_trigger)
        ):
            self._auto_trigger_count_this_step = int(auto_trigger.sum())
            self._trigger(auto_trigger.nonzero().flatten())

        self.airborne_time[contact] = 0.0
        self.airborne_time[~contact] = torch.clamp(
            self.airborne_time[~contact] + dt,
            max=self.cfg.airborne_time_observation_max_s,
        )
        horizontal_gravity = torch.linalg.vector_norm(
            self._robot.data.projected_gravity_b[:, :2],
            dim=1,
        )
        tracking_landing = self.has_triggered & ~self.has_landed
        self.takeoff_event.zero_()
        takeoff = tracking_landing & ~self.was_airborne & ~contact
        self.takeoff_event[takeoff] = 1.0
        self._takeoff_pos_xy[takeoff] = (
            self._robot.data.root_link_pos_w[takeoff, :2]
        )
        self.was_airborne |= tracking_landing & ~contact
        touchdown = tracking_landing & self.was_airborne & contact
        self.landing_event[touchdown] = 1.0
        self.hop_displacement_xy[touchdown] = (
            self._robot.data.root_link_pos_w[touchdown, :2]
            - self._takeoff_pos_xy[touchdown]
        )
        self.hop_period_s[touchdown] = self._time_since_touchdown_s[touchdown]
        self._time_since_touchdown_s[touchdown] = 0.0
        # v32: EMA of the tent fraction (achieved fraction of the commanded
        # distance, same quantity the displacement reward pays) over moving
        # hops. The speed curriculum promotes on this instead of the
        # hop-averaged vector velocity error: v31 showed the 0.6 relative
        # velocity gate has never been passed by any lineage (best ratio
        # ~1.03) because the robot is stationary between ballistic hops,
        # while per-hop displacement accuracy is directly achievable.
        if touchdown.any():
            planar_cmd = self._env.command_manager.get_command(
                PLANAR_VELOCITY_COMMAND_NAME
            )[touchdown]
            cmd_speed = torch.linalg.vector_norm(planar_cmd, dim=1)
            moving = cmd_speed >= 0.02
            if moving.any():
                direction = planar_cmd[moving] / cmd_speed[moving].unsqueeze(1)
                along = (
                    self.hop_displacement_xy[touchdown][moving] * direction
                ).sum(dim=1)
                commanded = (
                    cmd_speed[moving] * self.hop_period_s[touchdown][moving]
                )
                fraction = (
                    (commanded - (along - commanded).abs())
                    / commanded.clamp_min(1.0e-3)
                ).clamp(-2.0, 1.0)
                self._hop_fraction_ema += 0.001 * float(
                    fraction.mean() - self._hop_fraction_ema
                )
        # Foot touchdown while roughly upright counts as "landed without
        # falling" (a crashed robot lands on its body, not its foot).
        upright_touchdown = touchdown & (
            horizontal_gravity
            <= math.sin(math.radians(self.cfg.curriculum_landing_max_tilt_deg))
        )
        self.landing_upright_event[upright_touchdown] = 1.0
        self.landing_impact_speed[touchdown] = torch.clamp(
            -self._previous_root_vertical_velocity[touchdown],
            min=0.0,
        )
        self.has_landed[touchdown] = True
        # Return control to the standing-height objective as soon as flight ends.
        self._finish_jump(touchdown.nonzero().flatten())

        recovering = self.has_landed & ~self.landing_recovery_complete
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
        # Apex is tracked on the FOOT, not the base: base-tracked apex with
        # a trigger-time baseline paid for crouching at trigger and for
        # extending the leg mid-air (raising the base relative to the CoM
        # without any extra ballistic height). Foot apex rewards tucking
        # instead, which is at least a real motion of the lowest link.
        height_gain = torch.clamp(
            foot_height_w(self._env) - self.baseline_height,
            min=0.0,
        )
        new_apex = torch.maximum(self.apex_height, height_gain)
        self.apex_progress_delta[active] = (
            new_apex[active] - self.apex_height[active]
        )
        self.apex_height[active] = new_apex[active]

        expired = active & (self.active_time >= self.cfg.command_duration_s)
        self._finish_jump(expired.nonzero().flatten())
        landing_window = (
            contact
            & (self._command[:, 0] < 0.5)
            & (self.time_since_jump_end < self.cfg.landing_accel_window_s)
        )
        self.landing_impact_accel = base_accel * landing_window.float()
        self._previous_root_vertical_velocity[:] = (
            self._robot.data.root_link_lin_vel_w[:, 2]
        )
        self._previous_root_velocity_w[:] = current_velocity_w

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
        self._update_curriculum(dt)
        self._update_metrics()

    def _trigger(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        inactive = self._command[env_ids, 0] < 0.5
        env_ids = env_ids[inactive]
        if len(env_ids) == 0:
            return
        self._trigger_count_this_step += len(env_ids)
        self._command[env_ids, 0] = 1.0
        self.active_time[env_ids] = 0.0
        self.has_triggered[env_ids] = True
        self.was_airborne[env_ids] = False
        self.has_landed[env_ids] = False
        self.landing_event[env_ids] = 0.0
        self.takeoff_event[env_ids] = 0.0
        # model_trigger_event is deliberately NOT cleared here: _trigger runs
        # in the same call chain that just set it for policy-issued jumps, so
        # clearing it would erase the bonus event before rewards read it
        # (this exact bug silently zeroed the v21 bonus and reproduced the
        # trigger collapse). It is cleared per-step in _update_command.
        self._takeoff_pos_xy[env_ids] = 0.0
        self.hop_displacement_xy[env_ids] = 0.0
        self.airborne_time[env_ids] = 0.0
        self.landing_recovery_time[env_ids] = 0.0
        self.landing_recovery_complete[env_ids] = False
        self.landing_recovery_success_event[env_ids] = 0.0
        self.landing_upright_event[env_ids] = 0.0
        self.landing_impact_speed[env_ids] = 0.0
        self._previous_root_vertical_velocity[env_ids] = (
            self._robot.data.root_link_lin_vel_w[env_ids, 2]
        )
        # Foot is grounded at trigger, so this baseline is ~ground level.
        self.baseline_height[env_ids] = foot_height_w(self._env)[env_ids]
        self.apex_height[env_ids] = 0.0
        self.command_counter[env_ids] += 1

    def _sample_retrigger_interval(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Time until the next forced jump; shrinks with commanded speed.

        Faster commands need more push-offs per second (each ground contact
        is the only chance to change planar momentum), so the retrigger
        interval interpolates from ``resampling_time_range`` at zero command
        to ``speed_scaled_resampling_range`` at the current speed cap.
        """
        low0, high0 = self.cfg.resampling_time_range
        u = torch.rand(len(env_ids), device=self.device)
        if self.cfg.speed_scaled_resampling_range is None:
            return low0 + u * (high0 - low0)
        low1, high1 = self.cfg.speed_scaled_resampling_range
        planar = self._env.command_manager.get_term(
            PLANAR_VELOCITY_COMMAND_NAME
        )
        command_speed = torch.linalg.vector_norm(
            planar.command[env_ids],
            dim=1,
        )
        cap = max(float(planar.current_max_speed()), 1.0e-6)
        s = torch.clamp(command_speed / cap, min=0.0, max=1.0)
        low = low0 + s * (low1 - low0)
        high = high0 + s * (high1 - high0)
        interval = low + u * (high - low)
        # v14 handover: forced-trigger intervals stretch up to 4x across the
        # anneal window (then _auto_trigger_enabled cuts them entirely),
        # giving the policy a widening gap it must fill with its own
        # jump requests.
        interval = interval * (1.0 + 3.0 * self._trigger_anneal_progress())
        return interval

    def _finish_jump(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self._command[env_ids, 0] = 0.0
        self.active_time[env_ids] = 0.0
        self.time_since_jump_end[env_ids] = 0.0
        if self.cfg.repeat_auto_trigger and (
            not self.cfg.play or self.cfg.auto_trigger_in_play
        ):
            self.time_left[env_ids] = self._sample_retrigger_interval(env_ids)
        else:
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
    repeat_auto_trigger: bool = False
    auto_trigger_in_play: bool = False
    auto_trigger_min_planar_speed_m_s: float | None = None
    # When True, jumps are only triggered by the policy's jump_request action
    # (subject to grounding, inactivity, and the post-landing cooldown).
    model_triggered: bool = False
    trigger_cooldown_s: float = 0.25
    # Full-pipeline phase schedule (hop_start_step, model_only_step) in
    # common_step_counter units: before hop_start no jumps happen; between the
    # two, the env force-triggers jumps and model requests are also honored;
    # after model_only_step the policy alone decides when to jump. Overrides
    # auto_trigger/model_triggered when set. Play mode is model-only.
    phase_schedule_steps: tuple[int, int] | None = None
    command_duration_s: float = 1.5
    resampling_time_range: tuple[float, float] = (2.0, 4.0)
    # When set, the forced-jump retrigger interval interpolates from
    # resampling_time_range (zero command) to this range (command at the
    # current speed-curriculum cap). Hardware must mirror this mapping.
    speed_scaled_resampling_range: tuple[float, float] | None = None
    # v14 trigger handover: (start_step, end_step) in common_step_counter
    # units. Before start: normal forced cadence. Between: intervals stretch
    # linearly up to 4x. After end: forced triggers off, policy-only.
    auto_trigger_anneal_steps: tuple[int, int] | None = None
    landing_recovery_duration_s: float = 0.5
    landing_recovery_max_tilt_deg: float = 12.0
    landing_recovery_max_ang_vel_rad_s: float = 3.0
    # Post-touchdown window during which base |dv|/dt counts as landing
    # violence (see landing_impact_accel).
    landing_accel_window_s: float = 0.3
    airborne_time_observation_max_s: float = 1.0
    # When set, the jump-height target only advances once the EMA ratio of
    # recovered landings to triggered jumps clears the success threshold.
    success_gated_target_heights_m: tuple[float, ...] | None = None
    curriculum_success_threshold: float = 0.6
    curriculum_ema_tau_s: float = 30.0
    curriculum_min_level_time_s: float = 60.0
    # When True, the curriculum counts upright foot touchdowns ("landed
    # without falling") instead of full stand-still recoveries. Required
    # whenever velocity commands keep the robot moving between jumps.
    curriculum_gate_on_landing: bool = False
    curriculum_landing_max_tilt_deg: float = 30.0
    # v31 hop-start progress gate: when set, forced hops wait (past the
    # scheduled hop_start_step) until the EMA fraction of envs tilted past
    # balance_gate_tilt_deg drops below this threshold, i.e. the balance
    # phase has actually converged. The fallback latch bounds the delay.
    balance_gate_upset_threshold: float | None = None
    balance_gate_tilt_deg: float = 30.0
    balance_gate_max_delay_steps: int = 48_000
    # v31 scaffold-fade progress gate: when set, the fade starts only after
    # the trigger handover completes, the model self-trigger EMA clears
    # scaffold_fade_min_trigger_rate, and the speed ladder has earned its
    # first promotion; it then runs linearly over this many steps.
    scaffold_fade_duration_steps: int | None = None
    scaffold_fade_min_trigger_rate: float = 6.0

    def build(self, env: ManagerBasedRlEnv) -> JumpCommand:
        return JumpCommand(self, env)


def build_jump_commands(
    play: bool = False,
    auto_trigger: bool = True,
    repeat_auto_trigger: bool = False,
    auto_trigger_in_play: bool = False,
    auto_trigger_min_planar_speed_m_s: float | None = None,
    resampling_time_range: tuple[float, float] = (2.0, 4.0),
    success_gated_target_heights_m: tuple[float, ...] | None = None,
    model_triggered: bool = False,
    phase_schedule_steps: tuple[int, int] | None = None,
    curriculum_min_level_time_s: float = 60.0,
    curriculum_gate_on_landing: bool = False,
    speed_scaled_resampling_range: tuple[float, float] | None = None,
    auto_trigger_anneal_steps: tuple[int, int] | None = None,
    balance_gate_upset_threshold: float | None = None,
    scaffold_fade_duration_steps: int | None = None,
) -> dict[str, CommandTermCfg]:
    return {
        JUMP_COMMAND_NAME: JumpCommandCfg(
            play=play,
            auto_trigger=auto_trigger,
            repeat_auto_trigger=repeat_auto_trigger,
            auto_trigger_in_play=auto_trigger_in_play,
            auto_trigger_min_planar_speed_m_s=auto_trigger_min_planar_speed_m_s,
            success_gated_target_heights_m=success_gated_target_heights_m,
            model_triggered=model_triggered,
            phase_schedule_steps=phase_schedule_steps,
            curriculum_min_level_time_s=curriculum_min_level_time_s,
            curriculum_gate_on_landing=curriculum_gate_on_landing,
            speed_scaled_resampling_range=speed_scaled_resampling_range,
            auto_trigger_anneal_steps=auto_trigger_anneal_steps,
            balance_gate_upset_threshold=balance_gate_upset_threshold,
            scaffold_fade_duration_steps=scaffold_fade_duration_steps,
            resampling_time_range=(
                (
                    resampling_time_range
                    if auto_trigger_in_play
                    else (1.0e9, 1.0e9)
                )
                if play
                else resampling_time_range
            ),
        )
    }
