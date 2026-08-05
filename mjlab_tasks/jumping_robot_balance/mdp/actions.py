"""Action configuration for the jumping robot balance task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from mjlab_tasks.jumping_robot_balance.mdp.commands import HEIGHT_COMMAND_NAME
from mjlab_tasks.jumping_robot_balance.mdp.height_curriculum import (
    scheduled_height_half_width,
)
from mjlab_tasks.jumping_robot_balance.robot_cfg import (
    FLYWHEEL_X_JOINT,
    FLYWHEEL_Y_JOINT,
    LINEAR_BALANCE_FEEDFORWARD_LIMIT_N,
    LINEAR_JOINT,
    LINEAR_MAX_FORCE_N,
    LINEAR_POSITION_KD_N_S_M,
    LINEAR_POSITION_KP_N_M,
    LINEAR_RANGE_HALF_WIDTH_M,
    LINEAR_RANGE_MAX_M,
    LINEAR_RANGE_MIN_M,
    MAX_FLYWHEEL_TORQUE_NM,
    ROBOT_ENTITY_NAME,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_NOMINAL_BALANCE_LINEAR_SCALE_M = 0.0
_HEIGHT_POSITION_RESIDUAL_SCALE_M = 0.010
_HEIGHT_TARGET_MAX_SPEED_M_S = 1.0
_JUMP_POSITION_RESIDUAL_SCALE_M = LINEAR_RANGE_HALF_WIDTH_M
_JUMP_TARGET_MAX_SPEED_M_S = 3.0
_JUMP_VELOCITY_TARGET_SCALE_M_S = 3.0


class LinearMitAction(ActionTerm):
    cfg: LinearMitActionCfg

    def __init__(self, cfg: LinearMitActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        target_ids, target_names = self._entity.find_joints_by_actuator_names(
            (LINEAR_JOINT,)
        )
        if len(target_ids) != 1:
            raise ValueError(
                f"Expected one linear actuator, found {target_names}."
            )
        self._target_ids = torch.tensor(
            target_ids,
            dtype=torch.long,
            device=self.device,
        )
        self._action_dim = (
            1 + int(cfg.expose_velocity_target) + int(cfg.expose_feedforward)
        )
        self._raw_actions = torch.zeros(
            self.num_envs,
            self._action_dim,
            device=self.device,
        )
        self._position_target = self._entity.data.default_joint_pos[
            :, self._target_ids
        ].clone()
        self._velocity_target = torch.zeros_like(self._position_target)
        self._feedforward_force = torch.zeros_like(self._position_target)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def position_target(self) -> torch.Tensor:
        return self._position_target

    @property
    def velocity_target(self) -> torch.Tensor:
        return self._velocity_target

    @property
    def feedforward_force(self) -> torch.Tensor:
        return self._feedforward_force

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        half_width = scheduled_height_half_width(
            self._env.common_step_counter,
            self.cfg.position_scale_schedule,
        )
        if self.cfg.position_command_name is None:
            position_center = self._entity.data.default_joint_pos[
                :, self._target_ids
            ]
        else:
            position_center = self._env.command_manager.get_command(
                self.cfg.position_command_name
            )
        desired_position = torch.clamp(
            position_center + actions[:, :1] * half_width,
            min=LINEAR_RANGE_MIN_M,
            max=LINEAR_RANGE_MAX_M,
        )
        max_delta = self.cfg.max_target_speed_m_s * self._env.step_dt
        self._position_target += torch.clamp(
            desired_position - self._position_target,
            min=-max_delta,
            max=max_delta,
        )
        action_index = 1
        if self.cfg.expose_velocity_target:
            normalized_velocity = torch.clamp(
                actions[:, action_index : action_index + 1],
                min=-1.0,
                max=1.0,
            )
            normalized_velocity = (
                torch.sign(normalized_velocity)
                * torch.abs(normalized_velocity) ** self.cfg.velocity_target_exponent
            )
            self._velocity_target[:] = (
                normalized_velocity * self.cfg.velocity_target_scale_m_s
            )
            action_index += 1
        else:
            self._velocity_target.zero_()
        if self.cfg.expose_feedforward:
            normalized_force = torch.clamp(
                actions[:, action_index : action_index + 1],
                min=-1.0,
                max=1.0,
            )
            normalized_force = (
                torch.sign(normalized_force)
                * torch.abs(normalized_force) ** self.cfg.feedforward_exponent
            )
            self._feedforward_force[:] = (
                normalized_force * self.cfg.feedforward_force_scale_n
            )
        else:
            self._feedforward_force.zero_()

    def apply_actions(self) -> None:
        position = self._entity.data.joint_pos[:, self._target_ids]
        velocity = self._entity.data.joint_vel[:, self._target_ids]
        effort = (
            self.cfg.kp_n_m * (self._position_target - position)
            + self.cfg.kd_n_s_m * (self._velocity_target - velocity)
            + self._feedforward_force
        )
        effort = torch.clamp(
            effort,
            min=-self.cfg.max_force_n,
            max=self.cfg.max_force_n,
        )
        self._entity.set_joint_effort_target(
            effort,
            joint_ids=self._target_ids,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._velocity_target[env_ids] = 0.0
        self._feedforward_force[env_ids] = 0.0
        self._position_target[env_ids] = self._entity.data.joint_pos[env_ids][
            :, self._target_ids
        ]


@dataclass(kw_only=True)
class LinearMitActionCfg(ActionTermCfg):
    position_scale_schedule: tuple[tuple[int, float], ...] = (
        (0, _NOMINAL_BALANCE_LINEAR_SCALE_M),
    )
    max_target_speed_m_s: float = _HEIGHT_TARGET_MAX_SPEED_M_S
    kp_n_m: float = LINEAR_POSITION_KP_N_M
    kd_n_s_m: float = LINEAR_POSITION_KD_N_S_M
    max_force_n: float = LINEAR_MAX_FORCE_N
    expose_velocity_target: bool = False
    velocity_target_scale_m_s: float = 0.0
    velocity_target_exponent: float = 1.0
    expose_feedforward: bool = False
    feedforward_force_scale_n: float = LINEAR_BALANCE_FEEDFORWARD_LIMIT_N
    feedforward_exponent: float = 1.0
    position_command_name: str | None = None

    def build(self, env: ManagerBasedRlEnv) -> LinearMitAction:
        return LinearMitAction(self, env)


def build_action_terms() -> dict[str, ActionTermCfg]:
    return {
        "flywheel_x_torque": JointEffortActionCfg(
            entity_name=ROBOT_ENTITY_NAME,
            actuator_names=(FLYWHEEL_X_JOINT,),
            scale=MAX_FLYWHEEL_TORQUE_NM,
        ),
        "flywheel_y_torque": JointEffortActionCfg(
            entity_name=ROBOT_ENTITY_NAME,
            actuator_names=(FLYWHEEL_Y_JOINT,),
            scale=MAX_FLYWHEEL_TORQUE_NM,
        ),
        "linear_impedance": LinearMitActionCfg(
            entity_name=ROBOT_ENTITY_NAME,
            # Preserve the third policy output while holding the leg fixed during
            # the nominal balance curriculum stage.
        ),
    }


def build_height_action_terms(play: bool = False) -> dict[str, ActionTermCfg]:
    del play
    terms = build_action_terms()
    terms["linear_impedance"] = LinearMitActionCfg(
        entity_name=ROBOT_ENTITY_NAME,
        position_scale_schedule=((0, _HEIGHT_POSITION_RESIDUAL_SCALE_M),),
        max_target_speed_m_s=_HEIGHT_TARGET_MAX_SPEED_M_S,
        expose_feedforward=True,
        feedforward_force_scale_n=LINEAR_BALANCE_FEEDFORWARD_LIMIT_N,
        position_command_name=HEIGHT_COMMAND_NAME,
    )
    return terms


def build_jump_stage_one_action_terms() -> dict[str, ActionTermCfg]:
    terms = build_action_terms()
    terms["linear_impedance"] = LinearMitActionCfg(
        entity_name=ROBOT_ENTITY_NAME,
        position_scale_schedule=((0, _JUMP_POSITION_RESIDUAL_SCALE_M),),
        max_target_speed_m_s=_JUMP_TARGET_MAX_SPEED_M_S,
        expose_velocity_target=True,
        velocity_target_scale_m_s=_JUMP_VELOCITY_TARGET_SCALE_M_S,
        velocity_target_exponent=3.0,
        expose_feedforward=True,
        feedforward_force_scale_n=LINEAR_MAX_FORCE_N,
        feedforward_exponent=3.0,
        position_command_name=HEIGHT_COMMAND_NAME,
    )
    return terms
