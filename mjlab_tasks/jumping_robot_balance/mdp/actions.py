"""Action configuration for the jumping robot balance task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from mjlab_tasks.jumping_robot_balance.mdp.commands import HEIGHT_COMMAND_NAME
from mjlab_tasks.jumping_robot_balance.mdp.height_curriculum import (
    scheduled_height_half_width,
)
from mjlab_tasks.jumping_robot_balance.mdp.jump_commands import JUMP_COMMAND_NAME
from mjlab_tasks.jumping_robot_balance.robot_cfg import (
    FLYWHEEL_X_JOINT,
    FLYWHEEL_Y_JOINT,
    LINEAR_JOINT,
    LINEAR_MAX_FORCE_N,
    LINEAR_POSITION_KD_N_S_M,
    LINEAR_POSITION_KP_N_M,
    LINEAR_RANGE_HALF_WIDTH_M,
    LINEAR_RANGE_MAX_M,
    LINEAR_RANGE_MIN_M,
    MAX_FLYWHEEL_SPEED_RAD_S,
    ROBOT_ENTITY_NAME,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_NOMINAL_BALANCE_LINEAR_SCALE_M = 0.0
_HEIGHT_POSITION_RESIDUAL_SCALE_M = 0.010
_HEIGHT_TARGET_MAX_SPEED_M_S = 1.0
_JUMP_POSITION_RESIDUAL_SCALE_M = LINEAR_RANGE_HALF_WIDTH_M
_JUMP_TARGET_MAX_SPEED_M_S = 10.0


class LinearPositionAction(ActionTerm):
    cfg: LinearPositionActionCfg

    def __init__(self, cfg: LinearPositionActionCfg, env: ManagerBasedRlEnv):
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
        self._action_dim = 1
        self._raw_actions = torch.zeros(
            self.num_envs,
            self._action_dim,
            device=self.device,
        )
        self._position_target = self._entity.data.default_joint_pos[
            :, self._target_ids
        ].clone()

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
    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        action_gate = 1.0
        if self.cfg.activation_command_name is not None:
            action_gate = torch.clamp(
                self._env.command_manager.get_command(
                    self.cfg.activation_command_name
                ),
                min=0.0,
                max=1.0,
            )
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
            position_center + actions[:, :1] * half_width * action_gate,
            min=LINEAR_RANGE_MIN_M,
            max=LINEAR_RANGE_MAX_M,
        )
        max_delta = self.cfg.max_target_speed_m_s * self._env.step_dt
        self._position_target += torch.clamp(
            desired_position - self._position_target,
            min=-max_delta,
            max=max_delta,
        )
    def apply_actions(self) -> None:
        position = self._entity.data.joint_pos[:, self._target_ids]
        velocity = self._entity.data.joint_vel[:, self._target_ids]
        effort = (
            self.cfg.kp_n_m * (self._position_target - position)
            - self.cfg.kd_n_s_m * velocity
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
        self._position_target[env_ids] = self._entity.data.joint_pos[env_ids][
            :, self._target_ids
        ]


@dataclass(kw_only=True)
class LinearPositionActionCfg(ActionTermCfg):
    position_scale_schedule: tuple[tuple[int, float], ...] = (
        (0, _NOMINAL_BALANCE_LINEAR_SCALE_M),
    )
    max_target_speed_m_s: float = _HEIGHT_TARGET_MAX_SPEED_M_S
    kp_n_m: float = LINEAR_POSITION_KP_N_M
    kd_n_s_m: float = LINEAR_POSITION_KD_N_S_M
    max_force_n: float = LINEAR_MAX_FORCE_N
    position_command_name: str | None = None
    activation_command_name: str | None = None

    def build(self, env: ManagerBasedRlEnv) -> LinearPositionAction:
        return LinearPositionAction(self, env)


def build_action_terms() -> dict[str, ActionTermCfg]:
    return {
        "flywheel_x_velocity": JointVelocityActionCfg(
            entity_name=ROBOT_ENTITY_NAME,
            actuator_names=(FLYWHEEL_X_JOINT,),
            scale=MAX_FLYWHEEL_SPEED_RAD_S,
        ),
        "flywheel_y_velocity": JointVelocityActionCfg(
            entity_name=ROBOT_ENTITY_NAME,
            actuator_names=(FLYWHEEL_Y_JOINT,),
            scale=MAX_FLYWHEEL_SPEED_RAD_S,
        ),
        "linear_position": LinearPositionActionCfg(
            entity_name=ROBOT_ENTITY_NAME,
            # Preserve the third policy output while holding the leg fixed during
            # the nominal balance curriculum stage.
        ),
    }


def build_height_action_terms(play: bool = False) -> dict[str, ActionTermCfg]:
    del play
    terms = build_action_terms()
    terms["linear_position"] = LinearPositionActionCfg(
        entity_name=ROBOT_ENTITY_NAME,
        position_scale_schedule=((0, _HEIGHT_POSITION_RESIDUAL_SCALE_M),),
        max_target_speed_m_s=_HEIGHT_TARGET_MAX_SPEED_M_S,
        position_command_name=HEIGHT_COMMAND_NAME,
    )
    return terms


def build_jump_stage_one_action_terms(
    gate_with_jump_command: bool = False,
) -> dict[str, ActionTermCfg]:
    terms = build_action_terms()
    terms["linear_position"] = LinearPositionActionCfg(
        entity_name=ROBOT_ENTITY_NAME,
        position_scale_schedule=((0, _JUMP_POSITION_RESIDUAL_SCALE_M),),
        max_target_speed_m_s=_JUMP_TARGET_MAX_SPEED_M_S,
        position_command_name=HEIGHT_COMMAND_NAME,
        activation_command_name=(
            JUMP_COMMAND_NAME if gate_with_jump_command else None
        ),
    )
    return terms
