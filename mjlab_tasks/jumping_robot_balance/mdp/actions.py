"""Action configuration for the jumping robot balance task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import (
    JointEffortActionCfg,
    JointPositionAction,
    JointPositionActionCfg,
)
from mjlab.managers.action_manager import ActionTermCfg

from mjlab_tasks.jumping_robot_balance.mdp.height_curriculum import (
    HEIGHT_RANGE_SCHEDULE,
    scheduled_height_half_width,
)
from mjlab_tasks.jumping_robot_balance.robot_cfg import (
    FLYWHEEL_X_JOINT,
    FLYWHEEL_Y_JOINT,
    LINEAR_JOINT,
    LINEAR_RANGE_HALF_WIDTH_M,
    LINEAR_RANGE_MAX_M,
    LINEAR_RANGE_MIN_M,
    MAX_FLYWHEEL_TORQUE_NM,
    ROBOT_ENTITY_NAME,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_NOMINAL_BALANCE_LINEAR_SCALE_M = 0.0
_HEIGHT_TARGET_MAX_SPEED_M_S = 0.25


class CurriculumJointPositionAction(JointPositionAction):
    cfg: CurriculumJointPositionActionCfg

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        half_width = scheduled_height_half_width(
            self._env.common_step_counter,
            self.cfg.scale_schedule,
        )
        self._scale = half_width
        desired = self._raw_actions * half_width + self._offset
        if self.cfg.clip is not None:
            desired = torch.clamp(
                desired,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )
        max_delta = self.cfg.max_target_speed_m_s * self._env.step_dt
        self._processed_actions += torch.clamp(
            desired - self._processed_actions,
            min=-max_delta,
            max=max_delta,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        self._processed_actions[env_ids] = self._entity.data.joint_pos[env_ids][
            :, self._target_ids
        ]


@dataclass(kw_only=True)
class CurriculumJointPositionActionCfg(JointPositionActionCfg):
    scale_schedule: tuple[tuple[int, float], ...] = HEIGHT_RANGE_SCHEDULE
    max_target_speed_m_s: float = _HEIGHT_TARGET_MAX_SPEED_M_S

    def build(self, env: ManagerBasedRlEnv) -> CurriculumJointPositionAction:
        return CurriculumJointPositionAction(self, env)


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
        "linear_position": JointPositionActionCfg(
            entity_name=ROBOT_ENTITY_NAME,
            actuator_names=(LINEAR_JOINT,),
            # Preserve the third policy output while holding the leg fixed during
            # the nominal balance curriculum stage.
            scale=_NOMINAL_BALANCE_LINEAR_SCALE_M,
            use_default_offset=True,
        ),
    }


def build_height_action_terms(play: bool = False) -> dict[str, ActionTermCfg]:
    terms = build_action_terms()
    schedule = (
        ((0, LINEAR_RANGE_HALF_WIDTH_M),)
        if play
        else HEIGHT_RANGE_SCHEDULE
    )
    terms["linear_position"] = CurriculumJointPositionActionCfg(
        entity_name=ROBOT_ENTITY_NAME,
        actuator_names=(LINEAR_JOINT,),
        scale=schedule[0][1],
        scale_schedule=schedule,
        max_target_speed_m_s=_HEIGHT_TARGET_MAX_SPEED_M_S,
        clip={LINEAR_JOINT: (LINEAR_RANGE_MIN_M, LINEAR_RANGE_MAX_M)},
        use_default_offset=True,
    )
    return terms
