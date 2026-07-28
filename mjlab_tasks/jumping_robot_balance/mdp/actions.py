"""Action configuration for the jumping robot balance task."""

from __future__ import annotations

from mjlab.envs.mdp.actions import JointEffortActionCfg, JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg

from mjlab_tasks.jumping_robot_balance.robot_cfg import (
    FLYWHEEL_X_JOINT,
    FLYWHEEL_Y_JOINT,
    LINEAR_JOINT,
    LINEAR_RANGE_HALF_WIDTH_M,
    MAX_FLYWHEEL_TORQUE_NM,
    ROBOT_ENTITY_NAME,
)


def build_action_terms() -> dict[str, ActionTermCfg]:
    return {
        "flywheel_torque": JointEffortActionCfg(
            entity_name=ROBOT_ENTITY_NAME,
            actuator_names=(FLYWHEEL_X_JOINT, FLYWHEEL_Y_JOINT),
            scale=MAX_FLYWHEEL_TORQUE_NM,
        ),
        "linear_position": JointPositionActionCfg(
            entity_name=ROBOT_ENTITY_NAME,
            actuator_names=(LINEAR_JOINT,),
            scale=LINEAR_RANGE_HALF_WIDTH_M,
            use_default_offset=True,
        ),
    }
