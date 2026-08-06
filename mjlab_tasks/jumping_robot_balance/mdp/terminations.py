"""Termination terms for the jumping robot balance task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from mjlab_tasks.jumping_robot_balance.mdp.jump_commands import JUMP_COMMAND_NAME
from mjlab_tasks.jumping_robot_balance.robot_cfg import FALL_ANGLE_DEG, ROBOT_ENTITY_NAME

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg(ROBOT_ENTITY_NAME)
_FALL_LIMIT_RAD = math.radians(FALL_ANGLE_DEG)
_ACTIVE_JUMP_FALL_LIMIT_RAD = math.radians(60.0)


def jump_bad_orientation(env: "ManagerBasedRlEnv") -> torch.Tensor:
    robot = env.scene[_ROBOT_CFG.name]
    tilt_l2 = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)
    jump_active = env.command_manager.get_command(JUMP_COMMAND_NAME)[:, 0] > 0.5
    limit_l2 = torch.where(
        jump_active,
        math.sin(_ACTIVE_JUMP_FALL_LIMIT_RAD) ** 2,
        math.sin(_FALL_LIMIT_RAD) ** 2,
    )
    return tilt_l2 > limit_l2


def build_termination_terms(
    jump_stage_two: bool = False,
) -> dict[str, TerminationTermCfg]:
    return {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=jump_bad_orientation if jump_stage_two else envs_mdp.bad_orientation,
            params=(
                {}
                if jump_stage_two
                else {
                    "limit_angle": _FALL_LIMIT_RAD,
                    "asset_cfg": _ROBOT_CFG,
                }
            ),
        ),
        "nan_guard": TerminationTermCfg(func=envs_mdp.nan_detection),
    }
