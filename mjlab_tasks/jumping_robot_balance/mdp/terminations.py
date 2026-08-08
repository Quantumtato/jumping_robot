"""Termination terms for the jumping robot balance task."""

from __future__ import annotations

import math
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from mjlab_tasks.jumping_robot_balance.robot_cfg import FALL_ANGLE_DEG, ROBOT_ENTITY_NAME

_ROBOT_CFG = SceneEntityCfg(ROBOT_ENTITY_NAME)
_FALL_LIMIT_RAD = math.radians(FALL_ANGLE_DEG)
def build_termination_terms() -> dict[str, TerminationTermCfg]:
    return {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=envs_mdp.bad_orientation,
            params={
                "limit_angle": _FALL_LIMIT_RAD,
                "asset_cfg": _ROBOT_CFG,
            },
        ),
        "nan_guard": TerminationTermCfg(func=envs_mdp.nan_detection),
    }
