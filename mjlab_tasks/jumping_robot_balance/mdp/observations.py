"""Observation terms for the jumping robot balance task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_tasks.jumping_robot_balance.robot_cfg import (
    FLYWHEEL_X_JOINT,
    FLYWHEEL_Y_JOINT,
    LINEAR_JOINT,
    LINEAR_MAX_SPEED_M_S,
    LINEAR_RANGE_MAX_M,
    LINEAR_RANGE_MIN_M,
    MAX_FLYWHEEL_SPEED_RAD_S,
    ROBOT_ENTITY_NAME,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg(ROBOT_ENTITY_NAME)
_FLYWHEEL_CFG = SceneEntityCfg(
    ROBOT_ENTITY_NAME,
    joint_names=(FLYWHEEL_X_JOINT, FLYWHEEL_Y_JOINT),
)
_LINEAR_CFG = SceneEntityCfg(ROBOT_ENTITY_NAME, joint_names=(LINEAR_JOINT,))


def _normalized_joint_vel(
    env: "ManagerBasedRlEnv",
    denom: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids] / denom


def _normalized_linear_position(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _LINEAR_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return 2.0 * (q - LINEAR_RANGE_MIN_M) / (LINEAR_RANGE_MAX_M - LINEAR_RANGE_MIN_M) - 1.0


def build_observation_groups() -> dict[str, ObservationGroupCfg]:
    actor_terms = {
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            params={"asset_cfg": _ROBOT_CFG},
        ),
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            params={"asset_cfg": _ROBOT_CFG},
        ),
        "flywheel_vel_norm": ObservationTermCfg(
            func=_normalized_joint_vel,
            params={"denom": MAX_FLYWHEEL_SPEED_RAD_S, "asset_cfg": _FLYWHEEL_CFG},
        ),
        "linear_pos_norm": ObservationTermCfg(
            func=_normalized_linear_position,
            params={"asset_cfg": _LINEAR_CFG},
        ),
        "linear_vel_norm": ObservationTermCfg(
            func=_normalized_joint_vel,
            params={"denom": LINEAR_MAX_SPEED_M_S, "asset_cfg": _LINEAR_CFG},
        ),
        "last_action": ObservationTermCfg(func=envs_mdp.last_action),
    }

    return {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms={**actor_terms},
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }
