"""Reward terms for the jumping robot balance task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_tasks.jumping_robot_balance.robot_cfg import (
    FALL_ANGLE_DEG,
    FLYWHEEL_X_JOINT,
    FLYWHEEL_Y_JOINT,
    LINEAR_JOINT,
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
_FALL_LIMIT_RAD = math.radians(FALL_ANGLE_DEG)


def upright_alignment(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    z_up_component = -asset.data.projected_gravity_b[:, 2]
    return torch.clamp(z_up_component, min=0.0)


def flywheel_speed_l2(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _FLYWHEEL_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    wheel_speed_norm = asset.data.joint_vel[:, asset_cfg.joint_ids] / MAX_FLYWHEEL_SPEED_RAD_S
    return torch.sum(torch.square(wheel_speed_norm), dim=1)


def linear_velocity_l2(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _LINEAR_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def fell_over(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return envs_mdp.bad_orientation(
        env,
        limit_angle=_FALL_LIMIT_RAD,
        asset_cfg=_ROBOT_CFG,
    ).float()


def build_reward_terms() -> dict[str, RewardTermCfg]:
    return {
        "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=1.0),
        "upright": RewardTermCfg(
            func=upright_alignment,
            weight=1.0,
            params={"asset_cfg": _ROBOT_CFG},
        ),
        "flywheel_speed": RewardTermCfg(
            func=flywheel_speed_l2,
            weight=-0.1,
            params={"asset_cfg": _FLYWHEEL_CFG},
        ),
        "linear_velocity": RewardTermCfg(
            func=linear_velocity_l2,
            weight=-0.002,
            params={"asset_cfg": _LINEAR_CFG},
        ),
        "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
        "fall_event": RewardTermCfg(func=fell_over, weight=-100.0),
    }
