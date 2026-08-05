"""Observation terms for the jumping robot balance task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_tasks.jumping_robot_balance.mdp.commands import HEIGHT_COMMAND_NAME
from mjlab_tasks.jumping_robot_balance.mdp.contact import foot_ground_contact
from mjlab_tasks.jumping_robot_balance.mdp.jump_commands import JUMP_COMMAND_NAME
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
    preserve_order=True,
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


def _normalized_height_command(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = env.command_manager.get_command(HEIGHT_COMMAND_NAME)
    return (
        2.0
        * (command - LINEAR_RANGE_MIN_M)
        / (LINEAR_RANGE_MAX_M - LINEAR_RANGE_MIN_M)
        - 1.0
    )


def _normalized_linear_position_target(
    env: "ManagerBasedRlEnv",
) -> torch.Tensor:
    target = env.action_manager.get_term("linear_impedance").position_target
    return (
        2.0
        * (target - LINEAR_RANGE_MIN_M)
        / (LINEAR_RANGE_MAX_M - LINEAR_RANGE_MIN_M)
        - 1.0
    )


def _normalized_linear_velocity_target(
    env: "ManagerBasedRlEnv",
) -> torch.Tensor:
    term = env.action_manager.get_term("linear_impedance")
    return term.velocity_target / term.cfg.velocity_target_scale_m_s


def build_observation_groups(
    height_control: bool = False,
    jump_stage_one: bool = False,
    jump_stage_two: bool = False,
) -> dict[str, ObservationGroupCfg]:
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
    if height_control:
        actor_terms["height_command"] = ObservationTermCfg(
            func=_normalized_height_command,
        )
        actor_terms["linear_position_target"] = ObservationTermCfg(
            func=_normalized_linear_position_target,
        )
    if jump_stage_one or jump_stage_two:
        actor_terms["linear_velocity_target"] = ObservationTermCfg(
            func=_normalized_linear_velocity_target,
        )
        actor_terms["foot_contact"] = ObservationTermCfg(
            func=foot_ground_contact,
        )
    if jump_stage_two:
        actor_terms["jump_command"] = ObservationTermCfg(
            func=envs_mdp.generated_commands,
            params={"command_name": JUMP_COMMAND_NAME},
        )

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
