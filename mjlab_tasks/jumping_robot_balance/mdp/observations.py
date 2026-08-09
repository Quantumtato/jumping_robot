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
from mjlab_tasks.jumping_robot_balance.mdp.jump_commands import (
    JUMP_COMMAND_NAME,
    JumpCommand,
)
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
    target = env.action_manager.get_term("linear_position").position_target
    return (
        2.0
        * (target - LINEAR_RANGE_MIN_M)
        / (LINEAR_RANGE_MAX_M - LINEAR_RANGE_MIN_M)
        - 1.0
    )


def _privileged_root_height(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene[ROBOT_ENTITY_NAME]
    return (
        asset.data.root_link_pos_w[:, 2:3]
        - env.scene.env_origins[:, 2:3]
    )


def _privileged_root_linear_velocity(
    env: "ManagerBasedRlEnv",
) -> torch.Tensor:
    asset: Entity = env.scene[ROBOT_ENTITY_NAME]
    return asset.data.root_link_lin_vel_w


def _privileged_jump_state(env: "ManagerBasedRlEnv") -> torch.Tensor:
    term = env.command_manager.get_term(JUMP_COMMAND_NAME)
    if not isinstance(term, JumpCommand):
        raise TypeError(f"Expected JumpCommand, received {type(term).__name__}.")
    return torch.stack(
        (
            term.has_triggered.float(),
            term.was_airborne.float(),
            term.has_landed.float(),
            term.landing_impact_speed,
        ),
        dim=1,
    )


def _flight_phase_history_state(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Compact proprioceptive state whose history identifies jump phase."""
    return torch.cat(
        (
            _normalized_linear_position(env),
            _normalized_joint_vel(
                env,
                LINEAR_MAX_SPEED_M_S,
                _LINEAR_CFG,
            ),
            envs_mdp.last_action(env),
            foot_ground_contact(env),
            env.command_manager.get_command(JUMP_COMMAND_NAME),
        ),
        dim=1,
    )


def _airborne_time(env: "ManagerBasedRlEnv") -> torch.Tensor:
    term = env.command_manager.get_term(JUMP_COMMAND_NAME)
    if not isinstance(term, JumpCommand):
        raise TypeError(f"Expected JumpCommand, received {type(term).__name__}.")
    return term.airborne_time.unsqueeze(1)


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
        actor_terms["foot_contact"] = ObservationTermCfg(
            func=foot_ground_contact,
        )
    if jump_stage_two:
        actor_terms["jump_command"] = ObservationTermCfg(
            func=envs_mdp.generated_commands,
            params={"command_name": JUMP_COMMAND_NAME},
        )
        actor_terms["flight_phase_history"] = ObservationTermCfg(
            func=_flight_phase_history_state,
            history_length=8,
        )
        actor_terms["airborne_time"] = ObservationTermCfg(
            func=_airborne_time,
        )

    critic_terms = {**actor_terms}
    if jump_stage_two:
        critic_terms["privileged_root_height"] = ObservationTermCfg(
            func=_privileged_root_height,
        )
        critic_terms["privileged_root_linear_velocity"] = ObservationTermCfg(
            func=_privileged_root_linear_velocity,
        )
        critic_terms["privileged_jump_state"] = ObservationTermCfg(
            func=_privileged_jump_state,
        )

    return {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }
