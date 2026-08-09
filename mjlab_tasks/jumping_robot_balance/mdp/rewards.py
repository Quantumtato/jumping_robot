"""Reward terms for the jumping robot balance task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_tasks.jumping_robot_balance.mdp.commands import HEIGHT_COMMAND_NAME
from mjlab_tasks.jumping_robot_balance.mdp.contact import foot_ground_contact
from mjlab_tasks.jumping_robot_balance.mdp.jump_commands import (
    JUMP_COMMAND_NAME,
    JumpCommand,
)
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


def upright_stability(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    horizontal_gravity_l2 = torch.sum(
        torch.square(asset.data.projected_gravity_b[:, :2]),
        dim=1,
    )
    width = math.sin(math.radians(10.0)) ** 2
    return torch.exp(-horizontal_gravity_l2 / width)


def tilt_error_l2(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def base_angular_velocity_l2(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)


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


def _linear_action_start(env: "ManagerBasedRlEnv") -> int:
    term = env.action_manager.get_term("linear_position")
    return env.action_manager.total_action_dim - term.action_dim


def linear_action_rate_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    index = _linear_action_start(env)
    delta = env.action_manager.action[:, index] - env.action_manager.prev_action[:, index]
    return torch.square(delta)


def off_ground(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return 1.0 - foot_ground_contact(env)[:, 0]


def _jump_term(env: "ManagerBasedRlEnv") -> JumpCommand:
    term = env.command_manager.get_term(JUMP_COMMAND_NAME)
    if not isinstance(term, JumpCommand):
        raise TypeError(f"Expected JumpCommand, received {type(term).__name__}.")
    return term


def _jump_active(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return env.command_manager.get_command(JUMP_COMMAND_NAME)[:, 0]


def _balance_only(
    env: "ManagerBasedRlEnv",
    value: torch.Tensor,
) -> torch.Tensor:
    return value * (1.0 - _jump_active(env))


def balance_off_ground(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _balance_only(env, off_ground(env))


def height_command_tracking(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _LINEAR_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    position = asset.data.joint_pos[:, asset_cfg.joint_ids]
    command = env.command_manager.get_command(HEIGHT_COMMAND_NAME)
    error = torch.sum(torch.abs(position - command), dim=1)
    return torch.exp(-error / 0.025)


def balance_height_command_tracking(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _LINEAR_CFG,
) -> torch.Tensor:
    return height_command_tracking(env, asset_cfg) * (1.0 - _jump_active(env))


def balance_linear_velocity_l2(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _LINEAR_CFG,
) -> torch.Tensor:
    return _balance_only(env, linear_velocity_l2(env, asset_cfg))


def jump_apex_progress(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _jump_term(env).apex_progress_delta


def landing_impact_speed_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    term = _jump_term(env)
    return term.landing_event * torch.square(term.landing_impact_speed)


def landing_angular_velocity_l2(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    return _jump_term(env).landing_event * base_angular_velocity_l2(
        env,
        asset_cfg,
    )


def landing_recovery_success(env: "ManagerBasedRlEnv") -> torch.Tensor:
    term = _jump_term(env)
    return term.landing_recovery_success_event * term.apex_height


def balance_linear_action_rate_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _balance_only(env, linear_action_rate_l2(env))


def fell_over(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return envs_mdp.bad_orientation(
        env,
        limit_angle=_FALL_LIMIT_RAD,
        asset_cfg=_ROBOT_CFG,
    ).float()


def build_reward_terms(
    height_control: bool = False,
    jump_stage_one: bool = False,
    jump_stage_two: bool = False,
) -> dict[str, RewardTermCfg]:
    terms = {
        "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=1.0),
        "upright": RewardTermCfg(
            func=upright_stability,
            weight=3.0,
            params={"asset_cfg": _ROBOT_CFG},
        ),
        "tilt_error": RewardTermCfg(
            func=tilt_error_l2,
            weight=-4.0,
            params={"asset_cfg": _ROBOT_CFG},
        ),
        "base_angular_velocity": RewardTermCfg(
            func=base_angular_velocity_l2,
            weight=-0.05,
            params={"asset_cfg": _ROBOT_CFG},
        ),
        "flywheel_speed": RewardTermCfg(
            func=flywheel_speed_l2,
            weight=-0.05,
            params={"asset_cfg": _FLYWHEEL_CFG},
        ),
        "linear_velocity": RewardTermCfg(
            func=linear_velocity_l2,
            weight=-0.002,
            params={"asset_cfg": _LINEAR_CFG},
        ),
        "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.001),
        "fall_event": RewardTermCfg(func=fell_over, weight=-10_000.0),
    }
    if height_control:
        terms["upright"].weight = 4.0
        terms["linear_velocity"].weight = -0.01
        terms["linear_action_rate"] = RewardTermCfg(
            func=linear_action_rate_l2,
            weight=-0.005,
        )
        terms["height_tracking"] = RewardTermCfg(
            func=height_command_tracking,
            weight=2.0,
            params={"asset_cfg": _LINEAR_CFG},
        )
    if jump_stage_one or jump_stage_two:
        terms["off_ground"] = RewardTermCfg(func=off_ground, weight=-2.0)
    if jump_stage_two:
        terms["linear_velocity"].func = balance_linear_velocity_l2
        terms["linear_action_rate"].func = balance_linear_action_rate_l2
        terms["height_tracking"].func = balance_height_command_tracking
        terms["height_tracking"].weight = 0.5
        terms["off_ground"].func = balance_off_ground
        terms["jump_apex_progress"] = RewardTermCfg(
            func=jump_apex_progress,
            weight=100_000.0,
        )
        terms["landing_recovery_success"] = RewardTermCfg(
            func=landing_recovery_success,
            weight=100_000.0,
        )
        terms["landing_impact_speed"] = RewardTermCfg(
            func=landing_impact_speed_l2,
            weight=-500.0,
        )
        terms["landing_angular_velocity"] = RewardTermCfg(
            func=landing_angular_velocity_l2,
            weight=-250.0,
            params={"asset_cfg": _ROBOT_CFG},
        )
    return terms
