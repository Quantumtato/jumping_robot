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
    PHASE_FLIGHT,
    PHASE_RECOVERY,
    PHASE_TAKEOFF,
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
    term = env.action_manager.get_term("linear_impedance")
    return env.action_manager.total_action_dim - term.action_dim


def linear_action_rate_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    index = _linear_action_start(env)
    delta = env.action_manager.action[:, index] - env.action_manager.prev_action[:, index]
    return torch.square(delta)


def linear_feedforward_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return torch.square(env.action_manager.action[:, -1])


def linear_feedforward_rate_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    delta = env.action_manager.action[:, -1] - env.action_manager.prev_action[:, -1]
    return torch.square(delta)


def linear_velocity_action_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    index = _linear_action_start(env) + 1
    return torch.square(env.action_manager.action[:, index])


def linear_velocity_action_rate_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    index = _linear_action_start(env) + 1
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


def balance_off_ground(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return off_ground(env) * (1.0 - _jump_active(env))


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
    return linear_velocity_l2(env, asset_cfg) * (1.0 - _jump_active(env))


def jump_apex_progress(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _jump_term(env).apex_progress_delta


def jump_target_reached(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _jump_term(env).target_reached_event


def takeoff_upward_velocity(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    takeoff = (_jump_term(env).phase == PHASE_TAKEOFF).float()
    return takeoff * torch.clamp(
        asset.data.root_link_lin_vel_w[:, 2] / 3.0,
        0.0,
        1.0,
    )


def airborne_upright(env: "ManagerBasedRlEnv") -> torch.Tensor:
    flight = (_jump_term(env).phase == PHASE_FLIGHT).float()
    return flight * upright_stability(env)


def landing_quality(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _jump_term(env).landing_quality_event


def landing_impact(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _jump_term(env).landing_impact_event


def stable_recovery(env: "ManagerBasedRlEnv") -> torch.Tensor:
    term = _jump_term(env)
    return (
        (term.phase == PHASE_RECOVERY)
        & (term.stable_time > 0.0)
        & term.reached_target
    ).float()


def jump_completed(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _jump_term(env).completed_event


def jump_missed(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _jump_term(env).missed_event


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
            weight=-2.0,
            params={"asset_cfg": _ROBOT_CFG},
        ),
        "base_angular_velocity": RewardTermCfg(
            func=base_angular_velocity_l2,
            weight=-0.05,
            params={"asset_cfg": _ROBOT_CFG},
        ),
        "flywheel_speed": RewardTermCfg(
            func=flywheel_speed_l2,
            weight=-0.2,
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
        terms["linear_feedforward"] = RewardTermCfg(
            func=linear_feedforward_l2,
            weight=-0.002,
        )
        terms["linear_feedforward_rate"] = RewardTermCfg(
            func=linear_feedforward_rate_l2,
            weight=-0.002,
        )
        terms["height_tracking"] = RewardTermCfg(
            func=height_command_tracking,
            weight=2.0,
            params={"asset_cfg": _LINEAR_CFG},
        )
    if jump_stage_one or jump_stage_two:
        terms["linear_velocity_action"] = RewardTermCfg(
            func=linear_velocity_action_l2,
            weight=-0.001,
        )
        terms["linear_velocity_action_rate"] = RewardTermCfg(
            func=linear_velocity_action_rate_l2,
            weight=-0.002,
        )
        terms["off_ground"] = RewardTermCfg(func=off_ground, weight=-2.0)
    if jump_stage_two:
        terms["fall_event"].weight = -20_000.0
        terms["linear_velocity"].func = balance_linear_velocity_l2
        terms["height_tracking"].func = balance_height_command_tracking
        terms["off_ground"].func = balance_off_ground
        terms["jump_apex_progress"] = RewardTermCfg(
            func=jump_apex_progress,
            weight=10_000.0,
        )
        terms["jump_target_reached"] = RewardTermCfg(
            func=jump_target_reached,
            weight=5_000.0,
        )
        terms["takeoff_upward_velocity"] = RewardTermCfg(
            func=takeoff_upward_velocity,
            weight=1.0,
            params={"asset_cfg": _ROBOT_CFG},
        )
        terms["airborne_upright"] = RewardTermCfg(
            func=airborne_upright,
            weight=3.0,
        )
        terms["landing_quality"] = RewardTermCfg(
            func=landing_quality,
            weight=5_000.0,
        )
        terms["landing_impact"] = RewardTermCfg(
            func=landing_impact,
            weight=-2_000.0,
        )
        terms["stable_recovery"] = RewardTermCfg(
            func=stable_recovery,
            weight=5.0,
        )
        terms["jump_completed"] = RewardTermCfg(
            func=jump_completed,
            weight=10_000.0,
        )
        terms["jump_missed"] = RewardTermCfg(
            func=jump_missed,
            weight=-15_000.0,
        )
    return terms
