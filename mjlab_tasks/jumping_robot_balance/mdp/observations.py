"""Observation terms for the jumping robot balance task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, sample_uniform, yaw_quat
from mjlab.utils.noise import UniformNoiseCfg

from mjlab_tasks.jumping_robot_balance.mdp.commands import (
    HEIGHT_COMMAND_NAME,
    IMU_ACCEL_SCALE_M_S2,
    PLANAR_VELOCITY_COMMAND_NAME,
    PLANAR_VELOCITY_SCALE_M_S,
    PlanarVelocityCommand,
)
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


_MOTOR_ACTION_DIM = 3


def _motor_last_action(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Last motor actions (flywheels + linear leg), excluding jump_request.

    Keeps the observation layout identical across stages so warm starts from
    motor-only checkpoints stay aligned.
    """
    return env.action_manager.action[:, :_MOTOR_ACTION_DIM]


# Joints whose state feeds the flight-phase history, in the order the
# pre-rotor model exposed them. Historical quirk kept for checkpoint
# compatibility: the unresolved _LINEAR_CFG used to fall through to ALL
# joints, so v12-era policies trained on (flywheelY, flywheelX, linear)
# pos/vel here, all normalized by the LINEAR joint's range/speed. The
# legrotor joint must stay excluded: it is kinematically locked to the
# slide (redundant) and including it would shift the observation layout.
_HISTORY_JOINT_NAMES = (FLYWHEEL_Y_JOINT, FLYWHEEL_X_JOINT, LINEAR_JOINT)


def _history_joint_ids(asset: Entity) -> list[int]:
    names = list(asset.joint_names)
    return [names.index(name) for name in _HISTORY_JOINT_NAMES]


def _flight_phase_history_state(
    env: "ManagerBasedRlEnv",
    sensor_noise: bool = False,
) -> torch.Tensor:
    """Compact proprioceptive state whose history identifies jump phase."""
    asset: Entity = env.scene[ROBOT_ENTITY_NAME]
    joint_ids = _history_joint_ids(asset)
    q = asset.data.joint_pos[:, joint_ids]
    qd = asset.data.joint_vel[:, joint_ids]
    pos_norm = (
        2.0 * (q - LINEAR_RANGE_MIN_M) / (LINEAR_RANGE_MAX_M - LINEAR_RANGE_MIN_M)
        - 1.0
    )
    vel_norm = qd / LINEAR_MAX_SPEED_M_S
    state = torch.cat(
        (
            pos_norm,
            vel_norm,
            _motor_last_action(env),
            foot_ground_contact(env),
            env.command_manager.get_command(JUMP_COMMAND_NAME),
        ),
        dim=1,
    )
    if sensor_noise:
        noise = torch.zeros_like(state)
        noise[:, :1] = sample_uniform(
            -0.005,
            0.005,
            (env.num_envs, 1),
            device=env.device,
        )
        noise[:, 1:2] = sample_uniform(
            -0.001,
            0.001,
            (env.num_envs, 1),
            device=env.device,
        )
        return state + noise
    return state


def _airborne_time(env: "ManagerBasedRlEnv") -> torch.Tensor:
    term = env.command_manager.get_term(JUMP_COMMAND_NAME)
    if not isinstance(term, JumpCommand):
        raise TypeError(f"Expected JumpCommand, received {type(term).__name__}.")
    return term.airborne_time.unsqueeze(1)


def _planar_velocity_command(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return env.command_manager.get_command(PLANAR_VELOCITY_COMMAND_NAME)


def _heading_quat(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene[ROBOT_ENTITY_NAME]
    return yaw_quat(asset.data.root_link_quat_w)


def _heading_frame_velocity_command(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """World command rotated into the robot heading frame.

    The actor has no yaw sensor, so a raw world-frame command becomes
    unobservable once heading drifts; rotating it here keeps the task
    observable while the user still commands in the terrain frame.
    """
    command = env.command_manager.get_command(PLANAR_VELOCITY_COMMAND_NAME)
    command_w = torch.cat(
        (command, torch.zeros_like(command[:, :1])),
        dim=1,
    )
    rotated = quat_apply_inverse(_heading_quat(env), command_w)
    return torch.nan_to_num(rotated[:, :2])


def _imu_specific_force(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Body-frame IMU specific force so the actor can estimate its velocity."""
    term = env.command_manager.get_term(PLANAR_VELOCITY_COMMAND_NAME)
    if not isinstance(term, PlanarVelocityCommand):
        raise TypeError(
            f"Expected PlanarVelocityCommand, received {type(term).__name__}."
        )
    return torch.clamp(
        term.imu_specific_force_b / IMU_ACCEL_SCALE_M_S2,
        min=-4.0,
        max=4.0,
    )


def _heading_frame_average_velocity(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Hop-averaged planar velocity in the heading frame, for the actor.

    Stands in for the short-horizon IMU dead-reckoning estimate the hardware
    computes between landings; training adds noise on top so the policy can't
    over-trust it. Normalized by the command speed scale.
    """
    term = env.command_manager.get_term(PLANAR_VELOCITY_COMMAND_NAME)
    if not isinstance(term, PlanarVelocityCommand):
        raise TypeError(
            f"Expected PlanarVelocityCommand, received {type(term).__name__}."
        )
    velocity_w = torch.cat(
        (
            term.average_planar_velocity,
            torch.zeros_like(term.average_planar_velocity[:, :1]),
        ),
        dim=1,
    )
    velocity_b = quat_apply_inverse(_heading_quat(env), velocity_w)
    return torch.clamp(
        torch.nan_to_num(velocity_b[:, :2]) / PLANAR_VELOCITY_SCALE_M_S,
        min=-4.0,
        max=4.0,
    )


def _normalized_heading_frame_planar_velocity(
    env: "ManagerBasedRlEnv",
) -> torch.Tensor:
    asset: Entity = env.scene[ROBOT_ENTITY_NAME]
    velocity_b = quat_apply_inverse(
        _heading_quat(env),
        asset.data.root_link_lin_vel_w,
    )
    return torch.clamp(
        torch.nan_to_num(velocity_b[:, :2]) / PLANAR_VELOCITY_SCALE_M_S,
        min=-4.0,
        max=4.0,
    )


def build_observation_groups(
    height_control: bool = False,
    jump_stage_one: bool = False,
    jump_stage_two: bool = False,
    sensor_noise: bool = False,
    navigation: bool = False,
    actor_velocity_estimate: bool = False,
) -> dict[str, ObservationGroupCfg]:
    actor_terms = {
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            params={"asset_cfg": _ROBOT_CFG},
            noise=(
                UniformNoiseCfg(n_min=-0.01, n_max=0.01)
                if sensor_noise
                else None
            ),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            params={"asset_cfg": _ROBOT_CFG},
            noise=(
                UniformNoiseCfg(n_min=-0.05, n_max=0.05) if sensor_noise else None
            ),
        ),
        "flywheel_vel_norm": ObservationTermCfg(
            func=_normalized_joint_vel,
            params={"denom": MAX_FLYWHEEL_SPEED_RAD_S, "asset_cfg": _FLYWHEEL_CFG},
            noise=(
                UniformNoiseCfg(n_min=-0.005, n_max=0.005)
                if sensor_noise
                else None
            ),
        ),
        "linear_pos_norm": ObservationTermCfg(
            func=_normalized_linear_position,
            params={"asset_cfg": _LINEAR_CFG},
            noise=(
                UniformNoiseCfg(n_min=-0.005, n_max=0.005)
                if sensor_noise
                else None
            ),
        ),
        "linear_vel_norm": ObservationTermCfg(
            func=_normalized_joint_vel,
            params={"denom": LINEAR_MAX_SPEED_M_S, "asset_cfg": _LINEAR_CFG},
            noise=(
                UniformNoiseCfg(n_min=-0.001, n_max=0.001)
                if sensor_noise
                else None
            ),
        ),
        "last_action": ObservationTermCfg(func=_motor_last_action),
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
            params={"sensor_noise": sensor_noise},
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
    if navigation:
        # Append these terms after the existing actor and critic layouts so a
        # balance checkpoint's inputs retain their exact feature offsets.
        actor_terms["planar_velocity_command"] = ObservationTermCfg(
            func=_heading_frame_velocity_command,
        )
        # 64 steps at 100 Hz spans a bit more than a full hop, giving the
        # actor enough IMU history to integrate its own planar velocity.
        actor_terms["imu_specific_force_history"] = ObservationTermCfg(
            func=_imu_specific_force,
            noise=(
                UniformNoiseCfg(n_min=-0.02, n_max=0.02)
                if sensor_noise
                else None
            ),
            history_length=64,
        )
        critic_terms["planar_velocity_command"] = ObservationTermCfg(
            func=_heading_frame_velocity_command,
        )
        critic_terms["planar_velocity"] = ObservationTermCfg(
            func=_normalized_heading_frame_planar_velocity,
        )
    if navigation and actor_velocity_estimate:
        # MUST stay the last actor term: warm starts from estimate-less
        # checkpoints copy old input weights into the leading columns and
        # zero-init only the trailing ones.
        actor_terms["velocity_estimate"] = ObservationTermCfg(
            func=_heading_frame_average_velocity,
            noise=(
                UniformNoiseCfg(n_min=-0.1, n_max=0.1)
                if sensor_noise
                else None
            ),
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
