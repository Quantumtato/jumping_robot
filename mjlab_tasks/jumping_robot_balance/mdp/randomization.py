"""Domain randomization and reset events for robust balancing."""

from __future__ import annotations

import math

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from mjlab_tasks.jumping_robot_balance.robot_cfg import (
    BASE_BODY_NAME,
    FOOT_COLLISION_CAPSULE_RADIUS_M,
    FOOT_COLLISION_LOW_ENDPOINT_AT_ZERO_M,
    LINEAR_JOINT,
    LINEAR_RANGE_CENTER_M,
    ROBOT_ENTITY_NAME,
)

_BASE_BODY_CFG = SceneEntityCfg(ROBOT_ENTITY_NAME, body_names=(BASE_BODY_NAME,))
_LINEAR_CFG = SceneEntityCfg(ROBOT_ENTITY_NAME, joint_names=(LINEAR_JOINT,))


def build_randomization_events() -> dict[str, EventTermCfg]:
    return {
        "reset_robot_state": EventTermCfg(
            func=reset_foot_grounded_state,
            mode="reset",
        ),
    }


def _write_foot_grounded_root_state(
    env,
    asset: Entity,
    env_ids: torch.Tensor,
    linear_pos: torch.Tensor,
    linear_vel: torch.Tensor,
    tilt_limit_rad: float,
    angular_velocity_limit_rad_s: float,
) -> None:
    linear_joint_ids, linear_joint_names = asset.find_joints(LINEAR_JOINT)
    if len(linear_joint_ids) != 1:
        raise ValueError(
            f"Expected one linear joint, found {linear_joint_names}."
        )
    linear_ids = torch.tensor(
        linear_joint_ids,
        dtype=torch.long,
        device=env.device,
    )
    asset.write_joint_state_to_sim(
        linear_pos,
        linear_vel,
        env_ids=env_ids,
        joint_ids=linear_ids,
    )

    default_root_state = asset.data.default_root_state
    if default_root_state is None:
        raise ValueError("Robot default root state is incomplete.")
    root_state = default_root_state[env_ids].clone()
    positions = root_state[:, :3] + env.scene.env_origins[env_ids]
    roll_pitch = sample_uniform(
        -tilt_limit_rad,
        tilt_limit_rad,
        (len(env_ids), 2),
        device=env.device,
    )
    yaw = torch.zeros(len(env_ids), device=env.device)
    orientation_delta = quat_from_euler_xyz(
        roll_pitch[:, 0],
        roll_pitch[:, 1],
        yaw,
    )
    orientations = quat_mul(root_state[:, 3:7], orientation_delta)
    # Place the foot collision capsule's lowest point at z=0 for the sampled
    # linear position and base tilt, rather than leaving a fixed clearance.
    base_z_axis_world_z = 1.0 - 2.0 * (
        orientations[:, 1].square() + orientations[:, 2].square()
    )
    capsule_endpoint_z = (
        FOOT_COLLISION_LOW_ENDPOINT_AT_ZERO_M - linear_pos[:, 0]
    )
    positions[:, 2] = (
        env.scene.env_origins[env_ids, 2]
        + FOOT_COLLISION_CAPSULE_RADIUS_M
        - base_z_axis_world_z * capsule_endpoint_z
    )
    velocities = root_state[:, 7:13].clone()
    if angular_velocity_limit_rad_s:
        velocities[:, 3:5] += sample_uniform(
            -angular_velocity_limit_rad_s,
            angular_velocity_limit_rad_s,
            (len(env_ids), 2),
            device=env.device,
        )
    asset.write_root_link_pose_to_sim(
        torch.cat((positions, orientations), dim=1),
        env_ids=env_ids,
    )
    asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def reset_foot_grounded_state(
    env,
    env_ids: torch.Tensor | None,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    asset: Entity = env.scene[ROBOT_ENTITY_NAME]
    default_joint_pos = asset.data.default_joint_pos
    default_joint_vel = asset.data.default_joint_vel
    if default_joint_pos is None or default_joint_vel is None:
        raise ValueError("Robot default joint state is incomplete.")

    linear_joint_ids, linear_joint_names = asset.find_joints(LINEAR_JOINT)
    if len(linear_joint_ids) != 1:
        raise ValueError(
            f"Expected one linear joint, found {linear_joint_names}."
        )
    linear_ids = torch.tensor(
        linear_joint_ids,
        dtype=torch.long,
        device=env.device,
    )
    _write_foot_grounded_root_state(
        env,
        asset,
        env_ids,
        default_joint_pos[env_ids][:, linear_ids],
        default_joint_vel[env_ids][:, linear_ids],
        math.radians(3.0),
        0.0,
    )


def build_robustness_events() -> dict[str, EventTermCfg]:
    return {
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (-math.radians(5.0), math.radians(5.0)),
                    "pitch": (-math.radians(5.0), math.radians(5.0)),
                    "yaw": (0.0, 0.0),
                },
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (-0.1, 0.1),
                    "pitch": (-0.1, 0.1),
                    "yaw": (0.0, 0.0),
                },
            },
        ),
        "push_disturbance": EventTermCfg(
            func=envs_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(2.5, 5.0),
            params={
                "velocity_range": {
                    "x": (-0.15, 0.15),
                    "y": (-0.15, 0.15),
                    "z": (0.0, 0.0),
                    "roll": (-0.2, 0.2),
                    "pitch": (-0.2, 0.2),
                    "yaw": (0.0, 0.0),
                }
            },
        ),
        "base_com_offset": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": _BASE_BODY_CFG,
                "operation": "add",
                "ranges": {
                    0: (-0.002, 0.002),
                    1: (-0.002, 0.002),
                    2: (-0.003, 0.003),
                },
            },
        ),
    }


def reset_height_control_state(
    env,
    env_ids: torch.Tensor | None,
) -> None:
    from mjlab_tasks.jumping_robot_balance.mdp.height_curriculum import (
        scheduled_height_half_width,
    )

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    asset: Entity = env.scene[ROBOT_ENTITY_NAME]

    default_joint_pos = asset.data.default_joint_pos
    default_joint_vel = asset.data.default_joint_vel
    joint_pos_limits = asset.data.soft_joint_pos_limits
    default_root_state = asset.data.default_root_state
    if (
        default_joint_pos is None
        or default_joint_vel is None
        or joint_pos_limits is None
        or default_root_state is None
    ):
        raise ValueError("Robot reset state is incomplete.")

    linear_joint_ids, linear_joint_names = asset.find_joints(LINEAR_JOINT)
    if len(linear_joint_ids) != 1:
        raise ValueError(
            f"Expected one linear joint, found {linear_joint_names}."
        )
    linear_ids = torch.tensor(
        linear_joint_ids,
        dtype=torch.long,
        device=env.device,
    )
    half_width = scheduled_height_half_width(env.common_step_counter)
    linear_pos = default_joint_pos[env_ids][:, linear_ids].clone()
    linear_pos += sample_uniform(
        -half_width,
        half_width,
        linear_pos.shape,
        device=env.device,
    )
    limits = joint_pos_limits[env_ids][:, linear_ids]
    linear_pos.clamp_(limits[..., 0], limits[..., 1])
    linear_vel = default_joint_vel[env_ids][:, linear_ids]
    _write_foot_grounded_root_state(
        env,
        asset,
        env_ids,
        linear_pos,
        linear_vel,
        math.radians(5.0),
        0.1,
    )


def build_height_robustness_events() -> dict[str, EventTermCfg]:
    events = build_robustness_events()
    del events["reset_base"]
    events["reset_robot_state"] = EventTermCfg(
        func=reset_height_control_state,
        mode="reset",
    )
    return events


def build_jump_stage_two_events() -> dict[str, EventTermCfg]:
    events = build_height_robustness_events()
    del events["push_disturbance"]
    return events
