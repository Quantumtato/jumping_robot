"""Domain randomization and reset events for robust balancing."""

from __future__ import annotations

import math

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from mjlab_tasks.jumping_robot_balance.robot_cfg import (
    BASE_BODY_NAME,
    FOOT_COLLISION_CAPSULE_RADIUS_M,
    FOOT_COLLISION_LOW_ENDPOINT_AT_ZERO_M,
    LEG_ROTOR_JOINT,
    LINEAR_JOINT,
    LINEAR_RANGE_CENTER_M,
    LINEAR_RANGE_HALF_WIDTH_M,
    ROBOT_ENTITY_NAME,
    SCREW_COUPLING_RAD_PER_M,
)

_BASE_BODY_CFG = SceneEntityCfg(ROBOT_ENTITY_NAME, body_names=(BASE_BODY_NAME,))
_LINEAR_CFG = SceneEntityCfg(ROBOT_ENTITY_NAME, joint_names=(LINEAR_JOINT,))
_FLYWHEEL_ACTUATOR_NAMES = (
    f"{ROBOT_ENTITY_NAME}/flywheelX",
    f"{ROBOT_ENTITY_NAME}/flywheelY",
)


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
    # Keep the ballscrew rotor consistent with the randomized slide position;
    # otherwise the equality constraint snaps them together at reset and
    # kicks the base in yaw (up to ~8 rad of mismatch resolving in ~20 ms).
    rotor_joint_ids, _ = asset.find_joints(LEG_ROTOR_JOINT)
    if rotor_joint_ids:
        rotor_ids = torch.tensor(
            rotor_joint_ids,
            dtype=torch.long,
            device=env.device,
        )
        asset.write_joint_state_to_sim(
            linear_pos * SCREW_COUPLING_RAD_PER_M,
            linear_vel * SCREW_COUPLING_RAD_PER_M,
            env_ids=env_ids,
            joint_ids=rotor_ids,
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


def build_robustness_events(
    push_interval_range_s: tuple[float, float] = (2.5, 5.0),
    push_linear_velocity_limit_m_s: float = 0.15,
    push_angular_velocity_limit_rad_s: float = 0.2,
    com_offset_xy_limit_m: float = 0.002,
    com_offset_z_limit_m: float = 0.003,
) -> dict[str, EventTermCfg]:
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
            interval_range_s=push_interval_range_s,
            params={
                "velocity_range": {
                    "x": (
                        -push_linear_velocity_limit_m_s,
                        push_linear_velocity_limit_m_s,
                    ),
                    "y": (
                        -push_linear_velocity_limit_m_s,
                        push_linear_velocity_limit_m_s,
                    ),
                    "z": (0.0, 0.0),
                    "roll": (
                        -push_angular_velocity_limit_rad_s,
                        push_angular_velocity_limit_rad_s,
                    ),
                    "pitch": (
                        -push_angular_velocity_limit_rad_s,
                        push_angular_velocity_limit_rad_s,
                    ),
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
                    0: (-com_offset_xy_limit_m, com_offset_xy_limit_m),
                    1: (-com_offset_xy_limit_m, com_offset_xy_limit_m),
                    2: (-com_offset_z_limit_m, com_offset_z_limit_m),
                },
            },
        ),
    }


def reset_height_control_state(
    env,
    env_ids: torch.Tensor | None,
    range_schedule: tuple[tuple[int, float], ...] | None = None,
    tilt_limit_rad: float = math.radians(5.0),
    angular_velocity_limit_rad_s: float = 0.1,
) -> None:
    from mjlab_tasks.jumping_robot_balance.mdp.height_curriculum import (
        HEIGHT_RANGE_SCHEDULE,
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
    half_width = scheduled_height_half_width(
        env.common_step_counter,
        range_schedule if range_schedule is not None else HEIGHT_RANGE_SCHEDULE,
    )
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
        tilt_limit_rad,
        angular_velocity_limit_rad_s,
    )


def build_height_robustness_events(
    full_stroke: bool = False,
    reset_tilt_limit_deg: float = 5.0,
    reset_angular_velocity_limit_rad_s: float = 0.1,
    push_interval_range_s: tuple[float, float] = (2.5, 5.0),
    push_linear_velocity_limit_m_s: float = 0.15,
    push_angular_velocity_limit_rad_s: float = 0.2,
    com_offset_xy_limit_m: float = 0.002,
    com_offset_z_limit_m: float = 0.003,
) -> dict[str, EventTermCfg]:
    events = build_robustness_events(
        push_interval_range_s=push_interval_range_s,
        push_linear_velocity_limit_m_s=push_linear_velocity_limit_m_s,
        push_angular_velocity_limit_rad_s=push_angular_velocity_limit_rad_s,
        com_offset_xy_limit_m=com_offset_xy_limit_m,
        com_offset_z_limit_m=com_offset_z_limit_m,
    )
    del events["reset_base"]
    events["reset_robot_state"] = EventTermCfg(
        func=reset_height_control_state,
        mode="reset",
        params={
            "range_schedule": (
                ((0, LINEAR_RANGE_HALF_WIDTH_M),)
                if full_stroke
                else None
            ),
            "tilt_limit_rad": math.radians(reset_tilt_limit_deg),
            "angular_velocity_limit_rad_s": reset_angular_velocity_limit_rad_s,
        },
    )
    return events


def scheduled_push_by_setting_velocity(
    env,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    off_step: int,
    ramp_start_step: int,
    ramp_end_step: int,
    ramp_final_scale: float,
) -> None:
    """Push with pipeline-phase-scheduled strength (v19).

    Full strength while the policy learns balance, zero while hop placement
    and velocity tracking are first learned (a 0.25 m/s shove is 2.5x the
    initial command cap and drowns the displacement gradient), then linearly
    ramped back in late for deployment robustness.
    """
    step = env.common_step_counter
    if step < off_step:
        scale = 1.0
    elif step < ramp_start_step:
        scale = 0.0
    else:
        progress = min(
            1.0,
            (step - ramp_start_step) / max(1, ramp_end_step - ramp_start_step),
        )
        scale = ramp_final_scale * progress
    if scale <= 0.0:
        return
    scaled_range = {
        axis: (low * scale, high * scale)
        for axis, (low, high) in velocity_range.items()
    }
    envs_mdp.push_by_setting_velocity(
        env,
        env_ids,
        velocity_range=scaled_range,
    )


def build_strong_robustness_events(
    push_scale_schedule: tuple[int, int, int, float] | None = None,
) -> dict[str, EventTermCfg]:
    """Build full-stroke perturbations for the second balance robustness pass."""
    events = build_height_robustness_events(
        full_stroke=True,
        reset_tilt_limit_deg=7.0,
        reset_angular_velocity_limit_rad_s=0.15,
        push_interval_range_s=(2.0, 4.0),
        push_linear_velocity_limit_m_s=0.25,
        push_angular_velocity_limit_rad_s=0.35,
        com_offset_xy_limit_m=0.003,
        com_offset_z_limit_m=0.004,
    )
    events["inertial_properties"] = EventTermCfg(
        func=dr.pseudo_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(ROBOT_ENTITY_NAME),
            "alpha_range": (
                0.5 * math.log(0.95),
                0.5 * math.log(1.05),
            ),
        },
    )
    events["actuator_authority"] = EventTermCfg(
        func=randomize_actuator_authority,
        mode="startup",
        params={
            "authority_scale_range": (0.95, 1.05),
            "velocity_gain_scale_range": (0.95, 1.05),
        },
    )
    events["linear_encoder_bias"] = EventTermCfg(
        func=dr.encoder_bias,
        mode="reset",
        params={
            "asset_cfg": _LINEAR_CFG,
            "bias_range": (-0.0005, 0.0005),
        },
    )
    if push_scale_schedule is not None:
        off_step, ramp_start_step, ramp_end_step, final_scale = (
            push_scale_schedule
        )
        push = events["push_disturbance"]
        events["push_disturbance"] = EventTermCfg(
            func=scheduled_push_by_setting_velocity,
            mode="interval",
            interval_range_s=push.interval_range_s,
            params={
                "velocity_range": push.params["velocity_range"],
                "off_step": off_step,
                "ramp_start_step": ramp_start_step,
                "ramp_end_step": ramp_end_step,
                "ramp_final_scale": final_scale,
            },
        )
    return events


def build_warm_start_jump_events() -> dict[str, EventTermCfg]:
    """Use light disturbances until the policy establishes jump and recovery."""
    return build_height_robustness_events(
        full_stroke=True,
        reset_tilt_limit_deg=5.0,
        reset_angular_velocity_limit_rad_s=0.1,
        push_interval_range_s=(4.0, 6.0),
        push_linear_velocity_limit_m_s=0.1,
        push_angular_velocity_limit_rad_s=0.15,
        com_offset_xy_limit_m=0.002,
        com_offset_z_limit_m=0.003,
    )


@requires_model_fields("actuator_forcerange", "actuator_gainprm")
def randomize_actuator_authority(
    env,
    env_ids: torch.Tensor | None,
    authority_scale_range: tuple[float, float] = (0.9, 1.1),
    velocity_gain_scale_range: tuple[float, float] = (0.9, 1.1),
) -> None:
    """Randomize available force and flywheel velocity-servo gain per environment."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    model = env.sim.model
    default_forcerange = env.sim.get_default_field("actuator_forcerange")
    actuator_ids = torch.arange(
        default_forcerange.shape[0],
        device=env.device,
        dtype=torch.long,
    )
    authority_scale = sample_uniform(
        authority_scale_range[0],
        authority_scale_range[1],
        (len(env_ids), len(actuator_ids), 1),
        device=env.device,
    )
    model.actuator_forcerange[env_ids[:, None], actuator_ids] = (
        default_forcerange[actuator_ids] * authority_scale
    )

    flywheel_actuator_ids = torch.tensor(
        [
            mujoco.mj_name2id(
                env.sim.mj_model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_name,
            )
            for actuator_name in _FLYWHEEL_ACTUATOR_NAMES
        ],
        device=env.device,
        dtype=torch.long,
    )
    if torch.any(flywheel_actuator_ids < 0):
        raise ValueError("Could not resolve flywheel velocity actuators.")
    default_gainprm = env.sim.get_default_field("actuator_gainprm")
    velocity_gain_scale = sample_uniform(
        velocity_gain_scale_range[0],
        velocity_gain_scale_range[1],
        (len(env_ids), len(flywheel_actuator_ids), 1),
        device=env.device,
    )
    model.actuator_gainprm[env_ids[:, None], flywheel_actuator_ids] = (
        default_gainprm[flywheel_actuator_ids] * velocity_gain_scale
    )


def build_jump_stage_two_events() -> dict[str, EventTermCfg]:
    events = build_height_robustness_events()
    del events["push_disturbance"]
    return events
