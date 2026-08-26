"""Reward terms for the jumping robot balance task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_tasks.jumping_robot_balance.mdp.commands import (
    HEIGHT_COMMAND_NAME,
    PLANAR_VELOCITY_COMMAND_NAME,
    PLANAR_VELOCITY_SCALE_M_S,
    PlanarVelocityCommand,
)
from mjlab_tasks.jumping_robot_balance.mdp.contact import (
    foot_ground_contact,
    foot_height_w,
)
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
SMALL_JUMP_HEIGHT_SCHEDULE: tuple[tuple[int, float], ...] = (
    (0, 0.015),
    (64_000, 0.025),
    (128_000, 0.040),
)


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
    manager = env.action_manager
    start = 0
    for name in manager.active_terms:
        if name == "linear_position":
            return start
        start += manager.get_term(name).action_dim
    raise ValueError("linear_position action term not found.")


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


def _small_jump_target_height(
    env: "ManagerBasedRlEnv",
    schedule: tuple[tuple[int, float], ...] = SMALL_JUMP_HEIGHT_SCHEDULE,
) -> float:
    target = schedule[0][1]
    for threshold, height in schedule:
        if env.common_step_counter < threshold:
            break
        target = height
    return target


def capped_small_jump_apex_progress(
    env: "ManagerBasedRlEnv",
    target_schedule: tuple[tuple[int, float], ...] = SMALL_JUMP_HEIGHT_SCHEDULE,
    use_jump_curriculum: bool = False,
) -> torch.Tensor:
    """Reward ascent only until the current small-jump target is reached."""
    term = _jump_term(env)
    target_height = (
        term.current_target_height
        if use_jump_curriculum
        else _small_jump_target_height(env, target_schedule)
    )
    previous_apex = term.apex_height - term.apex_progress_delta
    return torch.clamp(
        torch.minimum(term.apex_height, torch.full_like(term.apex_height, target_height))
        - torch.minimum(
            previous_apex,
            torch.full_like(previous_apex, target_height),
        ),
        min=0.0,
    )


def sustained_flight_clearance(
    env: "ManagerBasedRlEnv",
    descent_velocity_cutoff_m_s: float = -0.10,
) -> torch.Tensor:
    """Foot clearance (capped at the curriculum target) integrated over flight.

    The max-progress apex reward paid a 10 ms foot spike as much as a held
    tuck, which trained a jerky snap-and-drop. Paying per step of airborne
    clearance instead makes only *sustained* height through the flight
    window profitable; a momentary spike earns almost nothing. The
    curriculum ladder still gates on the max-based apex_height metric.

    v15: pays only while ascending (base vertical velocity above the cutoff,
    i.e. up through a short window past the ballistic apex). Paying on
    descent taught the policy to hold the foot tucked all the way down and
    crash-land with zero extension; the descent leg of the flight is now
    reward-neutral so the foot is free to extend for the landing.
    """
    term = _jump_term(env)
    robot: Entity = env.scene[ROBOT_ENTITY_NAME]
    clearance = torch.clamp(
        foot_height_w(env) - term.baseline_height,
        min=0.0,
        max=term.current_target_height,
    )
    jump_active = (term.command[:, 0] > 0.5).float()
    airborne = 1.0 - foot_ground_contact(env)[:, 0]
    ascending = (
        robot.data.root_link_lin_vel_w[:, 2] >= descent_velocity_cutoff_m_s
    ).float()
    return clearance * jump_active * airborne * ascending


def landing_impact_speed_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    term = _jump_term(env)
    return term.landing_event * torch.square(term.landing_impact_speed)


def landing_impact_accel_excess(
    env: "ManagerBasedRlEnv",
    accel_threshold_m_s2: float = 40.0,
) -> torch.Tensor:
    """Base deceleration above a comfort threshold during the landing window.

    Proxy for foot contact force (F ~ m * |dv|/dt), measured over the whole
    post-touchdown window instead of at the instant of first contact. Unlike
    the speed-at-touchdown penalty this cannot be gamed by extending the leg
    mid-descent to touch down early: only the actual violence of the stop
    pays. Orientation-agnostic -- landing posture is entirely free.
    """
    return torch.clamp(
        _jump_term(env).landing_impact_accel - accel_threshold_m_s2,
        min=0.0,
    )


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


def small_jump_recovery_success(
    env: "ManagerBasedRlEnv",
    target_schedule: tuple[tuple[int, float], ...] = SMALL_JUMP_HEIGHT_SCHEDULE,
    use_jump_curriculum: bool = False,
) -> torch.Tensor:
    term = _jump_term(env)
    target_height = (
        term.current_target_height
        if use_jump_curriculum
        else _small_jump_target_height(env, target_schedule)
    )
    reached_target = term.apex_height >= 0.8 * target_height
    return term.landing_recovery_success_event * reached_target.float()


def balance_linear_action_rate_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _balance_only(env, linear_action_rate_l2(env))


def _planar_velocity_term(env: "ManagerBasedRlEnv") -> PlanarVelocityCommand:
    term = env.command_manager.get_term(PLANAR_VELOCITY_COMMAND_NAME)
    if not isinstance(term, PlanarVelocityCommand):
        raise TypeError(
            f"Expected PlanarVelocityCommand, received {type(term).__name__}."
        )
    return term


def hop_averaged_velocity_tracking(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Track hop-averaged world velocity, ignoring intra-hop oscillation.

    The kernel width (sigma 0.2 m/s) keeps a usable gradient out to roughly
    half the command cap; precision near the target is the directional
    progress term's job to bootstrap and this kernel's job to finish.
    """
    term = _planar_velocity_term(env)
    velocity_error = term.average_planar_velocity - term.command
    error_l2 = torch.sum(torch.square(velocity_error), dim=1)
    return torch.exp(-error_l2 / 0.04)


def hop_averaged_directional_progress(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Pay linearly for hop-averaged velocity along the commanded direction.

    An exp tracking kernel is nearly flat far from the target, so a policy
    hopping in place against a command sees no gradient from it. This term
    pays from the first cm/s of drift in the right direction, capped at the
    commanded speed so overshooting doesn't pay, and charges the sideways
    component. With a zero command it reduces to -|average velocity|.
    """
    term = _planar_velocity_term(env)
    command_speed = torch.linalg.vector_norm(term.command, dim=1)
    direction = term.command / (command_speed.unsqueeze(1) + 1e-6)
    along = torch.sum(term.average_planar_velocity * direction, dim=1)
    perpendicular = torch.linalg.vector_norm(
        term.average_planar_velocity - along.unsqueeze(1) * direction,
        dim=1,
    )
    progress = torch.minimum(along, command_speed) - perpendicular
    return progress / PLANAR_VELOCITY_SCALE_M_S


def hop_averaged_overspeed(
    env: "ManagerBasedRlEnv",
    margin_m_s: float = 0.10,
) -> torch.Tensor:
    """Hop-averaged speed beyond commanded speed plus a margin.

    v12 cruised at 0.70 m/s against a 0.15 m/s cap nearly for free: the
    tracking kernel is flat that far out and the directional term's cap
    only stops overshoot from *paying*, it doesn't charge for it. This
    term makes raw speed above the command cost real reward, normalized
    like the other velocity terms.
    """
    term = _planar_velocity_term(env)
    average_speed = torch.linalg.vector_norm(
        term.average_planar_velocity,
        dim=1,
    )
    command_speed = torch.linalg.vector_norm(term.command, dim=1)
    excess = torch.clamp(
        average_speed - (command_speed + margin_m_s),
        min=0.0,
    )
    return excess / PLANAR_VELOCITY_SCALE_M_S


def hop_averaged_velocity_error_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    term = _planar_velocity_term(env)
    normalized_error = (
        term.average_planar_velocity - term.command
    ) / PLANAR_VELOCITY_SCALE_M_S
    return torch.clamp(
        torch.sum(torch.square(normalized_error), dim=1),
        max=4.0,
    )


def instantaneous_velocity_tracking(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Track instantaneous world-frame XY velocity at every control step."""
    term = _planar_velocity_term(env)
    velocity = term._robot.data.root_link_lin_vel_w[:, :2]
    error_l2 = torch.sum(torch.square(velocity - term.command), dim=1)
    return torch.exp(-error_l2 / 0.04)


def instantaneous_directional_progress(env: "ManagerBasedRlEnv") -> torch.Tensor:
    term = _planar_velocity_term(env)
    velocity = term._robot.data.root_link_lin_vel_w[:, :2]
    command_speed = torch.linalg.vector_norm(term.command, dim=1)
    direction = term.command / (command_speed.unsqueeze(1) + 1.0e-6)
    along = torch.sum(velocity * direction, dim=1)
    perpendicular = torch.linalg.vector_norm(
        velocity - along.unsqueeze(1) * direction,
        dim=1,
    )
    return (
        torch.minimum(along, command_speed) - perpendicular
    ) / PLANAR_VELOCITY_SCALE_M_S


def instantaneous_velocity_error_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    term = _planar_velocity_term(env)
    velocity = term._robot.data.root_link_lin_vel_w[:, :2]
    normalized_error = (
        velocity - term.command
    ) / PLANAR_VELOCITY_SCALE_M_S
    return torch.clamp(
        torch.sum(torch.square(normalized_error), dim=1),
        max=4.0,
    )


def instantaneous_overspeed(
    env: "ManagerBasedRlEnv",
    margin_m_s: float = 0.10,
) -> torch.Tensor:
    term = _planar_velocity_term(env)
    speed = torch.linalg.vector_norm(
        term._robot.data.root_link_lin_vel_w[:, :2],
        dim=1,
    )
    command_speed = torch.linalg.vector_norm(term.command, dim=1)
    return torch.clamp(
        speed - (command_speed + margin_m_s),
        min=0.0,
    ) / PLANAR_VELOCITY_SCALE_M_S


def _velocity_income_gate(
    env: "ManagerBasedRlEnv",
    gate_s: float = 2.0,
) -> torch.Tensor:
    """1.0 while a jump is active or ended within ``gate_s`` seconds (v14).

    Velocity rewards only pay near jump activity, so continuous hopping is
    the only way to keep the tracking income flowing; standing (or scooting)
    while drifting toward the command earns nothing.
    """
    jump = _jump_term(env)
    recently_jumped = (jump.time_since_jump_end <= gate_s).float()
    return torch.clamp(_jump_active(env) + recently_jumped, max=1.0)


def gated_hop_averaged_velocity_tracking(
    env: "ManagerBasedRlEnv",
    gate_s: float = 2.0,
) -> torch.Tensor:
    """Hop-averaged tracking kernel, paid only near jump activity (v14)."""
    return hop_averaged_velocity_tracking(env) * _velocity_income_gate(
        env,
        gate_s,
    )


def gated_hop_averaged_directional_progress(
    env: "ManagerBasedRlEnv",
    gate_s: float = 2.0,
) -> torch.Tensor:
    """Hop-averaged directional progress, paid only near jump activity."""
    return hop_averaged_directional_progress(env) * _velocity_income_gate(
        env,
        gate_s,
    )


def gated_instantaneous_velocity_tracking(
    env: "ManagerBasedRlEnv",
    gate_s: float = 2.0,
) -> torch.Tensor:
    """Instantaneous tracking kernel, paid only near jump activity (v14)."""
    return instantaneous_velocity_tracking(env) * _velocity_income_gate(
        env,
        gate_s,
    )


def takeoff_velocity_along_command(
    env: "ManagerBasedRlEnv",
    min_command_speed_m_s: float = 0.02,
) -> torch.Tensor:
    """Signed planar takeoff velocity along the command direction (v16).

    Pays once per hop at the instant the foot leaves the ground -- the
    moment of control authority. Demoted to bootstrap shaping in v18: the
    instant-payout aim signal helps before the critic learns to anticipate
    the touchdown displacement payout.
    """
    jump = _jump_term(env)
    planar = _planar_velocity_term(env)
    velocity = planar._robot.data.root_link_lin_vel_w[:, :2]
    command = planar.command
    command_speed = torch.linalg.vector_norm(command, dim=1)
    direction = command / command_speed.clamp_min(1.0e-6).unsqueeze(1)
    along = (velocity * direction).sum(dim=1)
    moving = (command_speed >= min_command_speed_m_s).float()
    return jump.takeoff_event * along * moving


def takeoff_velocity_perpendicular_l1(
    env: "ManagerBasedRlEnv",
    min_command_speed_m_s: float = 0.02,
) -> torch.Tensor:
    """Planar takeoff velocity perpendicular to the command, per takeoff."""
    jump = _jump_term(env)
    planar = _planar_velocity_term(env)
    velocity = planar._robot.data.root_link_lin_vel_w[:, :2]
    command = planar.command
    command_speed = torch.linalg.vector_norm(command, dim=1)
    direction = command / command_speed.clamp_min(1.0e-6).unsqueeze(1)
    perpendicular = (
        velocity[:, 1] * direction[:, 0] - velocity[:, 0] * direction[:, 1]
    )
    moving = (command_speed >= min_command_speed_m_s).float()
    return jump.takeoff_event * perpendicular.abs() * moving


def hop_displacement_along_command(
    env: "ManagerBasedRlEnv",
    min_command_speed_m_s: float = 0.02,
) -> torch.Tensor:
    """Signed takeoff-to-touchdown planar displacement along the command
    direction, paid once at touchdown.

    v18: prices the whole hop instead of the launch instant. A well-aimed
    takeoff that ends in a crash pays nothing here (touchdown on a fallen
    base still pays, but the fall penalty dwarfs it), so aiming the hop and
    landing it stop being competing objectives.
    """
    jump = _jump_term(env)
    planar = _planar_velocity_term(env)
    command = planar.command
    command_speed = torch.linalg.vector_norm(command, dim=1)
    direction = command / command_speed.clamp_min(1.0e-6).unsqueeze(1)
    along = (jump.hop_displacement_xy * direction).sum(dim=1)
    # v24 capped the payout at the commanded distance (speed x
    # touchdown-to-touchdown period) so overshoot couldn't out-earn
    # accuracy. v28 exposed the flaw: min(realized, commanded) pays
    # overshoot at the SAME rate as a perfect hop, so once the policy
    # learned to always over-jump, income saturated and became insensitive
    # to precision -- 40 frantic hops/episode at 2x the commanded speed
    # earned the full cap while the action std ratcheted 0.31 -> 1.21.
    # v29: tent-shaped payout peaked exactly at the commanded distance.
    # A perfect hop earns the commanded distance; overshoot bleeds at the
    # same rate as undershoot; hops against the command go negative. This
    # also closes the hop-frequency channel: once realized displacement
    # matches the command, extra hops are overshoot and lose money.
    commanded_distance = command_speed * jump.hop_period_s
    along = commanded_distance - (along - commanded_distance).abs()
    moving = (command_speed >= min_command_speed_m_s).float()
    return jump.landing_event * along * moving


def hop_displacement_perpendicular_l1(
    env: "ManagerBasedRlEnv",
    min_command_speed_m_s: float = 0.02,
) -> torch.Tensor:
    """Takeoff-to-touchdown planar displacement perpendicular to the
    command, per touchdown (aim-tightening cost for the v18 payout)."""
    jump = _jump_term(env)
    planar = _planar_velocity_term(env)
    command = planar.command
    command_speed = torch.linalg.vector_norm(command, dim=1)
    direction = command / command_speed.clamp_min(1.0e-6).unsqueeze(1)
    perpendicular = (
        jump.hop_displacement_xy[:, 1] * direction[:, 0]
        - jump.hop_displacement_xy[:, 0] * direction[:, 1]
    )
    moving = (command_speed >= min_command_speed_m_s).float()
    return jump.landing_event * perpendicular.abs() * moving


def model_trigger_bonus(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Flat bonus each time the policy's own jump request fires (v21).

    Forced cadence preempts any income advantage from self-triggering, so
    the marginal self-trigger is pure fall risk and cold-start policies
    prune the jump channel long before the handover anneal (v19/v20). This
    keeps the channel alive; the 0.25 s cooldown and grounded-only
    eligibility cap the rate at ~1 per hop cycle.
    """
    jump = _jump_term(env)
    return jump.model_trigger_event


def apex_overshoot(
    env: "ManagerBasedRlEnv",
    margin_m: float = 0.05,
) -> torch.Tensor:
    """Apex growth beyond the curriculum target plus a margin (v16).

    Charged as the per-step increase of the overshoot (mirror of the apex
    progress idiom), so each hop pays proportionally to how far past
    target + margin it flew. Keeps the takeoff-impulse income from being
    farmed with ever-taller vertical launches.
    """
    term = _jump_term(env)
    threshold = term.current_target_height + margin_m
    previous_apex = term.apex_height - term.apex_progress_delta
    overshoot_now = torch.clamp(term.apex_height - threshold, min=0.0)
    overshoot_prev = torch.clamp(previous_apex - threshold, min=0.0)
    return torch.clamp(overshoot_now - overshoot_prev, min=0.0)


def free_hop_clearance(
    env: "ManagerBasedRlEnv",
    max_clearance_m: float = 0.10,
) -> torch.Tensor:
    """Gently reward sustained foot clearance under a moving command."""
    term = _planar_velocity_term(env)
    moving = (
        torch.linalg.vector_norm(term.command, dim=1) > 0.01
    ).float()
    airborne = 1.0 - foot_ground_contact(env)[:, 0]
    clearance = torch.clamp(
        foot_height_w(env) - term.grounded_foot_height,
        min=0.0,
        max=max_clearance_m,
    )
    return clearance * airborne * moving


def free_hop_touchdown_impact_speed_l2(
    env: "ManagerBasedRlEnv",
) -> torch.Tensor:
    return torch.square(_planar_velocity_term(env).touchdown_impact_speed)


def free_hop_touchdown_angular_velocity_l2(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    return (
        base_angular_velocity_l2(env, asset_cfg)
        * _planar_velocity_term(env).touchdown_event
    )


def grounded_planar_speed(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    grounded_grace_s: float | None = None,
    no_hop_timeout_s: float | None = None,
) -> torch.Tensor:
    """Planar base speed while the foot is grounded.

    Penalizing this closes the scooting loophole: dragging along the floor
    stops paying, so hopping becomes the only way to satisfy the velocity
    command.

    All planar momentum change happens through ground contact, so taxing
    grounded motion wholesale taxes the only phase where velocity control
    can act (this suppressed velocity learning in v7 and again, milder, in
    v10). When ``no_hop_timeout_s`` is set the penalty only fires when no
    jump has ended within that window -- i.e. it punishes locomoting while
    refusing to hop, and carrying momentum between regular hops is free.
    ``grounded_grace_s`` is the older post-landing grace-window form.
    """
    asset: Entity = env.scene[asset_cfg.name]
    planar_speed = torch.linalg.vector_norm(
        asset.data.root_link_lin_vel_w[:, :2],
        dim=1,
    )
    penalty = planar_speed * foot_ground_contact(env)[:, 0]
    if no_hop_timeout_s is not None:
        jump = _jump_term(env)
        jump_inactive = 1.0 - _jump_active(env)
        refusing_to_hop = (jump.time_since_jump_end > no_hop_timeout_s).float()
        penalty = penalty * jump_inactive * refusing_to_hop
    elif grounded_grace_s is not None:
        jump = _jump_term(env)
        jump_inactive = 1.0 - _jump_active(env)
        past_grace = (jump.time_since_jump_end > grounded_grace_s).float()
        penalty = penalty * jump_inactive * past_grace
    return penalty


def stance_tilt_error_l2(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    moving_scale: float = 0.25,
    moving_speed_threshold_m_s: float = 0.02,
) -> torch.Tensor:
    """Tilt penalty relaxed while a nonzero velocity is commanded.

    Leaning into the command is how the robot aims push-offs and accelerates
    under gravity; taxing it at the balance-stage rate fights locomotion.
    With a ~zero command the full penalty applies so the robot still prefers
    standing upright.
    """
    tilt = tilt_error_l2(env, asset_cfg)
    command_speed = torch.linalg.vector_norm(
        env.command_manager.get_command(PLANAR_VELOCITY_COMMAND_NAME),
        dim=1,
    )
    scale = torch.where(
        command_speed > moving_speed_threshold_m_s,
        torch.full_like(tilt, moving_scale),
        torch.ones_like(tilt),
    )
    return tilt * scale


def fell_over(
    env: "ManagerBasedRlEnv",
    limit_angle_rad: float = _FALL_LIMIT_RAD,
) -> torch.Tensor:
    return envs_mdp.bad_orientation(
        env,
        limit_angle=limit_angle_rad,
        asset_cfg=_ROBOT_CFG,
    ).float()


def annealed_reward(
    env: "ManagerBasedRlEnv",
    term_func,
    anneal_start_step: int,
    anneal_end_step: int,
    **term_params,
) -> torch.Tensor:
    """Wrap a scaffold reward with a linear fade to zero (v29).

    Full strength through anneal_start_step, linear decay to zero at
    anneal_end_step, exactly zero afterward. Lets a cold start learn on
    the full shaping stack and then graduate onto the diet-2.0 economics
    (capped hop displacement as sole tracking income) without a
    checkpoint handoff.
    """
    step = int(env.common_step_counter)
    if step >= anneal_end_step:
        return torch.zeros(env.num_envs, device=env.device)
    value = term_func(env, **term_params)
    if step <= anneal_start_step:
        return value
    scale = 1.0 - (step - anneal_start_step) / float(
        anneal_end_step - anneal_start_step
    )
    return value * scale


def build_reward_terms(
    height_control: bool = False,
    jump_stage_one: bool = False,
    jump_stage_two: bool = False,
    robust_balance: bool = False,
    warm_start_jump: bool = False,
    navigation: bool = False,
    fall_angle_deg: float = FALL_ANGLE_DEG,
    relaxed_stance_tilt: bool = False,
    free_hop_velocity: bool = False,
    trigger_handover: bool = False,
    reward_diet: bool = False,
    scaffold_anneal_steps: tuple[int, int] | None = None,
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
        # Must stay aligned with the fell_over termination angle: if the
        # penalty fired at a lower angle than termination, every fall would
        # accrue -10k per step instead of once.
        "fall_event": RewardTermCfg(
            func=fell_over,
            weight=-10_000.0,
            params={"limit_angle_rad": math.radians(fall_angle_deg)},
        ),
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
        if warm_start_jump:
            terms["tilt_error"].weight = -8.0
            terms["base_angular_velocity"].weight = -0.2
            terms["action_rate"].weight = -0.005
            terms["off_ground"].weight = -0.5
            terms["jump_apex_progress"] = RewardTermCfg(
                func=capped_small_jump_apex_progress,
                weight=5_000.0,
            )
            terms["landing_recovery_success"] = RewardTermCfg(
                func=small_jump_recovery_success,
                weight=250.0,
            )
        if navigation:
            if free_hop_velocity:
                for name in (
                    "off_ground",
                    "height_tracking",
                    "jump_apex_progress",
                    "landing_recovery_success",
                    "landing_impact_speed",
                    "landing_angular_velocity",
                ):
                    terms.pop(name, None)
                terms["linear_velocity"].func = linear_velocity_l2
                terms["linear_velocity"].weight = -0.002
                terms["linear_action_rate"].func = linear_action_rate_l2
                terms["tilt_error"].weight = -8.0
                terms["base_angular_velocity"].weight = -0.2
                terms["action_rate"].weight = -0.005
                terms["free_hop_clearance"] = RewardTermCfg(
                    func=free_hop_clearance,
                    weight=75.0,
                )
                terms["landing_impact_speed"] = RewardTermCfg(
                    func=free_hop_touchdown_impact_speed_l2,
                    weight=-500.0,
                )
                terms["landing_angular_velocity"] = RewardTermCfg(
                    func=free_hop_touchdown_angular_velocity_l2,
                    weight=-250.0,
                    params={"asset_cfg": _ROBOT_CFG},
                )
                terms["planar_velocity_tracking"] = RewardTermCfg(
                    func=instantaneous_velocity_tracking,
                    weight=40.0,
                )
                terms["directional_progress"] = RewardTermCfg(
                    func=instantaneous_directional_progress,
                    weight=25.0,
                )
                terms["planar_velocity_error"] = RewardTermCfg(
                    func=instantaneous_velocity_error_l2,
                    weight=-2.0,
                )
                terms["planar_overspeed"] = RewardTermCfg(
                    func=instantaneous_overspeed,
                    weight=-15.0,
                    params={"margin_m_s": 0.10},
                )
                return terms
            # Impact violence measured as base deceleration over the landing
            # window, replacing the speed-at-first-contact penalty that paid
            # the policy to extend the leg mid-descent and touch down early.
            terms.pop("landing_impact_speed", None)
            terms["landing_impact_accel"] = RewardTermCfg(
                func=landing_impact_accel_excess,
                weight=-8.0,
                params={"accel_threshold_m_s2": 40.0},
            )
            # v13: integral clearance instead of max-progress. Payout parity:
            # the old term paid 5000 x min(apex, target) once per jump
            # (~0.15 units); this pays per airborne step, and a good hop
            # holds ~0.10 m clearance for ~0.35 s (~0.035 unit-seconds), so
            # 450 x 0.035 / dt-normalization lands near the old per-jump
            # total while a momentary spike earns ~nothing.
            terms["jump_apex_progress"] = RewardTermCfg(
                func=sustained_flight_clearance,
                weight=450.0,
            )
            terms["landing_recovery_success"] = RewardTermCfg(
                func=small_jump_recovery_success,
                weight=250.0,
                params={"use_jump_curriculum": True},
            )
            terms["tilt_error"].weight = -8.0
            if relaxed_stance_tilt:
                # Full -8 only at ~zero command; -2 effective while moving.
                terms["tilt_error"] = RewardTermCfg(
                    func=stance_tilt_error_l2,
                    weight=-8.0,
                    params={"asset_cfg": _ROBOT_CFG, "moving_scale": 0.25},
                )
            terms["base_angular_velocity"].weight = -0.2
            terms["action_rate"].weight = -0.005
            terms["off_ground"].weight = -0.5
            # v13 rebalance: velocity terms were ~2.5% of v12's episode
            # reward, so the policy optimized stability and ignored
            # commands. Boosted to first-order (~20-30% of a well-behaved
            # episode), plus a raw overspeed cost -- v12 cruised at 0.70
            # m/s against a 0.15 cap nearly free.
            terms["planar_velocity_tracking"] = RewardTermCfg(
                func=hop_averaged_velocity_tracking,
                weight=40.0,
            )
            terms["directional_progress"] = RewardTermCfg(
                func=hop_averaged_directional_progress,
                weight=25.0,
            )
            terms["planar_velocity_error"] = RewardTermCfg(
                func=hop_averaged_velocity_error_l2,
                weight=-2.0,
            )
            terms["planar_overspeed"] = RewardTermCfg(
                func=hop_averaged_overspeed,
                weight=-15.0,
                params={"margin_m_s": 0.10},
            )
            # Anti-scoot guard rail only: with forced jumps re-firing every
            # 1-2 s the timeout never elapses during normal hopping, so
            # ground-phase momentum (the only phase where velocity control
            # can act) is free. It bites solely if the robot locomotes while
            # refusing to hop for over 2 s.
            terms["grounded_planar_speed"] = RewardTermCfg(
                func=grounded_planar_speed,
                weight=-10.0,
                params={"asset_cfg": _ROBOT_CFG, "no_hop_timeout_s": 2.0},
            )
            if trigger_handover:
                # Direct-lineage stages (v14+): the policy owns jump timing,
                # so velocity income is gated to jump activity -- hopping is
                # the only way to keep the tracking meters running.
                terms["planar_velocity_tracking"] = RewardTermCfg(
                    func=gated_hop_averaged_velocity_tracking,
                    weight=28.0,
                )
                terms["directional_progress"] = RewardTermCfg(
                    func=gated_hop_averaged_directional_progress,
                    weight=25.0,
                )
                terms["instant_velocity_tracking"] = RewardTermCfg(
                    func=gated_instantaneous_velocity_tracking,
                    weight=12.0,
                )
                terms["planar_overspeed"] = RewardTermCfg(
                    func=hop_averaged_overspeed,
                    weight=-15.0,
                    params={"margin_m_s": 0.05},
                )
                # v15: impact violence priced by the actual post-touchdown
                # deceleration, tight threshold, expensive.
                terms["landing_impact_accel"] = RewardTermCfg(
                    func=landing_impact_accel_excess,
                    weight=-25.0,
                    params={"accel_threshold_m_s2": 25.0},
                )
                # v18 headline: pay for realized takeoff-to-touchdown
                # displacement along the command, at touchdown. A 0.10 m hop
                # along command pays 10000*0.10*0.01 = 10 -- about one dwell
                # income -- and a hop that ends in a crash never cashes in,
                # so aim and surviving the landing are one objective.
                terms["hop_displacement"] = RewardTermCfg(
                    func=hop_displacement_along_command,
                    weight=10000.0,
                )
                terms["hop_displacement_perpendicular"] = RewardTermCfg(
                    func=hop_displacement_perpendicular_l1,
                    weight=-3000.0,
                )
                # v21: +3 per self-trigger (300 x 0.01) -- decisively above
                # the early marginal fall risk of one extra hop, mild
                # relative to per-jump income once skilled.
                terms["model_trigger_bonus"] = RewardTermCfg(
                    func=model_trigger_bonus,
                    weight=300.0,
                )
                # v16 takeoff terms demoted to bootstrap shaping (v18): the
                # instant-payout aim signal helps before the critic learns
                # to anticipate the touchdown displacement payout above.
                terms["takeoff_impulse"] = RewardTermCfg(
                    func=takeoff_velocity_along_command,
                    weight=500.0,
                )
                terms["takeoff_perpendicular"] = RewardTermCfg(
                    func=takeoff_velocity_perpendicular_l1,
                    weight=-150.0,
                )
                terms["apex_overshoot"] = RewardTermCfg(
                    func=apex_overshoot,
                    weight=-3000.0,
                    params={"margin_m": 0.05},
                )
            if reward_diet:
                # v26: strip every income stream that pays per jump without
                # caring where the hop went. In v25b (noise clamped at 0.25)
                # these paid ~13/episode against ~18 of tracking-linked
                # income, so "energetic hops, mediocre aim" was a stable
                # optimum and error plateaued at ~0.19 -- far above the
                # promotion gate.
                #
                # v28 (diet 2.0): v27 proved the gated per-step velocity
                # terms are also hop-count-scaling income -- every extra
                # hop holds the income gate open longer, so trigger noise
                # kept ratcheting the std (0.25 -> 0.62 overnight) even
                # with the v26 diet. The capped hop displacement is the one
                # income stream bounded by the commanded distance itself,
                # so it becomes the sole tracking income. Accuracy is the
                # only way to earn; noise has nothing left to farm.
                for name in (
                    "jump_apex_progress",
                    "height_tracking",
                    "model_trigger_bonus",
                    "takeoff_impulse",
                    "takeoff_perpendicular",
                    "apex_overshoot",
                    "planar_velocity_tracking",
                    "directional_progress",
                    "instant_velocity_tracking",
                    "landing_recovery_success",
                ):
                    terms.pop(name, None)
                # Sole tracking income, doubled now that it carries the
                # whole incentive: a full-accuracy hop (cap = command speed
                # x hop period) pays ~2x the old v18 calibration.
                terms["hop_displacement"] = RewardTermCfg(
                    func=hop_displacement_along_command,
                    weight=20000.0,
                )
                # Noise tax with teeth: -0.005 made channel noise nearly
                # free while the std ratchet ran.
                terms["action_rate"].weight = -0.01
            if scaffold_anneal_steps is not None:
                # v29 annealed cold start: same endpoint economics as the
                # v28 diet, reached by fading the shaping stack instead of
                # deleting it. Scaffolds teach the skills (hop, aim, land),
                # then linearly lose their income so the policy is weaned
                # onto accuracy-only earnings within one run -- no
                # overfit-to-old-rewards checkpoint to inherit.
                start_step, end_step = scaffold_anneal_steps
                for name in (
                    "jump_apex_progress",
                    "landing_recovery_success",
                    "height_tracking",
                    "model_trigger_bonus",
                    "takeoff_impulse",
                    "takeoff_perpendicular",
                    "apex_overshoot",
                    "planar_velocity_tracking",
                    "directional_progress",
                    "instant_velocity_tracking",
                ):
                    term = terms.get(name)
                    if term is None:
                        continue
                    terms[name] = RewardTermCfg(
                        func=annealed_reward,
                        weight=term.weight,
                        params={
                            "term_func": term.func,
                            "anneal_start_step": start_step,
                            "anneal_end_step": end_step,
                            **(term.params or {}),
                        },
                    )
                # Diet-2.0 endpoint terms live at full strength from the
                # start: the displacement payout is capped by the commanded
                # distance, so it cannot out-shout the scaffolds early, and
                # the critic gets its full history to learn the payout.
                terms["hop_displacement"] = RewardTermCfg(
                    func=hop_displacement_along_command,
                    weight=20000.0,
                )
                terms["action_rate"].weight = -0.01
    if robust_balance:
        terms["tilt_error"].weight = -8.0
        terms["base_angular_velocity"].weight = -0.2
        terms["action_rate"].weight = -0.005
        terms["height_tracking"].weight = 0.5
        for name in (
            "off_ground",
            "jump_apex_progress",
            "landing_recovery_success",
            "landing_impact_speed",
            "landing_angular_velocity",
        ):
            terms.pop(name, None)
    return terms
