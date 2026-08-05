"""Environment configuration for jumping robot balance in mjlab."""

from __future__ import annotations

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from mjlab_tasks.jumping_robot_balance.mdp import (
    build_action_terms,
    build_disturbance_commands,
    build_height_action_terms,
    build_height_commands,
    build_jump_stage_one_action_terms,
    build_observation_groups,
    build_randomization_events,
    build_reward_terms,
    build_termination_terms,
)
from mjlab_tasks.jumping_robot_balance.robot_cfg import (
    BASE_BODY_NAME,
    EPISODE_LENGTH_S,
    ROBOT_ENTITY_NAME,
    SIM_DECIMATION,
    SIM_TIMESTEP_S,
    make_robot_entity_cfg,
)


def _configure_scene_spec(spec: mujoco.MjSpec) -> None:
    spec.njmax = 128


def jumping_robot_balance_env_cfg(
    play: bool = False,
    height_control: bool = False,
    jump_stage_one: bool = False,
) -> ManagerBasedRlEnvCfg:
    height_control = height_control or jump_stage_one
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={ROBOT_ENTITY_NAME: make_robot_entity_cfg()},
            num_envs=1,
            env_spacing=2.5,
            spec_fn=_configure_scene_spec,
        ),
        observations=build_observation_groups(
            height_control=height_control,
            jump_stage_one=jump_stage_one,
        ),
        actions=(
            build_jump_stage_one_action_terms()
            if jump_stage_one
            else (
                build_height_action_terms(play=play)
                if height_control
                else build_action_terms()
            )
        ),
        events=build_randomization_events(),
        rewards=build_reward_terms(
            height_control=height_control,
            jump_stage_one=jump_stage_one,
        ),
        terminations=build_termination_terms(),
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name=ROBOT_ENTITY_NAME,
            body_name=BASE_BODY_NAME,
            distance=1.5,
            elevation=-20.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=SIM_TIMESTEP_S,
                iterations=10,
                ls_iterations=20,
            )
        ),
        decimation=SIM_DECIMATION,
        episode_length_s=EPISODE_LENGTH_S,
    )

    commands = build_height_commands(play=play) if height_control else {}
    if play:
        cfg.episode_length_s = 1e10
        cfg.observations["actor"].enable_corruption = False
        commands.update(build_disturbance_commands())
    cfg.commands = commands

    return cfg
