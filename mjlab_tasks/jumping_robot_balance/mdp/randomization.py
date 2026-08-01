"""Domain randomization and reset events for robust balancing."""

from __future__ import annotations

import math

from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_tasks.jumping_robot_balance.robot_cfg import (
    BASE_BODY_NAME,
    DEFAULT_BASE_HEIGHT_M,
    ROBOT_ENTITY_NAME,
)

_BASE_BODY_CFG = SceneEntityCfg(ROBOT_ENTITY_NAME, body_names=(BASE_BODY_NAME,))


def build_randomization_events() -> dict[str, EventTermCfg]:
    return {
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (DEFAULT_BASE_HEIGHT_M, DEFAULT_BASE_HEIGHT_M),
                    "roll": (-math.radians(3.0), math.radians(3.0)),
                    "pitch": (-math.radians(3.0), math.radians(3.0)),
                    "yaw": (0.0, 0.0),
                },
                "velocity_range": {},
            },
        ),
    }


def build_robustness_events() -> dict[str, EventTermCfg]:
    return {
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (DEFAULT_BASE_HEIGHT_M, DEFAULT_BASE_HEIGHT_M),
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
