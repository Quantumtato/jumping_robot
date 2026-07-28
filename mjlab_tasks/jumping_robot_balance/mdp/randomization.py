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
    LINEAR_JOINT,
    LINEAR_RANGE_CENTER_M,
    LINEAR_RANGE_MAX_M,
    LINEAR_RANGE_MIN_M,
    ROBOT_ENTITY_NAME,
)

_ROBOT_CFG = SceneEntityCfg(ROBOT_ENTITY_NAME)
_LINEAR_CFG = SceneEntityCfg(ROBOT_ENTITY_NAME, joint_names=(LINEAR_JOINT,))
_BASE_BODY_CFG = SceneEntityCfg(ROBOT_ENTITY_NAME, body_names=(BASE_BODY_NAME,))


def build_randomization_events() -> dict[str, EventTermCfg]:
    linear_offset_min = LINEAR_RANGE_MIN_M - LINEAR_RANGE_CENTER_M
    linear_offset_max = LINEAR_RANGE_MAX_M - LINEAR_RANGE_CENTER_M

    return {
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.02, 0.02),
                    "y": (-0.02, 0.02),
                    "z": (DEFAULT_BASE_HEIGHT_M - 0.03, DEFAULT_BASE_HEIGHT_M + 0.03),
                    "roll": (-math.radians(5.0), math.radians(5.0)),
                    "pitch": (-math.radians(5.0), math.radians(5.0)),
                    "yaw": (-math.pi, math.pi),
                },
                "velocity_range": {},
            },
        ),
        "reset_linear_position": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (linear_offset_min, linear_offset_max),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": _LINEAR_CFG,
            },
        ),
        "push_disturbance": EventTermCfg(
            func=envs_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "velocity_range": {
                    "x": (-0.4, 0.4),
                    "y": (-0.4, 0.4),
                    "z": (-0.2, 0.2),
                    "roll": (-0.3, 0.3),
                    "pitch": (-0.3, 0.3),
                    "yaw": (-0.5, 0.5),
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
                    0: (-0.004, 0.004),
                    1: (-0.004, 0.004),
                    2: (-0.006, 0.006),
                },
            },
        ),
        "base_mass_scale": EventTermCfg(
            mode="startup",
            func=dr.body_mass,
            params={
                "asset_cfg": _BASE_BODY_CFG,
                "operation": "scale",
                "ranges": (0.95, 1.05),
            },
        ),
    }
