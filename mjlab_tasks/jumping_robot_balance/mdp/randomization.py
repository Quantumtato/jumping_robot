"""Domain randomization and reset events for robust balancing."""

from __future__ import annotations

import math

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg

from mjlab_tasks.jumping_robot_balance.robot_cfg import (
    DEFAULT_BASE_HEIGHT_M,
)


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
