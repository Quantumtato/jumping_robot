"""Simulated foot-contact signal shared by observations and rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab_tasks.jumping_robot_balance.robot_cfg import ROBOT_ENTITY_NAME

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_TERRAIN_GEOM_NAME = "terrain"
_FOOT_BODY_PATH = f"{ROBOT_ENTITY_NAME}/foot"
_CONTACT_IDS_CACHE = "_jumping_robot_contact_ids"


def _contact_ids(
    env: "ManagerBasedRlEnv",
) -> tuple[int, int]:
    cached = getattr(env, _CONTACT_IDS_CACHE, None)
    if cached is not None:
        return cached

    model = env.sim.mj_model
    terrain_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        _TERRAIN_GEOM_NAME,
    )
    foot_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        _FOOT_BODY_PATH,
    )
    if terrain_geom_id < 0 or foot_body_id < 0:
        raise ValueError(
            "Could not resolve the terrain geometry and robot foot body."
        )

    if not torch.any(env.sim.model.geom_bodyid == foot_body_id):
        raise ValueError("The robot foot body has no collision geometries.")

    cached = (terrain_geom_id, foot_body_id)
    setattr(env, _CONTACT_IDS_CACHE, cached)
    return cached


def foot_ground_contact(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Return one when any foot geometry contacts the terrain."""
    terrain_geom_id, foot_body_id = _contact_ids(env)
    contacts = env.sim.data.contact
    geom = contacts.geom.to(dtype=torch.long)
    valid = contacts.dim > 0
    geom_body_ids = env.sim.model.geom_bodyid
    foot_first = geom_body_ids[geom[:, 0]] == foot_body_id
    foot_second = geom_body_ids[geom[:, 1]] == foot_body_id
    ground_contact = valid & (
        ((geom[:, 0] == terrain_geom_id) & foot_second)
        | ((geom[:, 1] == terrain_geom_id) & foot_first)
    )

    result = torch.zeros(
        (env.num_envs, 1),
        dtype=torch.float32,
        device=env.device,
    )
    world_ids = contacts.worldid[ground_contact].to(dtype=torch.long)
    result[world_ids, 0] = 1.0
    return result
