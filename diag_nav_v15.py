"""Navigation forensics on the v15 final checkpoint.

Holds fixed velocity commands (+x, +y, -x, zero) across envs, logs per-hop
displacement vectors, stance lean, flywheel usage, and yaw kicks, and checks
the heading-frame observations for sign/frame bugs.
"""

import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

CHECKPOINT = (
    "logs/rsl_rl/jumping_robot_balance/"
    "2026-08-19_21-42-59_direct_continue_v15/model_3300.pt"
)
NUM_ENVS = 16
BLOCK_STEPS = 700
CMD_SPEED = 0.12
BLOCKS = {
    "+x": (CMD_SPEED, 0.0),
    "+y": (0.0, CMD_SPEED),
    "-x": (-CMD_SPEED, 0.0),
    "zero": (0.0, 0.0),
}

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat

from mjlab_tasks.jumping_robot_balance.task_registry import (
    DIRECT_CONTINUE_TASK_ID,
    register_tasks,
)
from mjlab_tasks.jumping_robot_balance.mdp.contact import foot_ground_contact
from mjlab_tasks.jumping_robot_balance.robot_cfg import (
    FLYWHEEL_X_JOINT,
    FLYWHEEL_Y_JOINT,
    LINEAR_JOINT,
    MAX_FLYWHEEL_SPEED_RAD_S,
)

register_tasks()
device = "cuda:0"

env_cfg = load_env_cfg(DIRECT_CONTINUE_TASK_ID, play=False)
env_cfg.scene.num_envs = NUM_ENVS
agent_cfg = load_rl_cfg(DIRECT_CONTINUE_TASK_ID)
base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
runner.load(CHECKPOINT, load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

robot = base_env.scene["robot"]
planar = base_env.command_manager.get_term("planar_velocity")
jump = base_env.command_manager.get_term("jump")
dt = base_env.step_dt

joint_names = list(robot.joint_names)
fly_x_id = joint_names.index(FLYWHEEL_X_JOINT)
fly_y_id = joint_names.index(FLYWHEEL_Y_JOINT)
leg_id = joint_names.index(LINEAR_JOINT)

fixed_cmd = torch.zeros(NUM_ENVS, 2, device=device)
orig_resample = planar._resample_command


def pinned_resample(env_ids):
    orig_resample(env_ids)  # keeps state sanitization
    planar._command[env_ids] = fixed_cmd[env_ids]


planar._resample_command = pinned_resample


def yaw_of(quat_wxyz):
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def unpack(out):
    return out[0] if isinstance(out, tuple) else out


try:
    obs = unpack(env.get_observations())
except Exception:
    obs = unpack(env.reset())

up_body = torch.tensor([0.0, 0.0, 1.0], device=device).expand(NUM_ENVS, 3)

# Observation frame sanity accumulators
obs_cmd_cos, obs_vel_cos = [], []

results = {}
for name, cmd in BLOCKS.items():
    fixed_cmd[:] = torch.tensor(cmd, device=device)
    planar._command[:] = fixed_cmd
    log = {k: [] for k in ("contact", "pos", "yaw", "tilt", "fly", "legv",
                            "reset", "active")}
    with torch.no_grad():
        for step in range(BLOCK_STEPS):
            actions = policy(obs)
            obs = unpack(env.step(actions))
            planar._command[:] = fixed_cmd  # belt and suspenders
            contact = foot_ground_contact(base_env)[:, 0] > 0.5
            quat = robot.data.root_link_quat_w
            up_w = quat_apply(quat, up_body)
            log["contact"].append(contact.cpu().numpy().copy())
            log["pos"].append(robot.data.root_link_pos_w[:, :2].cpu().numpy().copy())
            log["yaw"].append(yaw_of(quat).cpu().numpy().copy())
            log["tilt"].append(up_w[:, :2].cpu().numpy().copy())
            log["fly"].append(
                robot.data.joint_vel[:, [fly_x_id, fly_y_id]].cpu().numpy().copy())
            log["legv"].append(robot.data.joint_vel[:, leg_id].cpu().numpy().copy())
            log["reset"].append(
                (base_env.episode_length_buf == 0).cpu().numpy().copy())
            log["active"].append((jump.command[:, 0] > 0.5).cpu().numpy().copy())

            # Observation frame check (noiseless, direct computation)
            if step % 25 == 0 and (cmd[0] != 0 or cmd[1] != 0):
                hq = yaw_quat(quat)
                cmd3 = torch.zeros(NUM_ENVS, 3, device=device)
                cmd3[:, :2] = planar.command
                obs_cmd = quat_apply_inverse(hq, cmd3)[:, :2]
                yaw = yaw_of(quat)
                c, s = torch.cos(-yaw), torch.sin(-yaw)
                manual = torch.stack(
                    (c * planar.command[:, 0] - s * planar.command[:, 1],
                     s * planar.command[:, 0] + c * planar.command[:, 1]),
                    dim=1)
                cs = torch.nn.functional.cosine_similarity(obs_cmd, manual, dim=1)
                obs_cmd_cos.extend(cs.cpu().tolist())
                vel3 = torch.zeros(NUM_ENVS, 3, device=device)
                vel3[:, :2] = planar.average_planar_velocity
                est = quat_apply_inverse(hq, vel3)[:, :2]
                manual_v = torch.stack(
                    (c * vel3[:, 0] - s * vel3[:, 1],
                     s * vel3[:, 0] + c * vel3[:, 1]),
                    dim=1)
                mask = torch.linalg.vector_norm(manual_v, dim=1) > 0.02
                if mask.any():
                    cs_v = torch.nn.functional.cosine_similarity(
                        est[mask], manual_v[mask], dim=1)
                    obs_vel_cos.extend(cs_v.cpu().tolist())

    # ---- per-hop extraction ----
    arr = {k: np.stack(v) for k, v in log.items()}  # (T, N, ...)
    hops = []
    for e in range(NUM_ENVS):
        contact = arr["contact"][:, e]
        reset = arr["reset"][:, e]
        t = 1
        while t < BLOCK_STEPS:
            if contact[t - 1] and not contact[t]:  # takeoff
                to = t
                td = None
                for u in range(to + 1, BLOCK_STEPS):
                    if reset[u]:
                        break
                    if contact[u]:
                        td = u
                        break
                if td is not None and td - to >= 3:  # ignore micro-bounces
                    disp = arr["pos"][td, e] - arr["pos"][to, e]
                    lean = arr["tilt"][max(0, to - 3):to, e].mean(axis=0)
                    fly = np.abs(arr["fly"][to:td, e])
                    sat = float(
                        (fly > 0.95 * MAX_FLYWHEEL_SPEED_RAD_S).mean())
                    # yaw kick during push-off (5 steps pre-takeoff)
                    yaw_kick = float(
                        arr["yaw"][to, e] - arr["yaw"][max(0, to - 5), e])
                    hops.append(dict(disp=disp, lean=lean, sat=sat,
                                     dur=(td - to) * dt, yaw_kick=yaw_kick))
                    t = td
            t += 1
    results[name] = hops

# ---- reporting ----
print(f"\nOBS SANITY: cmd-obs cos-sim mean={np.mean(obs_cmd_cos):.4f} "
      f"min={np.min(obs_cmd_cos):.4f} (n={len(obs_cmd_cos)}) | "
      f"vel-est cos-sim mean={np.mean(obs_vel_cos):.4f} "
      f"min={np.min(obs_vel_cos):.4f} (n={len(obs_vel_cos)})")

for name, hops in results.items():
    if not hops:
        print(f"\nBLOCK {name}: no hops recorded")
        continue
    disp = np.stack([h["disp"] for h in hops])
    mean_disp = disp.mean(axis=0)
    mag = np.linalg.norm(disp, axis=1)
    dur = np.array([h["dur"] for h in hops])
    sat = np.array([h["sat"] for h in hops])
    lean = np.stack([h["lean"] for h in hops])
    print(f"\nBLOCK {name} cmd={BLOCKS[name]}: hops={len(hops)}")
    print(f"  mean hop disp: ({mean_disp[0]:+.4f}, {mean_disp[1]:+.4f}) m | "
          f"mean |disp|={mag.mean():.4f} m | coherence "
          f"|mean|/mean| |={np.linalg.norm(mean_disp)/max(mag.mean(),1e-9):.2f}")
    print(f"  flight dur: {dur.mean():.3f}s | fly saturation frac: "
          f"{sat.mean():.2f}")
    if BLOCKS[name] != (0.0, 0.0):
        cdir = np.array(BLOCKS[name]) / np.linalg.norm(BLOCKS[name])
        along = disp @ cdir
        perp = disp @ np.array([-cdir[1], cdir[0]])
        ang = np.degrees(np.arctan2(perp, along))
        print(f"  along-cmd/hop: mean={along.mean():+.4f} m "
              f"std={along.std():.4f} | perp: mean={perp.mean():+.4f} "
              f"std={perp.std():.4f}")
        print(f"  aim angle err deg: mean={ang.mean():+.1f} "
              f"std={ang.std():.1f} | frac |err|<45deg: "
              f"{(np.abs(ang) < 45).mean():.2f}")
        lean_along = lean @ cdir
        print(f"  stance lean along cmd: mean={lean_along.mean():+.4f} "
              f"(positive=leaning toward command)")
        # correlation: does lean direction predict hop direction?
        lean_ang = np.arctan2(lean[:, 1], lean[:, 0])
        disp_ang = np.arctan2(disp[:, 1], disp[:, 0])
        d = np.degrees(np.arctan2(np.sin(disp_ang - lean_ang),
                                  np.cos(disp_ang - lean_ang)))
        print(f"  hop-dir minus lean-dir deg: mean={d.mean():+.1f} "
              f"std={d.std():.1f}")
        # achieved speed vs commanded over the block
        rate = len(hops) / (BLOCK_STEPS * dt * NUM_ENVS)
        v_ach = along.mean() * rate * NUM_ENVS * (BLOCK_STEPS * dt) / (
            BLOCK_STEPS * dt) / NUM_ENVS * len(hops) / len(hops)
        v_mean = along.sum() / (BLOCK_STEPS * dt * NUM_ENVS)
        print(f"  achieved along-cmd speed: {v_mean:.4f} m/s vs commanded "
              f"{np.linalg.norm(BLOCKS[name]):.2f} | hop rate "
              f"{len(hops)/(BLOCK_STEPS*dt*NUM_ENVS):.2f} /s/env")

print("\nPHYSICAL FLOOR: with per-hop along std s and ~1 hop per EMA window,")
print("velocity error floor ~ sqrt((cmd - v_ach)^2 + (s/T_hop)^2-ish)")
print("DIAG_NAV_DONE")
