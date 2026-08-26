"""Headless rollout of the free-hop checkpoint in TRAINING command mode.

Verifies whether the overnight policy shuffles and tracks when commands are
actually sampled (as in training), versus the play/viewer path where the
command starts at zero until the user pushes one via the UI.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

CHECKPOINT = (
    "logs/rsl_rl/jumping_robot_balance/"
    "2026-08-19_04-41-56_free_hop_velocity_from_robust_balance_v1/model_4999.pt"
)
NUM_ENVS = 8
STEPS = 400

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from mjlab_tasks.jumping_robot_balance.task_registry import (
    FREE_HOP_VELOCITY_TASK_ID,
    register_tasks,
)
from mjlab_tasks.jumping_robot_balance.mdp.contact import foot_ground_contact

register_tasks()
device = "cuda:0"

env_cfg = load_env_cfg(FREE_HOP_VELOCITY_TASK_ID, play=False)
env_cfg.scene.num_envs = NUM_ENVS
agent_cfg = load_rl_cfg(FREE_HOP_VELOCITY_TASK_ID)

base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
runner.load(CHECKPOINT, load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

robot = base_env.scene["robot"]
planar = base_env.command_manager.get_term("planar_velocity")

def unpack(out):
    return out[0] if isinstance(out, tuple) else out

try:
    obs = unpack(env.get_observations())
except Exception:
    obs = unpack(env.reset())

cmd_speeds, meas_speeds, errs = [], [], []
takeoffs, airborne_steps, total = 0, 0, 0
prev_contact = None
with torch.no_grad():
    for step in range(STEPS):
        actions = policy(obs)
        obs = unpack(env.step(actions))
        contact = foot_ground_contact(base_env)[:, 0] > 0.5
        if prev_contact is not None:
            takeoffs += int((prev_contact & ~contact).sum())
        prev_contact = contact
        airborne_steps += int((~contact).sum())
        total += NUM_ENVS
        cmd = planar.command
        vel = robot.data.root_link_lin_vel_w[:, :2]
        cmd_speeds.append(float(torch.linalg.vector_norm(cmd, dim=1).mean()))
        meas_speeds.append(float(torch.linalg.vector_norm(vel, dim=1).mean()))
        errs.append(float(torch.linalg.vector_norm(vel - cmd, dim=1).mean()))

n = len(cmd_speeds)
print(f"ROLLOUT {STEPS} steps x {NUM_ENVS} envs (training command mode)")
print(f"mean commanded speed: {sum(cmd_speeds)/n:.3f} m/s (nonzero => commands flow)")
print(f"mean measured speed:  {sum(meas_speeds)/n:.3f} m/s")
print(f"mean instant vel err: {sum(errs)/n:.3f} m/s")
print(f"takeoff rate: {takeoffs/total:.4f} per env-step")
print(f"airborne fraction: {airborne_steps/total:.3f}")
print("DIAG_DONE")
