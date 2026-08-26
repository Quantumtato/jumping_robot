"""Verify the PLAY-path command wiring end-to-end for the free-hop viewer.

Injects a velocity command through the same queue the viser sliders use,
then checks (a) the command reaches the term and survives episode resets,
and (b) whether the policy's base actually moves in response.
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

env_cfg = load_env_cfg(FREE_HOP_VELOCITY_TASK_ID, play=True)
env_cfg.scene.num_envs = 1
agent_cfg = load_rl_cfg(FREE_HOP_VELOCITY_TASK_ID)

base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
runner.load(CHECKPOINT, load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

robot = base_env.scene["robot"]
planar = base_env.command_manager.get_term("planar_velocity")
print(f"cfg.play={planar.cfg.play} apply_only_when_grounded="
      f"{planar.cfg.apply_only_when_grounded}")

def unpack(out):
    return out[0] if isinstance(out, tuple) else out

try:
    obs = unpack(env.get_observations())
except Exception:
    obs = unpack(env.reset())


def run_block(steps, label):
    start_pos = robot.data.root_link_pos_w[0, :2].clone()
    speeds, cmd_norms, grounded_n = [], [], 0
    resets = 0
    with torch.no_grad():
        for _ in range(steps):
            global obs
            actions = policy(obs)
            obs = unpack(env.step(actions))
            if base_env.episode_length_buf[0] == 0:
                resets += 1
            speeds.append(float(torch.linalg.vector_norm(
                robot.data.root_link_lin_vel_w[0, :2])))
            cmd_norms.append(float(torch.linalg.vector_norm(planar.command[0])))
            grounded_n += int(foot_ground_contact(base_env)[0, 0] > 0.5)
    disp = float(torch.linalg.vector_norm(
        robot.data.root_link_pos_w[0, :2] - start_pos))
    print(f"[{label}] steps={steps} resets={resets} "
          f"cmd_norm mean={sum(cmd_norms)/len(cmd_norms):.3f} "
          f"final={cmd_norms[-1]:.3f} | "
          f"speed mean={sum(speeds)/len(speeds):.3f} m/s | "
          f"net displacement={disp:.3f} m | "
          f"grounded frac={grounded_n/steps:.2f}")
    return disp


d0 = run_block(150, "zero command")
print("injecting knob command vx=0.15 via GUI queue ...")
planar._pending.put((0, 0.15, 0.0))
d1 = run_block(600, "cmd 0.15 m/s +X")

print("VERDICT:")
print(f"  command stuck at {float(planar.command[0,0]):.3f} m/s "
      f"(should be 0.150 across resets)")
print(f"  displacement zero-cmd {d0:.3f} m vs commanded {d1:.3f} m over 6 s")
print("DIAG_PLAY_DONE")
