"""Calibrate the v15 landing-impact weight on the v14 crash-landing policy.

Measures, per jump: dwell income under the new ascent-gated clearance reward
(weight 450), and per landing: the integral of impact accel excess over the
25 m/s^2 threshold. Reports the weight that makes a tucked slam cost about
one jump's income.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

CHECKPOINT = (
    "logs/rsl_rl/jumping_robot_balance/"
    "2026-08-19_20-22-46_direct_continue_v14/model_900.pt"
)
NUM_ENVS = 16
STEPS = 900
THRESHOLD = 25.0
DWELL_WEIGHT = 450.0

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from mjlab_tasks.jumping_robot_balance.task_registry import (
    DIRECT_CONTINUE_TASK_ID,
    register_tasks,
)
from mjlab_tasks.jumping_robot_balance.mdp.contact import (
    foot_ground_contact,
    foot_height_w,
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
jump = base_env.command_manager.get_term("jump")
dt = base_env.step_dt


def unpack(out):
    return out[0] if isinstance(out, tuple) else out


try:
    obs = unpack(env.get_observations())
except Exception:
    obs = unpack(env.reset())

income_acc = torch.zeros(NUM_ENVS, device=device)
cost_acc = torch.zeros(NUM_ENVS, device=device)
in_jump = torch.zeros(NUM_ENVS, dtype=torch.bool, device=device)
jump_incomes, landing_costs, peak_accels = [], [], []
peak_acc = torch.zeros(NUM_ENVS, device=device)

with torch.no_grad():
    for _ in range(STEPS):
        actions = policy(obs)
        obs = unpack(env.step(actions))
        active = jump.command[:, 0] > 0.5
        airborne = foot_ground_contact(base_env)[:, 0] < 0.5
        target = jump.current_target_height
        clearance = torch.clamp(
            foot_height_w(base_env) - jump.baseline_height, min=0.0, max=target
        )
        budget = 1.15 * 2.0 * (2.0 * target / 9.81) ** 0.5
        ascending = robot.data.root_link_lin_vel_w[:, 2] >= -0.10
        pay = (
            active & airborne & ascending & (jump.airborne_time <= budget)
        ).float()
        income_acc += DWELL_WEIGHT * clearance * pay * dt

        excess = torch.clamp(jump.landing_impact_accel - THRESHOLD, min=0.0)
        cost_acc += excess * dt
        peak_acc = torch.maximum(peak_acc, jump.landing_impact_accel)

        # A jump cycle ends when active flips off; harvest the accumulators.
        ended = in_jump & ~active
        for i in ended.nonzero().flatten().tolist():
            jump_incomes.append(float(income_acc[i]))
            landing_costs.append(float(cost_acc[i]))
            peak_accels.append(float(peak_acc[i]))
            income_acc[i] = 0.0
            cost_acc[i] = 0.0
            peak_acc[i] = 0.0
        in_jump = active


def q(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * len(s)))]


n = len(jump_incomes)
print(f"jumps observed: {n}")
print(f"dwell income/jump (weight {DWELL_WEIGHT}, new gate): "
      f"p50={q(jump_incomes,0.5):.2f} p90={q(jump_incomes,0.9):.2f}")
print(f"impact excess integral/landing (thr {THRESHOLD}): "
      f"p50={q(landing_costs,0.5):.4f} p90={q(landing_costs,0.9):.4f}")
print(f"peak landing accel m/s2: p50={q(peak_accels,0.5):.1f} "
      f"p90={q(peak_accels,0.9):.1f}")
p50c = q(landing_costs, 0.5)
p90c = q(landing_costs, 0.9)
p50i = q(jump_incomes, 0.5)
if p90c > 0:
    print(f"weight for p90 slam to cost p50 income: {-p50i/p90c:.0f}")
if p50c > 0:
    print(f"weight for p50 landing to cost p50 income: {-p50i/p50c:.0f}")
print("DIAG_IMPACT_DONE")
