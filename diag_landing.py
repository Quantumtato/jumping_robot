"""Landing violence calibration: probe force data and measure decel profiles."""

import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

CHECKPOINT = (
    "logs/rsl_rl/jumping_robot_balance/"
    "2026-08-18_16-34-02_direct_continue_v11/model_400.pt"
)
NUM_ENVS = 8
STEPS = 600

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from mjlab_tasks.jumping_robot_balance.task_registry import (
    DIRECT_CONTINUE_TASK_ID,
    register_tasks,
)
from mjlab_tasks.jumping_robot_balance.mdp.contact import foot_ground_contact

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
jump_term = base_env.command_manager.get_term("jump")

# ---- probe available constraint/contact force fields ----
data = base_env.sim.data
print("PROBE sim.data fields:", [a for a in dir(data) if "efc" in a.lower() or "force" in a.lower()])
contact = data.contact
print("PROBE contact fields:", [a for a in dir(contact) if not a.startswith("__")])

def unpack(out):
    return out[0] if isinstance(out, tuple) else out

try:
    obs = unpack(env.get_observations())
except Exception:
    obs = unpack(env.reset())

prev_vel = robot.data.root_link_lin_vel_w.clone()
rows = []
DT = 0.01
with torch.no_grad():
    for step in range(STEPS):
        actions = policy(obs)
        out = env.step(actions)
        obs = unpack(out)
        dones = out[2] if isinstance(out, tuple) and len(out) > 2 else None
        vel = robot.data.root_link_lin_vel_w.clone()
        accel = torch.linalg.vector_norm(vel - prev_vel, dim=1) / DT
        prev_vel = vel
        rows.append(
            dict(
                accel=accel.cpu(),
                vz=vel[:, 2].cpu(),
                contact=(foot_ground_contact(base_env)[:, 0] > 0.5).cpu(),
                active=(jump_term.command[:, 0] > 0.5).cpu(),
                t_end=jump_term.time_since_jump_end.cpu().clone(),
                done=(dones.cpu() if dones is not None else torch.zeros(NUM_ENVS)),
            )
        )

accel = torch.stack([r["accel"] for r in rows])
contact_m = torch.stack([r["contact"] for r in rows])
active = torch.stack([r["active"] for r in rows])
t_end = torch.stack([r["t_end"] for r in rows])
done = torch.stack([r["done"] for r in rows]) > 0.5

# Classify steps
landing_win = (t_end < 0.3) & contact_m & ~active & ~done
pushoff = active & contact_m & ~done
stance = contact_m & ~active & (t_end >= 0.3) & ~done
flight = ~contact_m & ~done

def stats(mask, name):
    vals = accel[mask]
    if vals.numel() == 0:
        print(f"{name}: none")
        return
    q = torch.quantile(vals, torch.tensor([0.5, 0.9, 0.99]))
    print(
        f"{name}: n={vals.numel()} median={q[0]:.1f} p90={q[1]:.1f} "
        f"p99={q[2]:.1f} max={vals.max():.1f} m/s^2"
    )

stats(landing_win, "landing window (t<0.3s post-jump, contact)")
stats(pushoff, "push-off (jump active, contact)")
stats(stance, "stance (contact, t>=0.3s)")
stats(flight, "flight")

# Per-landing peak accel within the window
peaks = []
T, N = accel.shape
for n in range(N):
    t = 1
    while t < T:
        if landing_win[t, n] and not landing_win[t - 1, n]:
            end = t
            while end < T and landing_win[end, n]:
                end += 1
            peaks.append(float(accel[t:end, n].max()))
            t = end
        else:
            t += 1
peaks_t = torch.tensor(peaks)
if len(peaks) > 0:
    q = torch.quantile(peaks_t, torch.tensor([0.1, 0.5, 0.9]))
    print(
        f"per-landing peak accel: n={len(peaks)} p10={q[0]:.1f} "
        f"median={q[1]:.1f} p90={q[2]:.1f} max={peaks_t.max():.1f}"
    )
print("CALIB_DONE")
