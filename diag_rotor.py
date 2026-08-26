"""Rotor coupling verification: reflected inertia, yaw kick, foot torsion.

Loads the v12 final checkpoint in the direct-continue env with the new
legrotor body + ballscrew equality constraint and condim=4 foot. Verifies:
  A. the equality constraint holds (rotor vel = 157.08 x slide vel),
  B. leg tracking still works during push-off under reflected inertia,
  C. rotor acceleration imparts base yaw with the expected sign/magnitude
     in flight (momentum exchange),
  D. the grounded foot's torsional friction absorbs much of the reaction
     torque (grounded yaw response << flight yaw response).
Report-only; changes nothing.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

CHECKPOINT = (
    "logs/rsl_rl/jumping_robot_balance/"
    "2026-08-18_18-19-29_direct_continue_v12/model_5999.pt"
)
NUM_ENVS = 8
STEPS = 600  # 6 s at 100 Hz

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from mjlab_tasks.jumping_robot_balance.task_registry import (
    DIRECT_CONTINUE_TASK_ID,
    register_tasks,
)
from mjlab_tasks.jumping_robot_balance.mdp.contact import foot_ground_contact
from mjlab_tasks.jumping_robot_balance.robot_cfg import (
    LEG_ROTOR_INERTIA_KG_M2,
    SCREW_COUPLING_RAD_PER_M,
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
lin_ids, _ = robot.find_joints(("linear",))
lin_id = int(lin_ids[0])
rot_ids, rot_names = robot.find_joints(("legrotor",))
print(f"MODEL rotor joint found: {rot_names}")
rot_id = int(rot_ids[0])

for attr in ("mj_model", "model", "_mj_model"):
    m = getattr(base_env.sim, attr, None)
    if m is not None and hasattr(m, "nv"):
        print(f"MODEL nv={m.nv} neq={m.neq}")
        break

jump_term = base_env.command_manager.get_term("jump")

def unpack(out):
    return out[0] if isinstance(out, tuple) else out

obs = None
try:
    obs = unpack(env.get_observations())
except Exception:
    obs = unpack(env.reset())

rows = []
with torch.no_grad():
    for step in range(STEPS):
        actions = policy(obs)
        out = env.step(actions)
        obs = unpack(out)
        dones = out[2] if isinstance(out, tuple) and len(out) > 2 else None

        lin_pos = robot.data.joint_pos[:, lin_id]
        lin_vel = robot.data.joint_vel[:, lin_id]
        rot_vel = robot.data.joint_vel[:, rot_id]
        contact = foot_ground_contact(base_env)[:, 0] > 0.5
        active = jump_term.command[:, 0] > 0.5
        # Base yaw rate about the base body's z (the leg/rotor axis).
        try:
            yaw_rate = robot.data.root_link_ang_vel_b[:, 2]
        except AttributeError:
            yaw_rate = robot.data.root_link_ang_vel_w[:, 2]
        base_h = robot.data.root_link_pos_w[:, 2]
        base_vz = robot.data.root_link_lin_vel_w[:, 2]
        rows.append(
            dict(
                lin_pos=lin_pos.cpu(),
                lin_vel=lin_vel.cpu(),
                rot_vel=rot_vel.cpu(),
                contact=contact.cpu(),
                active=active.cpu(),
                yaw=yaw_rate.cpu(),
                base_h=base_h.cpu(),
                base_vz=base_vz.cpu(),
                done=(dones.cpu() if dones is not None else torch.zeros(NUM_ENVS)),
            )
        )

print(f"rollout done: {STEPS} steps x {NUM_ENVS} envs")

DT = 0.01

def col(key):
    return torch.stack([r[key].float() for r in rows])

lin_pos = col("lin_pos"); lin_vel = col("lin_vel"); rot_vel = col("rot_vel")
contact = col("contact") > 0.5; active = col("active") > 0.5
yaw = col("yaw"); base_h = col("base_h"); base_vz = col("base_vz")
done = col("done") > 0.5

# --- A. constraint ratio ---
mask = lin_vel.abs() > 0.05
if mask.any():
    ratios = (rot_vel[mask] / lin_vel[mask])
    print(
        f"A. rotor/slide velocity ratio: median={ratios.median():.2f} "
        f"expected +/-{SCREW_COUPLING_RAD_PER_M:.2f} "
        f"(p10={ratios.quantile(0.1):.2f} p90={ratios.quantile(0.9):.2f})"
    )
else:
    print("A. no slide motion above threshold?!")

# --- C/D. yaw response to rotor acceleration, flight vs grounded ---
rot_acc = (rot_vel[1:] - rot_vel[:-1]) / DT
yaw_acc = (yaw[1:] - yaw[:-1]) / DT
valid = ~done[1:] & ~done[:-1]
big = rot_acc.abs() > 500.0  # rad/s^2, only meaningful accelerations

def slope(mask):
    x = rot_acc[mask]
    y = yaw_acc[mask]
    if len(x) < 20:
        return float("nan"), 0
    return float((x * y).sum() / (x * x).sum()), len(x)

flight = valid & big & ~contact[1:] 
ground = valid & big & contact[1:]
s_f, n_f = slope(flight)
s_g, n_g = slope(ground)
print(f"C. FLIGHT yaw_acc/rotor_acc slope: {s_f:.5f} (n={n_f})")
print(f"   expected ~ -J_rotor/Izz_base = -{LEG_ROTOR_INERTIA_KG_M2:.3e}/~1.4e-3 = ~-0.013")
print(f"D. GROUNDED yaw_acc/rotor_acc slope: {s_g:.5f} (n={n_g})")
if s_g == s_g and s_f == s_f and abs(s_f) > 0:
    print(f"   grounded/flight response ratio: {abs(s_g) / abs(s_f):.2f} (want << 1)")

# sign check: extending leg (lin_vel > 0 growing => rot_acc > 0) should
# counter-rotate the base (yaw_acc < 0) in flight.
ext = flight & (rot_acc > 500.0)
if ext.sum() > 10:
    frac_counter = float((yaw_acc[ext] < 0).float().mean())
    print(
        f"   sign check (flight, rotor spinning up): yaw_acc<0 in "
        f"{100 * frac_counter:.0f}% of samples (expect >50%)"
    )

# flight yaw kick magnitude during hard pushes
if flight.any():
    print(
        f"   flight peak |yaw_acc| {yaw_acc[flight].abs().max():.1f} rad/s^2, "
        f"peak |rotor_acc| {rot_acc[flight].abs().max():.0f} rad/s^2"
    )

# --- B. leg tracking / jump quality quick stats ---
push = active & contact
if push.any():
    print(
        f"B. push-off: peak ext speed {lin_vel[push].abs().max():.2f} m/s, "
        f"peak rotor speed {rot_vel[push].abs().max():.0f} rad/s"
    )
takeoffs = (~contact[1:]) & contact[:-1] & active[1:]
if takeoffs.any():
    v = base_vz[1:][takeoffs]
    print(
        f"   takeoff vz: median {v.median():.2f} m/s, max {v.max():.2f} "
        f"(theoretical apex median {v.median() ** 2 / (2 * 9.81) * 100:.1f} cm)"
    )
print(f"   fall/reset count in rollout: {int(done.sum())}")
print("DIAG_DONE")
