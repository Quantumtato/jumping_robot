"""Leg actuator diagnostic: what limits jump height?

Loads the final v9b checkpoint in the direct-continue env (train mode so the
env force-triggers jumps), rolls out a few hundred control steps, and records
the linear actuator's force, tracking error, target motion, and base
kinematics through every push-off. Report-only; changes nothing.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

CHECKPOINT = (
    "logs/rsl_rl/jumping_robot_balance/"
    "2026-08-18_06-53-34_direct_continue_v10/model_5999.pt"
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

am = base_env.action_manager
try:
    lin_term = am.get_term("linear_position")
except Exception:
    lin_term = am._terms["linear_position"]  # noqa: SLF001
kp = float(lin_term.cfg.kp_n_m)
kd = float(lin_term.cfg.kd_n_s_m)
fmax = float(lin_term.cfg.max_force_n)
rate = float(lin_term.cfg.max_target_speed_m_s)
print(f"CONFIG kp={kp} kd={kd} max_force={fmax} target_rate_limit={rate} m/s")
print(
    "CONFIG jump_kp="
    f"{lin_term.cfg.jump_kp_n_m} jump_kd={lin_term.cfg.jump_kd_n_s_m}"
)

# Try to print the compiled MuJoCo actuator limits, if reachable.
for attr in ("mj_model", "model", "_mj_model"):
    m = getattr(base_env.sim, attr, None)
    if m is not None and hasattr(m, "actuator_forcerange"):
        print("MJ actuator_forcerange:", m.actuator_forcerange)
        print("MJ actuator_gainprm[:, :3]:", m.actuator_gainprm[:, :3])
        break

jump_term = base_env.command_manager.get_term("jump")

def unpack(out):
    return out[0] if isinstance(out, tuple) else out

obs = unpack(env.reset()) if hasattr(env, "reset") else None
try:
    obs = unpack(env.get_observations())
except Exception:
    pass

rows = []  # per step per env records
with torch.no_grad():
    for step in range(STEPS):
        actions = policy(obs)
        out = env.step(actions)
        obs = unpack(out)
        dones = out[2] if isinstance(out, tuple) and len(out) > 2 else None

        pos = robot.data.joint_pos[:, lin_id]
        vel = robot.data.joint_vel[:, lin_id]
        tgt = lin_term.position_target[:, 0]
        contact = foot_ground_contact(base_env)[:, 0] > 0.5
        active = jump_term.command[:, 0] > 0.5
        # Mirror apply_actions: jump-window gains while the command is active.
        if lin_term.cfg.jump_kp_n_m is not None:
            kp_eff = torch.where(
                active,
                torch.full_like(pos, lin_term.cfg.jump_kp_n_m),
                torch.full_like(pos, kp),
            )
            kd_eff = torch.where(
                active,
                torch.full_like(pos, lin_term.cfg.jump_kd_n_s_m),
                torch.full_like(pos, kd),
            )
        else:
            kp_eff = torch.full_like(pos, kp)
            kd_eff = torch.full_like(pos, kd)
        raw_pd = kp_eff * (tgt - pos) - kd_eff * vel
        force = torch.clamp(raw_pd, min=-fmax, max=fmax)
        kp_term = kp_eff * (tgt - pos)
        kd_term = -kd_eff * vel
        base_h = robot.data.root_link_pos_w[:, 2]
        base_vz = robot.data.root_link_lin_vel_w[:, 2]
        rows.append(
            dict(
                step=step,
                pos=pos.cpu(),
                vel=vel.cpu(),
                tgt=tgt.cpu(),
                force=force.cpu(),
                raw_pd=raw_pd.cpu(),
                kp_term=kp_term.cpu(),
                kd_term=kd_term.cpu(),
                contact=contact.cpu(),
                active=active.cpu(),
                base_h=base_h.cpu(),
                base_vz=base_vz.cpu(),
                done=(dones.cpu() if dones is not None else torch.zeros(NUM_ENVS)),
            )
        )

print(f"rollout done: {STEPS} steps x {NUM_ENVS} envs")

# ---- per-jump segmentation and analysis ----
import math

G = 9.81
DT = 0.01

def col(key):
    return torch.stack([r[key].float() for r in rows])  # (T, N)

pos = col("pos"); vel = col("vel"); tgt = col("tgt")
force = col("force"); raw_pd = col("raw_pd")
kp_t = col("kp_term"); kd_t = col("kd_term")
contact = col("contact") > 0.5; active = col("active") > 0.5
base_h = col("base_h"); base_vz = col("base_vz"); done = col("done") > 0.5

jumps = []
T = pos.shape[0]
for n in range(NUM_ENVS):
    t = 1
    while t < T:
        if active[t, n] and not active[t - 1, n]:
            start = t
            end = start
            while end < T and active[end, n] and not done[end, n]:
                end += 1
            if end >= T or done[min(end, T - 1), n]:
                t = end + 1
                continue
            seg = slice(start, end)
            seg_contact = contact[seg, n]
            # push-off: grounded steps from trigger until first airborne step
            airborne_idx = (~seg_contact).nonzero()
            if len(airborne_idx) == 0:
                t = end + 1
                continue
            takeoff_local = int(airborne_idx[0])
            if takeoff_local == 0:
                t = end + 1
                continue
            push = slice(start, start + takeoff_local)
            takeoff_t = start + takeoff_local
            # apex: max base height while airborne within window (+20 steps)
            look_end = min(T, end + 20)
            apex = float(base_h[takeoff_t:look_end, n].max() - base_h[start, n])
            v_takeoff = float(base_vz[takeoff_t, n])
            jumps.append(
                dict(
                    env=n,
                    start=start,
                    push_steps=takeoff_local,
                    peak_force=float(force[push, n].abs().max()),
                    mean_force=float(force[push, n].abs().mean()),
                    sat_frac=float((force[push, n].abs() >= 0.99 * fmax).float().mean()),
                    raw_pd_peak=float(raw_pd[push, n].abs().max()),
                    max_err=float((tgt[push, n] - pos[push, n]).abs().max()),
                    mean_err=float((tgt[push, n] - pos[push, n]).abs().mean()),
                    peak_ext_speed=float(vel[push, n].abs().max()),
                    kp_term_peak=float(kp_t[push, n].abs().max()),
                    kd_term_peak=float(kd_t[push, n].abs().max()),
                    tgt_travel=float((tgt[takeoff_t - 1, n] - tgt[start, n]).abs()),
                    pos_travel=float((pos[takeoff_t - 1, n] - pos[start, n]).abs()),
                    v_takeoff=v_takeoff,
                    apex_theory=v_takeoff * v_takeoff / (2 * G),
                    apex_meas=apex,
                )
            )
            t = end + 1
        else:
            t += 1

print(f"jumps analyzed: {len(jumps)}")
if jumps:
    hdr = (
        "env start pushN peakF meanF satFr rawPD maxErr meanErr extV "
        "kpPk kdPk tgtTrav posTrav vTake apexTh apexMeas"
    )
    print(hdr)
    for j in jumps:
        print(
            f"{j['env']} {j['start']} {j['push_steps']} {j['peak_force']:.1f} "
            f"{j['mean_force']:.1f} {j['sat_frac']:.2f} {j['raw_pd_peak']:.1f} "
            f"{j['max_err']:.4f} {j['mean_err']:.4f} {j['peak_ext_speed']:.2f} "
            f"{j['kp_term_peak']:.1f} {j['kd_term_peak']:.1f} "
            f"{j['tgt_travel']:.4f} {j['pos_travel']:.4f} {j['v_takeoff']:.2f} "
            f"{j['apex_theory']:.3f} {j['apex_meas']:.3f}"
        )

    def mean(k):
        return sum(j[k] for j in jumps) / len(jumps)

    print("--- SUMMARY ---")
    print(f"jumps: {len(jumps)}")
    print(f"mean push-off duration: {mean('push_steps') * DT * 1000:.0f} ms")
    print(f"mean peak |force|: {mean('peak_force'):.1f} N (clamp {fmax} N)")
    print(f"mean saturation fraction during push-off: {mean('sat_frac'):.2f}")
    print(f"mean unclamped PD demand peak: {mean('raw_pd_peak'):.1f} N")
    print(f"mean max tracking error: {mean('max_err') * 1000:.1f} mm")
    print(f"mean peak extension speed: {mean('peak_ext_speed'):.2f} m/s")
    print(f"mean peak kp term: {mean('kp_term_peak'):.1f} N")
    print(f"mean peak kd term: {mean('kd_term_peak'):.1f} N")
    print(f"mean target travel before takeoff: {mean('tgt_travel') * 1000:.1f} mm")
    print(f"mean pos travel before takeoff: {mean('pos_travel') * 1000:.1f} mm")
    print(f"mean takeoff vz: {mean('v_takeoff'):.2f} m/s")
    print(f"mean theoretical apex (v^2/2g): {mean('apex_theory') * 100:.1f} cm")
    print(f"mean measured apex: {mean('apex_meas') * 100:.1f} cm")
    need_v = math.sqrt(2 * G * 0.15)
    print(f"takeoff speed needed for 15 cm apex: {need_v:.2f} m/s")
    print(
        "extension speed at which kd term alone equals the clamp: "
        f"{fmax / kd:.2f} m/s (kd={kd})"
    )
print("DIAG_DONE")
