"""Print the width of each flight_phase_history component."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from mjlab_tasks.jumping_robot_balance.task_registry import (
    DIRECT_CONTINUE_TASK_ID,
    register_tasks,
)
from mjlab_tasks.jumping_robot_balance.mdp import observations as O
from mjlab_tasks.jumping_robot_balance.mdp.contact import foot_ground_contact

register_tasks()
env_cfg = load_env_cfg(DIRECT_CONTINUE_TASK_ID, play=False)
env_cfg.scene.num_envs = 4
env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode=None)
env.reset()

print("lin_pos", O._normalized_linear_position(env).shape)
print(
    "lin_vel",
    O._normalized_joint_vel(env, 60.0, O._LINEAR_CFG).shape,
)
print("last_action", O._motor_last_action(env).shape)
print("contact", foot_ground_contact(env).shape)
print("jump_cmd", env.command_manager.get_command("jump").shape)
print("full_state", O._flight_phase_history_state(env).shape)
robot = env.scene["robot"]
print("joint_names", robot.joint_names)
print("PROBE_DONE")
