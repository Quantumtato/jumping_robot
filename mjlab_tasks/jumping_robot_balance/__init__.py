"""Balance task package for the jumping robot."""

from mjlab_tasks.jumping_robot_balance.task_registry import TASK_ID, register_tasks

register_tasks()

__all__ = ["TASK_ID", "register_tasks"]
