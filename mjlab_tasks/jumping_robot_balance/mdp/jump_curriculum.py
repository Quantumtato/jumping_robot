"""Vertical-hop target curriculum."""

from __future__ import annotations

JUMP_HEIGHT_SCHEDULE: tuple[tuple[int, float], ...] = (
    (0, 0.02),
    (96_000, 0.05),
    (192_000, 0.10),
)


def scheduled_jump_height(
    common_step_counter: int,
    schedule: tuple[tuple[int, float], ...] = JUMP_HEIGHT_SCHEDULE,
) -> float:
    height = schedule[0][1]
    for start_step, scheduled_height in schedule:
        if common_step_counter < start_step:
            break
        height = scheduled_height
    return height
