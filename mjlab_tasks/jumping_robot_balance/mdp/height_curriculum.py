"""Shared schedule for progressive linear-actuator training."""

from __future__ import annotations

HEIGHT_RANGE_SCHEDULE: tuple[tuple[int, float], ...] = (
    (0, 0.010),
    (32_000, 0.025),
    (64_000, 0.045),
    (96_000, 0.078),
)


def scheduled_height_half_width(
    step: int,
    schedule: tuple[tuple[int, float], ...] = HEIGHT_RANGE_SCHEDULE,
) -> float:
    half_width = schedule[0][1]
    for threshold, value in schedule:
        if step < threshold:
            break
        half_width = value
    return half_width
