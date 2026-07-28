"""Robustness evaluation scaffold for the jumping robot balance policy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RobustnessSweep:
    name: str
    description: str
    value_range: tuple[float, float]
    steps: int


def _default_sweeps() -> list[RobustnessSweep]:
    return [
        RobustnessSweep(
            name="linear_init_position",
            description="Initial linear joint position sweep (meters).",
            value_range=(-0.1458580691518949, 0.010141930848105107),
            steps=9,
        ),
        RobustnessSweep(
            name="com_x_offset",
            description="Center-of-mass X offset sweep (meters).",
            value_range=(-0.006, 0.006),
            steps=7,
        ),
        RobustnessSweep(
            name="push_xy_velocity",
            description="Disturbance push velocity sweep (m/s).",
            value_range=(0.0, 0.5),
            steps=6,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        default="jumping_robot_balance_eval_plan.json",
        help="Path to save the evaluation sweep plan.",
    )
    args = parser.parse_args()

    output_path = Path(args.output_json)
    output_path.write_text(
        json.dumps([asdict(sweep) for sweep in _default_sweeps()], indent=2),
        encoding="utf-8",
    )
    print(f"Wrote evaluation sweep plan to: {output_path}")


if __name__ == "__main__":
    main()
