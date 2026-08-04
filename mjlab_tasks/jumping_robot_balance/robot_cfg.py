"""Robot asset and physical limits for the jumping robot balance task."""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_XML_PATH = REPO_ROOT / "onshape_mjcf" / "robot.xml"

ROBOT_ENTITY_NAME = "robot"
BASE_BODY_NAME = "baserotor"
FLYWHEEL_X_JOINT = "flywheelX"
FLYWHEEL_Y_JOINT = "flywheelY"
LINEAR_JOINT = "linear"

MAX_FLYWHEEL_TORQUE_NM = 0.33
MAX_FLYWHEEL_SPEED_RAD_S = 1500.0
LINEAR_MAX_FORCE_N = 120.0
LINEAR_MAX_SPEED_M_S = 60.0
LINEAR_MAX_STROKE_M = 0.152
# Approximately 5 Hz and critically damped for the 1.6 kg physical robot.
LINEAR_POSITION_KP_N_M = 1600.0
LINEAR_POSITION_KD_N_S_M = 100.0
LINEAR_BALANCE_FEEDFORWARD_LIMIT_N = 30.0
FALL_ANGLE_DEG = 30.0

LINEAR_RANGE_MIN_M = -0.1458580691518949
LINEAR_RANGE_MAX_M = 0.010141930848105107
LINEAR_RANGE_CENTER_M = 0.5 * (LINEAR_RANGE_MIN_M + LINEAR_RANGE_MAX_M)
LINEAR_RANGE_HALF_WIDTH_M = 0.5 * (LINEAR_RANGE_MAX_M - LINEAR_RANGE_MIN_M)

CONTROL_FREQUENCY_HZ = 1000
SIM_TIMESTEP_S = 0.0005
SIM_DECIMATION = 2

EPISODE_LENGTH_S = 20.0
# At the nominal linear position, this leaves the foot about 7 mm above the floor.
DEFAULT_BASE_HEIGHT_M = -0.05


def get_robot_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(ROBOT_XML_PATH))


ROBOT_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        XmlActuatorCfg(
            target_names_expr=(FLYWHEEL_X_JOINT, FLYWHEEL_Y_JOINT),
            command_field="effort",
        ),
        XmlActuatorCfg(
            target_names_expr=(LINEAR_JOINT,),
            command_field="effort",
        ),
    ),
)


ROBOT_INITIAL_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, DEFAULT_BASE_HEIGHT_M),
    joint_pos={
        FLYWHEEL_X_JOINT: 0.0,
        FLYWHEEL_Y_JOINT: 0.0,
        LINEAR_JOINT: LINEAR_RANGE_CENTER_M,
    },
    joint_vel={".*": 0.0},
)


def make_robot_entity_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_robot_spec,
        articulation=ROBOT_ARTICULATION,
        init_state=ROBOT_INITIAL_STATE,
    )
