"""
Web-based hardware state viewer and motor control panel.

The serial/CAN protocol intentionally reuses motor_control_gui.py so both UIs
stay compatible with the same firmware.
"""

from __future__ import annotations

import argparse
import copy
import heapq
import math
from pathlib import Path
import threading
import time

import mujoco
from mjviser import ViserMujocoScene
import numpy as np
import serial
import serial.tools.list_ports
import viser

from motor_control_gui import (
    BAUD,
    CTRL_ID,
    FLYWHEEL_X_CMD_ID,
    FLYWHEEL_X_FB_ID,
    FLYWHEEL_Y_CMD_ID,
    FLYWHEEL_Y_FB_ID,
    IO_HZ,
    KD_MAX,
    KD_MIN,
    KP_MAX,
    KP_MIN,
    LQR_T_MAX,
    LQR_T_MIN,
    MOTOR_ID,
    NODES,
    P_MAX,
    P_MIN,
    SEND_HZ,
    SENSOR_ACCEL_FB_ID,
    SENSOR_FB_X_ID,
    SENSOR_FB_Y_ID,
    T_MAX,
    T_MIN,
    V_MAX,
    V_MIN,
    decode_faults,
    decode_sensor_faults,
    make_init_frame,
    pack_command,
    pack_ctrl_frame,
    pack_sensor_led_frame,
    uint_to_float,
)


MOTOR_IDS = (0x01, 0x02, 0x03)
SENSOR_ID = 0x04
SENSOR_YAW_FB_ID = 0x107
FEEDBACK_REQUEST = 0x04
NODE_LABELS = tuple(NODES.keys())
NODE_BY_LABEL = {label: value for label, value in NODES.items()}
DEFAULT_COMMAND = {"p": 0.0, "v": 0.0, "kp": 0.0, "kd": 0.0, "t": 0.0}


def available_ports() -> list[str]:
    return [port.device for port in serial.tools.list_ports.comports()]


def preferred_port() -> str | None:
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if "CH340" in (port.description or ""):
            return port.device
    return ports[0].device if ports else None


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )


def body_quaternion(roll: float, pitch: float, yaw: float = 0.0) -> np.ndarray:
    qx = np.array([math.cos(roll / 2.0), math.sin(roll / 2.0), 0.0, 0.0])
    qy = np.array([math.cos(pitch / 2.0), 0.0, math.sin(pitch / 2.0), 0.0])
    qz = np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])
    quat = quaternion_multiply(qz, quaternion_multiply(qy, qx))
    return quat / np.linalg.norm(quat)


def quaternion_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


class HardwareBridge:
    """Thread-safe Waveshare serial/CAN transport shared by the Viser UI."""

    def __init__(self) -> None:
        self.serial_port: serial.Serial | None = None
        self.running = False
        self.io_error = ""
        self.rx_count = 0
        self.tx_count = 0

        self.commands = {
            motor_id: copy.copy(DEFAULT_COMMAND) for motor_id in MOTOR_IDS
        }
        self.motor_output_enabled = False
        self.sensor_output_enabled = False
        self.actuator_poll_enabled = True
        self.sensor_led = {"r": 0, "g": 0, "b": 0}

        self.feedback = {
            node_id: {
                "position": 0.0,
                "velocity": 0.0,
                "torque": 0.0,
                "state": 0,
                "error": 0,
                "seen": False,
                "count": 0,
                "last_update": 0.0,
            }
            for node_id in (*MOTOR_IDS, SENSOR_ID)
        }
        self.axes = {
            axis: {
                "angle": 0.0,
                "wrapped_angle": 0.0,
                "rate": 0.0,
                "speed": 0.0,
                "command": 0.0,
                "accel_angle": 0.0,
                "accel_wrapped_angle": 0.0,
                "accel_seen": False,
                "accel_valid": False,
                "gyro_rejected": False,
                "resync_count": None,
                "sensor_seen": False,
                "motor_seen": False,
            }
            for axis in ("x", "y", "z")
        }

        self._lock = threading.RLock()
        self._rx_buffer = bytearray()
        self._control_queue: list[tuple[float, bytes]] = []
        self._thread: threading.Thread | None = None

    def connect(self, port: str) -> None:
        if self.running:
            return
        serial_port = serial.Serial(port, BAUD, timeout=0.01)
        try:
            serial_port.write(make_init_frame())
            time.sleep(0.2)
        except (serial.SerialException, OSError):
            serial_port.close()
            raise

        with self._lock:
            self.serial_port = serial_port
            self._rx_buffer.clear()
            self.io_error = ""
            self.running = True
        self._thread = threading.Thread(target=self._io_loop, daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        with self._lock:
            was_enabled = self.motor_output_enabled
            self.motor_output_enabled = False
            self.sensor_output_enabled = False
            self.running = False
            serial_port = self.serial_port

        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=0.5)
        self._thread = None

        if serial_port is not None and serial_port.is_open:
            try:
                if was_enabled:
                    for motor_id in MOTOR_IDS:
                        serial_port.write(pack_command(0, 0, 0, 0, 0, motor_id))
                    time.sleep(0.05)
                serial_port.close()
            except (serial.SerialException, OSError) as exc:
                with self._lock:
                    self.io_error = str(exc)

        with self._lock:
            self.serial_port = None
            self._control_queue.clear()

    def set_command(self, motor_id: int, field: str, value: float) -> None:
        if motor_id not in self.commands:
            raise ValueError(f"Node 0x{motor_id:02X} is not a motor")
        if field not in DEFAULT_COMMAND:
            raise ValueError(f"Unknown command field: {field}")
        with self._lock:
            self.commands[motor_id][field] = float(value)

    def zero_all(self) -> None:
        with self._lock:
            for command in self.commands.values():
                command.update(DEFAULT_COMMAND)

    def zero_torque(self) -> None:
        with self._lock:
            for command in self.commands.values():
                command["t"] = 0.0

    def set_led(self, rgb: tuple[int, int, int]) -> None:
        with self._lock:
            self.sensor_led = {
                "r": max(0, min(255, int(rgb[0]))),
                "g": max(0, min(255, int(rgb[1]))),
                "b": max(0, min(255, int(rgb[2]))),
            }

    def queue_control(self, motor_id: int, command: int) -> None:
        if motor_id not in MOTOR_IDS:
            raise ValueError(f"Node 0x{motor_id:02X} is not a motor")
        if not self.running:
            raise RuntimeError("Connect to the robot before sending motor controls")
        frame = bytes(pack_ctrl_frame(command, motor_id + 0x200))
        repeats = 3 if command == 0x03 else 2
        now = time.monotonic()
        with self._lock:
            for index in range(repeats):
                heapq.heappush(
                    self._control_queue, (now + index * 0.02, frame)
                )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "io_error": self.io_error,
                "rx_count": self.rx_count,
                "tx_count": self.tx_count,
                "commands": copy.deepcopy(self.commands),
                "feedback": copy.deepcopy(self.feedback),
                "axes": copy.deepcopy(self.axes),
            }

    def _parse_axis_telemetry(self, can_id: int, data: bytes) -> None:
        if can_id == SENSOR_ACCEL_FB_ID:
            values = {
                "x": (uint_to_float((data[2] << 8) | data[3], P_MIN, P_MAX, 16), data[6]),
                "y": (uint_to_float((data[4] << 8) | data[5], P_MIN, P_MAX, 16), data[7]),
            }
            accel_valid = bool(data[1] & 0x01)
            gyro_rejected = bool(data[1] & 0x02)
            for axis, (wrapped, resync_count) in values.items():
                state = self.axes[axis]
                wrapped = wrap_to_pi(wrapped)
                state["accel_angle"] = wrapped
                state["accel_wrapped_angle"] = wrapped
                state["accel_seen"] = True
                state["accel_valid"] = accel_valid
                state["gyro_rejected"] = gyro_rejected
                state["resync_count"] = resync_count
            return

        axis: str | None = None
        updates: dict[str, float | bool] = {}
        if can_id == SENSOR_FB_X_ID:
            axis = "x"
            updates = {
                "wrapped_angle": uint_to_float(
                    (data[2] << 8) | data[3], P_MIN, P_MAX, 16
                ),
                "rate": uint_to_float(
                    (data[4] << 8) | data[5], V_MIN, V_MAX, 16
                ),
                "sensor_seen": True,
            }
        elif can_id == SENSOR_FB_Y_ID:
            axis = "y"
            updates = {
                "wrapped_angle": uint_to_float(
                    (data[2] << 8) | data[3], P_MIN, P_MAX, 16
                ),
                "rate": uint_to_float(
                    (data[4] << 8) | data[5], V_MIN, V_MAX, 16
                ),
                "sensor_seen": True,
            }
        elif can_id == SENSOR_YAW_FB_ID:
            axis = "z"
            updates = {
                "wrapped_angle": uint_to_float(
                    (data[2] << 8) | data[3], P_MIN, P_MAX, 16
                ),
                "rate": uint_to_float(
                    (data[4] << 8) | data[5], V_MIN, V_MAX, 16
                ),
                "sensor_seen": True,
            }
        elif can_id in (FLYWHEEL_X_FB_ID, FLYWHEEL_Y_FB_ID):
            axis = "x" if can_id == FLYWHEEL_X_FB_ID else "y"
            updates = {
                "speed": uint_to_float(
                    (data[4] << 8) | data[5], V_MIN, V_MAX, 16
                ),
                "motor_seen": True,
            }
        elif can_id in (FLYWHEEL_X_CMD_ID, FLYWHEEL_Y_CMD_ID):
            axis = "x" if can_id == FLYWHEEL_X_CMD_ID else "y"
            torque_raw = ((data[6] & 0x0F) << 8) | data[7]
            updates = {
                "command": uint_to_float(
                    torque_raw, LQR_T_MIN, LQR_T_MAX, 12
                )
            }

        if axis is not None:
            state = self.axes[axis]
            if "wrapped_angle" in updates:
                wrapped = wrap_to_pi(float(updates["wrapped_angle"]))
                updates["wrapped_angle"] = wrapped
                updates["angle"] = wrapped
            state.update(updates)

    def _parse_frame(self, can_id: int, data: bytes) -> None:
        with self._lock:
            self._parse_axis_telemetry(can_id, data)
            if 0x101 <= can_id <= 0x104:
                node_id = can_id - 0x100
                feedback = self.feedback[node_id]
                feedback.update(
                    {
                        "position": uint_to_float(
                            (data[2] << 8) | data[3], P_MIN, P_MAX, 16
                        ),
                        "velocity": uint_to_float(
                            (data[4] << 8) | data[5], V_MIN, V_MAX, 16
                        ),
                        "torque": (
                            float((data[6] << 8) | data[7])
                            if node_id == SENSOR_ID
                            else uint_to_float(
                                (data[6] << 8) | data[7], T_MIN, T_MAX, 16
                            )
                        ),
                        "state": (data[0] >> 4) & 0x0F,
                        "error": data[1],
                        "seen": True,
                        "count": feedback["count"] + 1,
                        "last_update": time.monotonic(),
                    }
                )
                self.rx_count += 1

    def _io_loop(self) -> None:
        io_interval = 1.0 / IO_HZ
        send_interval = 1.0 / SEND_HZ
        next_send = time.monotonic()

        while self.running:
            started = time.monotonic()
            try:
                with self._lock:
                    serial_port = self.serial_port
                    due_frames: list[bytes] = []
                    while (
                        self._control_queue
                        and self._control_queue[0][0] <= started
                    ):
                        _, frame = heapq.heappop(self._control_queue)
                        due_frames.append(frame)
                    motor_enabled = self.motor_output_enabled
                    sensor_enabled = self.sensor_output_enabled
                    poll_enabled = self.actuator_poll_enabled
                    commands = copy.deepcopy(self.commands)
                    led = copy.copy(self.sensor_led)

                if serial_port is None:
                    raise RuntimeError("Serial port disappeared while I/O was running")

                for frame in due_frames:
                    serial_port.write(frame)
                    with self._lock:
                        self.tx_count += 1

                if started >= next_send:
                    next_send = started + send_interval
                    if poll_enabled:
                        for motor_id in MOTOR_IDS:
                            serial_port.write(
                                pack_ctrl_frame(
                                    FEEDBACK_REQUEST, motor_id + 0x200
                                )
                            )
                            with self._lock:
                                self.tx_count += 1
                    if motor_enabled:
                        for motor_id, command in commands.items():
                            serial_port.write(
                                pack_command(
                                    command["p"],
                                    command["v"],
                                    command["kp"],
                                    command["kd"],
                                    command["t"],
                                    motor_id,
                                )
                            )
                            with self._lock:
                                self.tx_count += 1
                    if sensor_enabled:
                        serial_port.write(
                            pack_sensor_led_frame(
                                led["r"], led["g"], led["b"], SENSOR_ID + 0x200
                            )
                        )
                        with self._lock:
                            self.tx_count += 1

                waiting = serial_port.in_waiting
                if waiting:
                    self._rx_buffer.extend(serial_port.read(waiting))

                while len(self._rx_buffer) >= 13:
                    if (
                        self._rx_buffer[0] == 0xAA
                        and self._rx_buffer[12] == 0x55
                    ):
                        frame = bytes(self._rx_buffer[:13])
                        del self._rx_buffer[:13]
                        can_id = frame[2] | (frame[3] << 8)
                        self._parse_frame(can_id, frame[4:12])
                    else:
                        del self._rx_buffer[0]
            except (serial.SerialException, OSError, RuntimeError) as exc:
                with self._lock:
                    self.io_error = str(exc)
                    self.running = False
                break

            remaining = io_interval - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)


def joint_qpos_address(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise RuntimeError(f"MJCF is missing joint '{name}'")
    return int(model.jnt_qposadr[joint_id])


class HardwareViserApp:
    def __init__(
        self,
        model_path: Path,
        server_port: int,
        initial_port: str | None,
        auto_connect: bool,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.bridge = HardwareBridge()
        self.server = viser.ViserServer(port=server_port, label="Jumping Robot")
        self.scene = ViserMujocoScene(self.server, self.model, num_envs=1)

        self.flywheel_x_qpos = joint_qpos_address(self.model, "flywheelX")
        self.flywheel_y_qpos = joint_qpos_address(self.model, "flywheelY")
        self.linear_qpos = joint_qpos_address(self.model, "linear")
        self.base_qpos = joint_qpos_address(self.model, "baserotor_freejoint")
        linear_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "linear"
        )
        self.linear_range = np.array(self.model.jnt_range[linear_joint_id])

        self.selected_motor_id = MOTOR_ID
        self._syncing_controls = False
        self.command_handles: dict[str, viser.GuiInputHandle] = {}
        self.torque_handles: dict[int, viser.GuiInputHandle] = {}
        self.telemetry_handles: dict[int, viser.GuiInputHandle] = {}
        self._build_gui(initial_port)

        self.data.qpos[self.base_qpos : self.base_qpos + 7] = [
            0.0,
            0.0,
            0.45 - 0.234258,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        mujoco.mj_forward(self.model, self.data)
        self.scene.update_from_mjdata(self.data)
        if auto_connect and initial_port:
            self._connect_read_only(initial_port)

    def _connect_read_only(self, port: str) -> None:
        try:
            self.bridge.connect(port)
        except (serial.SerialException, OSError) as exc:
            self.connection_status.value = f"Connection failed: {exc}"
            return
        self.connection_status.value = f"Connected read-only: {port}"

    def _build_gui(self, initial_port: str | None) -> None:
        tabs = self.scene.create_visualization_gui()
        with tabs.add_tab("Hardware", icon=viser.Icon.GAUGE):
            with self.server.gui.add_folder("Connection"):
                ports = available_ports()
                if initial_port and initial_port not in ports:
                    ports.insert(0, initial_port)
                if not ports:
                    ports = ["No serial ports"]
                port_dropdown = self.server.gui.add_dropdown(
                    "Serial port",
                    options=ports,
                    initial_value=initial_port or ports[0],
                )
                refresh_button = self.server.gui.add_button("Refresh ports")
                connect_button = self.server.gui.add_button(
                    "Connect (read-only)", color="green"
                )
                disconnect_button = self.server.gui.add_button(
                    "Disconnect", color="red"
                )
                self.connection_status = self.server.gui.add_text(
                    "Status", "Disconnected", disabled=True
                )
                actuator_poll = self.server.gui.add_checkbox(
                    "Poll actuator feedback", True
                )

                @actuator_poll.on_update
                def _actuator_poll_changed(_event) -> None:
                    with self.bridge._lock:
                        self.bridge.actuator_poll_enabled = actuator_poll.value

                @refresh_button.on_click
                def _refresh_ports(_event) -> None:
                    ports_now = available_ports()
                    port_dropdown.options = ports_now or ["No serial ports"]
                    if ports_now and port_dropdown.value not in ports_now:
                        port_dropdown.value = ports_now[0]

                @connect_button.on_click
                def _connect(_event) -> None:
                    if port_dropdown.value == "No serial ports":
                        self.connection_status.value = "No serial port available"
                        return
                    self._connect_read_only(port_dropdown.value)

                @disconnect_button.on_click
                def _disconnect(_event) -> None:
                    self.bridge.disconnect()
                    motor_output.value = False
                    sensor_output.value = False
                    self.connection_status.value = "Disconnected"

            with self.server.gui.add_folder("Command safety"):
                motor_output = self.server.gui.add_checkbox(
                    "Enable motor command streaming", False
                )
                sensor_output = self.server.gui.add_checkbox(
                    "Enable sensor LED streaming", False
                )
                zero_torque = self.server.gui.add_button(
                    "ZERO ALL TORQUES", color="red"
                )
                zero_all = self.server.gui.add_button(
                    "ZERO ALL COMMAND FIELDS", color="red"
                )

                @motor_output.on_update
                def _motor_output_changed(_event) -> None:
                    if motor_output.value and not self.bridge.running:
                        motor_output.value = False
                        self.connection_status.value = (
                            "Connect before enabling motor output"
                        )
                        return
                    with self.bridge._lock:
                        self.bridge.motor_output_enabled = motor_output.value

                @sensor_output.on_update
                def _sensor_output_changed(_event) -> None:
                    if sensor_output.value and not self.bridge.running:
                        sensor_output.value = False
                        self.connection_status.value = (
                            "Connect before enabling sensor output"
                        )
                        return
                    with self.bridge._lock:
                        self.bridge.sensor_output_enabled = sensor_output.value

                @zero_torque.on_click
                def _zero_torque(_event) -> None:
                    self.bridge.zero_torque()
                    for handle in self.torque_handles.values():
                        handle.value = 0.0
                    if "t" in self.command_handles:
                        self.command_handles["t"].value = 0.0

                @zero_all.on_click
                def _zero_all(_event) -> None:
                    self.bridge.zero_all()
                    self._load_selected_command_into_gui()

            with self.server.gui.add_folder("Actuator torque commands"):
                for motor_id, label in (
                    (0x01, "Flywheel X torque (Nm)"),
                    (0x02, "Flywheel Y torque (Nm)"),
                    (0x03, "Spring actuator torque (Nm)"),
                ):
                    handle = self.server.gui.add_slider(
                        label,
                        min=T_MIN,
                        max=T_MAX,
                        step=0.001,
                        initial_value=0.0,
                    )
                    self.torque_handles[motor_id] = handle

                    @handle.on_update
                    def _torque_changed(_event, mid=motor_id, gui=handle) -> None:
                        if self._syncing_controls:
                            return
                        self.bridge.set_command(mid, "t", float(gui.value))
                        if mid == self.selected_motor_id:
                            self._syncing_controls = True
                            try:
                                self.command_handles["t"].value = gui.value
                            finally:
                                self._syncing_controls = False

            with self.server.gui.add_folder("Selected motor MIT command"):
                motor_labels = [
                    label
                    for label, node in NODE_BY_LABEL.items()
                    if node["mode"] == "motor"
                ]
                selected_motor = self.server.gui.add_dropdown(
                    "Motor", options=motor_labels, initial_value=motor_labels[1]
                )
                specs = (
                    ("Position (rad)", "p", P_MIN, P_MAX, 0.001),
                    ("Velocity (rad/s)", "v", V_MIN, V_MAX, 0.1),
                    ("Kp (Nm/rad)", "kp", KP_MIN, KP_MAX, 0.1),
                    ("Kd (Nm s/rad)", "kd", KD_MIN, KD_MAX, 0.01),
                    ("Torque FF (Nm)", "t", T_MIN, T_MAX, 0.001),
                )
                for label, field, minimum, maximum, step in specs:
                    handle = self.server.gui.add_slider(
                        label,
                        min=minimum,
                        max=maximum,
                        step=step,
                        initial_value=0.0,
                    )
                    self.command_handles[field] = handle

                    @handle.on_update
                    def _command_changed(
                        _event, key=field, gui=handle
                    ) -> None:
                        if self._syncing_controls:
                            return
                        self.bridge.set_command(
                            self.selected_motor_id, key, float(gui.value)
                        )
                        if key == "t":
                            self._syncing_controls = True
                            try:
                                self.torque_handles[
                                    self.selected_motor_id
                                ].value = gui.value
                            finally:
                                self._syncing_controls = False

                @selected_motor.on_update
                def _selected_motor_changed(_event) -> None:
                    self.selected_motor_id = int(
                        NODE_BY_LABEL[selected_motor.value]["id"]
                    )
                    self._load_selected_command_into_gui()

            with self.server.gui.add_folder("Motor state commands"):
                all_on = self.server.gui.add_button(
                    "ALL MOTORS ON", color="green"
                )
                all_off = self.server.gui.add_button(
                    "ALL MOTORS OFF", color="red"
                )
                all_align = self.server.gui.add_button(
                    "ALL MOTORS ALIGN", color="yellow"
                )
                selected_on = self.server.gui.add_button("Selected motor ON")
                selected_off = self.server.gui.add_button(
                    "Selected motor OFF", color="red"
                )
                selected_align = self.server.gui.add_button(
                    "Selected motor ALIGN", color="yellow"
                )

                def send_control(command: int) -> None:
                    try:
                        self.bridge.queue_control(self.selected_motor_id, command)
                    except RuntimeError as exc:
                        self.connection_status.value = str(exc)

                def send_all_controls(command: int) -> None:
                    try:
                        for motor_id in MOTOR_IDS:
                            self.bridge.queue_control(motor_id, command)
                    except RuntimeError as exc:
                        self.connection_status.value = str(exc)

                @all_on.on_click
                def _all_on(_event) -> None:
                    send_all_controls(0x01)

                @all_off.on_click
                def _all_off(_event) -> None:
                    send_all_controls(0x02)

                @all_align.on_click
                def _all_align(_event) -> None:
                    send_all_controls(0x03)

                @selected_on.on_click
                def _selected_on(_event) -> None:
                    send_control(0x01)

                @selected_off.on_click
                def _selected_off(_event) -> None:
                    send_control(0x02)

                @selected_align.on_click
                def _selected_align(_event) -> None:
                    send_control(0x03)

                for axis, motor_id in (("X", 0x01), ("Y", 0x02)):
                    on_button = self.server.gui.add_button(f"Flywheel {axis} ON")
                    off_button = self.server.gui.add_button(
                        f"Flywheel {axis} OFF", color="red"
                    )
                    align_button = self.server.gui.add_button(
                        f"Flywheel {axis} ALIGN", color="yellow"
                    )

                    @on_button.on_click
                    def _axis_on(_event, mid=motor_id) -> None:
                        try:
                            self.bridge.queue_control(mid, 0x01)
                        except RuntimeError as exc:
                            self.connection_status.value = str(exc)

                    @off_button.on_click
                    def _axis_off(_event, mid=motor_id) -> None:
                        try:
                            self.bridge.queue_control(mid, 0x02)
                        except RuntimeError as exc:
                            self.connection_status.value = str(exc)

                    @align_button.on_click
                    def _axis_align(_event, mid=motor_id) -> None:
                        try:
                            self.bridge.queue_control(mid, 0x03)
                        except RuntimeError as exc:
                            self.connection_status.value = str(exc)

            with self.server.gui.add_folder("Sensor LED"):
                led = self.server.gui.add_rgb("RGB", (0, 0, 0))
                led_off = self.server.gui.add_button("LED off")

                @led.on_update
                def _led_changed(_event) -> None:
                    self.bridge.set_led(led.value)

                @led_off.on_click
                def _led_off(_event) -> None:
                    led.value = (0, 0, 0)
                    self.bridge.set_led((0, 0, 0))

            with self.server.gui.add_folder("Live robot state"):
                for node_id, name in (
                    (0x01, "Flywheel X"),
                    (0x02, "Flywheel Y"),
                    (0x03, "Spring actuator"),
                    (0x04, "Jumping sensor"),
                ):
                    self.telemetry_handles[node_id] = self.server.gui.add_text(
                        name, "Waiting for CAN telemetry", disabled=True
                    )
                self.axis_x_status = self.server.gui.add_text(
                    "Body X", "Waiting", disabled=True
                )
                self.axis_y_status = self.server.gui.add_text(
                    "Body Y", "Waiting", disabled=True
                )
                self.axis_z_status = self.server.gui.add_text(
                    "Body yaw", "Waiting for yaw firmware", disabled=True
                )
                self.io_counts = self.server.gui.add_text(
                    "Transport", "RX 0 / TX 0", disabled=True
                )

            with self.server.gui.add_folder("Visualization mapping"):
                self.base_height = self.server.gui.add_number(
                    "Base center height (m)",
                    0.45,
                    min=-1.0,
                    max=2.0,
                    step=0.01,
                )
                self.base_pivot_z = self.server.gui.add_number(
                    "Base center in model (m)",
                    0.234258,
                    min=-1.0,
                    max=1.0,
                    step=0.001,
                    hint="Local Z coordinate of the main housing center.",
                )
                self.linear_scale = self.server.gui.add_number(
                    "Spring scale (m/rad)",
                    0.008 / (2.0 * math.pi),
                    min=-0.1,
                    max=0.1,
                    step=0.0001,
                    hint="Default assumes an 8 mm lead per motor revolution.",
                )
                self.linear_offset = self.server.gui.add_number(
                    "Spring zero offset (m)",
                    float(np.mean(self.linear_range)),
                    min=float(self.linear_range[0]),
                    max=float(self.linear_range[1]),
                    step=0.001,
                )

    def _load_selected_command_into_gui(self) -> None:
        snapshot = self.bridge.snapshot()
        command = snapshot["commands"][self.selected_motor_id]
        self._syncing_controls = True
        try:
            for field, handle in self.command_handles.items():
                handle.value = command[field]
            self.torque_handles[self.selected_motor_id].value = command["t"]
        finally:
            self._syncing_controls = False

    def _update_model(self, snapshot: dict) -> None:
        feedback = snapshot["feedback"]
        if feedback[0x01]["seen"]:
            self.data.qpos[self.flywheel_x_qpos] = feedback[0x01]["position"]
        if feedback[0x02]["seen"]:
            self.data.qpos[self.flywheel_y_qpos] = feedback[0x02]["position"]
        if feedback[0x03]["seen"]:
            linear_position = (
                float(self.linear_offset.value)
                + feedback[0x03]["position"] * float(self.linear_scale.value)
            )
            self.data.qpos[self.linear_qpos] = float(
                np.clip(linear_position, *self.linear_range)
            )

        roll = (
            snapshot["axes"]["x"]["angle"]
            if snapshot["axes"]["x"]["sensor_seen"]
            else 0.0
        )
        pitch = (
            snapshot["axes"]["y"]["angle"]
            if snapshot["axes"]["y"]["sensor_seen"]
            else 0.0
        )
        yaw = (
            snapshot["axes"]["z"]["angle"]
            if snapshot["axes"]["z"]["sensor_seen"]
            else 0.0
        )
        quat = body_quaternion(roll, pitch, yaw)
        local_pivot = np.array([0.0, 0.0, float(self.base_pivot_z.value)])
        world_pivot = np.array([0.0, 0.0, float(self.base_height.value)])
        root_position = world_pivot - quaternion_matrix(quat) @ local_pivot
        self.data.qpos[self.base_qpos : self.base_qpos + 3] = root_position
        self.data.qpos[self.base_qpos + 3 : self.base_qpos + 7] = quat

        mujoco.mj_forward(self.model, self.data)
        self.scene.update_from_mjdata(self.data)

    def _update_telemetry_gui(self, snapshot: dict) -> None:
        for node_id, handle in self.telemetry_handles.items():
            feedback = snapshot["feedback"][node_id]
            if not feedback["seen"]:
                handle.value = "Waiting for CAN telemetry"
                continue
            fault_text = (
                decode_sensor_faults(feedback["error"])
                if node_id == SENSOR_ID
                else decode_faults(feedback["error"])
            )
            torque_label = "tof_mm" if node_id == SENSOR_ID else "torque"
            handle.value = (
                f"pos={feedback['position']:+.4f} rad | "
                f"vel={feedback['velocity']:+.3f} rad/s | "
                f"{torque_label}={feedback['torque']:+.4f} | "
                f"state={feedback['state']} | {fault_text} | "
                f"frames={feedback['count']}"
            )

        for axis, handle in (
            ("x", self.axis_x_status),
            ("y", self.axis_y_status),
            ("z", self.axis_z_status),
        ):
            values = snapshot["axes"][axis]
            if not values["sensor_seen"]:
                handle.value = "Waiting for CAN telemetry"
            else:
                handle.value = (
                    f"angle={values['angle']:+.4f} rad | "
                    f"rate={values['rate']:+.3f} rad/s | "
                    f"flywheel={values['speed']:+.2f} rad/s | "
                    f"command={values['command']:+.4f} Nm"
                )
        self.io_counts.value = (
            f"RX {snapshot['rx_count']} / TX {snapshot['tx_count']}"
        )
        if snapshot["io_error"]:
            self.connection_status.value = f"I/O error: {snapshot['io_error']}"
        elif snapshot["running"]:
            mode = (
                "MOTOR OUTPUT ENABLED"
                if self.bridge.motor_output_enabled
                else "read-only"
            )
            self.connection_status.value = f"Connected ({mode})"

    def run(self) -> None:
        print(f"Viser hardware UI: http://localhost:{self.server.get_port()}")
        try:
            while True:
                snapshot = self.bridge.snapshot()
                self._update_model(snapshot)
                self._update_telemetry_gui(snapshot)
                time.sleep(1.0 / 30.0)
        except KeyboardInterrupt:
            print("Stopping hardware viewer...")
        finally:
            self.bridge.disconnect()
            self.server.stop()


def run_probe(port: str, duration: float) -> None:
    bridge = HardwareBridge()
    bridge.connect(port)
    try:
        time.sleep(duration)
        snapshot = bridge.snapshot()
        print(
            f"Read-only probe: RX={snapshot['rx_count']} "
            f"TX={snapshot['tx_count']}"
        )
        for node_id, feedback in snapshot["feedback"].items():
            print(
                f"0x{node_id:02X}: seen={feedback['seen']} "
                f"frames={feedback['count']} pos={feedback['position']:+.4f} "
                f"vel={feedback['velocity']:+.3f}"
            )
    finally:
        bridge.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).with_name("onshape_mjcf") / "scene.xml",
    )
    parser.add_argument("--port", default=preferred_port())
    parser.add_argument("--server-port", type=int, default=8080)
    parser.add_argument(
        "--no-auto-connect",
        action="store_true",
        help="Launch without automatically connecting read-only to the serial port.",
    )
    parser.add_argument(
        "--probe-seconds",
        type=float,
        default=0.0,
        help="Read hardware telemetry without launching Viser or sending motor commands.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.probe_seconds > 0:
        if not args.port:
            raise RuntimeError("No serial port found for telemetry probe")
        run_probe(args.port, args.probe_seconds)
        return
    app = HardwareViserApp(
        args.model.resolve(),
        args.server_port,
        args.port,
        auto_connect=not args.no_auto_connect,
    )
    app.run()


if __name__ == "__main__":
    main()
