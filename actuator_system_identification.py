"""
Free-spinning actuator system identification over the Waveshare USB-CAN adapter.

Experiments:
  1. Slow bidirectional torque ramps for static breakaway friction.
  2. Bidirectional constant-speed plateaus for Coulomb friction and damping.
  3. Bidirectional torque chirps for rotor inertia and dynamic validation.

The analysis fits the free-rotor model

    torque = inertia * acceleration
           + damping * velocity
           + coulomb_friction * tanh(velocity / friction_velocity_scale)
           + torque_bias

Requirements:
    pip install pyserial numpy scipy matplotlib

Examples:
    python actuator_system_identification.py run --actuator flywheel_x --port COM11
    python actuator_system_identification.py analyze path\\to\\raw_samples.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


SERIAL_BAUD = 2_000_000
POSITION_MIN = -math.pi
POSITION_MAX = math.pi
FEEDBACK_VELOCITY_MIN = -1500.0
FEEDBACK_VELOCITY_MAX = 1500.0
KP_MIN = 0.0
KP_MAX = 500.0
KD_MIN = 0.0
KD_MAX = 15.0

STATE_NAMES = {
    0: "IDLE",
    1: "RUN",
    2: "STOP",
    3: "FAULT_NOW",
    4: "FAULT_OVER",
    5: "OTHER",
}

FAULT_NAMES = {
    0x01: "FOC_RATE",
    0x02: "OVER_VOLTAGE",
    0x04: "UNDER_VOLTAGE",
    0x08: "OVER_TEMPERATURE",
    0x10: "STARTUP",
    0x20: "SPEED_FEEDBACK",
    0x40: "OVER_CURRENT",
    0x80: "SOFTWARE",
}

CSV_FIELDS = [
    "host_time_utc",
    "session_time_s",
    "experiment",
    "trial",
    "direction",
    "target_velocity_rad_s",
    "chirp_frequency_hz",
    "command_position_rad",
    "command_velocity_rad_s",
    "command_kp_nm_rad",
    "command_kd_nm_s_rad",
    "command_torque_nm",
    "encoded_torque_nm",
    "position_rad",
    "unwrapped_position_rad",
    "velocity_rad_s",
    "measured_torque_nm",
    "motor_state",
    "fault_code",
    "round_trip_ms",
]


@dataclass(frozen=True)
class ActuatorProfile:
    name: str
    node_id: int
    command_velocity_max_rad_s: float
    torque_max_nm: float
    default_stiction_max_nm: float
    default_stiction_ramp_nm_s: float
    default_chirp_amplitude_nm: float
    default_max_velocity_rad_s: float
    default_velocity_kd_nm_s_rad: float
    default_friction_speeds_rad_s: tuple[float, ...]

    @property
    def feedback_id(self) -> int:
        return self.node_id + 0x100

    @property
    def control_id(self) -> int:
        return self.node_id + 0x200


ACTUATORS = {
    "flywheel_x": ActuatorProfile(
        name="flywheel_x",
        node_id=0x01,
        command_velocity_max_rad_s=45.0,
        torque_max_nm=0.5,
        default_stiction_max_nm=0.12,
        default_stiction_ramp_nm_s=0.015,
        default_chirp_amplitude_nm=0.04,
        default_max_velocity_rad_s=600.0,
        default_velocity_kd_nm_s_rad=0.01,
        default_friction_speeds_rad_s=(5.0, 15.0, 30.0, 40.0),
    ),
    "flywheel_y": ActuatorProfile(
        name="flywheel_y",
        node_id=0x02,
        command_velocity_max_rad_s=45.0,
        torque_max_nm=0.5,
        default_stiction_max_nm=0.12,
        default_stiction_ramp_nm_s=0.015,
        default_chirp_amplitude_nm=0.04,
        default_max_velocity_rad_s=600.0,
        default_velocity_kd_nm_s_rad=0.02,
        default_friction_speeds_rad_s=(5.0, 15.0, 30.0, 40.0),
    ),
    "spring_actuator": ActuatorProfile(
        name="spring_actuator",
        node_id=0x03,
        command_velocity_max_rad_s=1500.0,
        torque_max_nm=0.76,
        default_stiction_max_nm=0.30,
        default_stiction_ramp_nm_s=0.03,
        default_chirp_amplitude_nm=0.08,
        default_max_velocity_rad_s=600.0,
        default_velocity_kd_nm_s_rad=0.06,
        default_friction_speeds_rad_s=(5.0, 20.0, 50.0, 100.0),
    ),
}


@dataclass(frozen=True)
class MotorCommand:
    position_rad: float = 0.0
    velocity_rad_s: float = 0.0
    kp_nm_rad: float = 0.0
    kd_nm_s_rad: float = 0.0
    torque_nm: float = 0.0


@dataclass(frozen=True)
class Feedback:
    received_monotonic_s: float
    state: int
    motor_id: int
    fault_code: int
    position_rad: float
    velocity_rad_s: float
    torque_nm: float


class IdentificationError(RuntimeError):
    pass


def checksum(data: Iterable[int]) -> int:
    return sum(data) & 0xFF


def float_to_uint(value: float, minimum: float, maximum: float, bits: int) -> int:
    value = max(min(value, maximum), minimum)
    return int((value - minimum) * ((1 << bits) - 1) / (maximum - minimum))


def uint_to_float(value: int, minimum: float, maximum: float, bits: int) -> float:
    return minimum + float(value) * (maximum - minimum) / ((1 << bits) - 1)


def make_adapter_frame(can_id: int, data: bytes) -> bytes:
    if not 0 <= can_id <= 0x7FF:
        raise ValueError(f"Standard CAN ID is out of range: 0x{can_id:X}")
    if not 0 <= len(data) <= 8:
        raise ValueError("Classic CAN payload must contain at most 8 bytes")

    padded = data + bytes(8 - len(data))
    id_bytes = [can_id & 0xFF, (can_id >> 8) & 0xFF, 0x00, 0x00]
    core = [0x01, 0x01, 0x00] + id_bytes + [len(data)] + list(padded) + [0x00]
    return bytes([0xAA, 0x55] + core + [checksum(core)])


def make_adapter_init_frame() -> bytes:
    core = [
        0x12, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
    ]
    return bytes([0xAA, 0x55] + core + [checksum(core)])


def pack_motor_command(command: MotorCommand, profile: ActuatorProfile) -> bytes:
    velocity_min = -profile.command_velocity_max_rad_s
    velocity_max = profile.command_velocity_max_rad_s
    torque_min = -profile.torque_max_nm
    torque_max = profile.torque_max_nm

    p_int = float_to_uint(
        command.position_rad, POSITION_MIN, POSITION_MAX, 16
    )
    v_int = float_to_uint(command.velocity_rad_s, velocity_min, velocity_max, 12)
    kp_int = float_to_uint(command.kp_nm_rad, KP_MIN, KP_MAX, 12)
    kd_int = float_to_uint(command.kd_nm_s_rad, KD_MIN, KD_MAX, 12)
    torque_int = float_to_uint(command.torque_nm, torque_min, torque_max, 12)

    can_data = bytearray(8)
    can_data[0] = (p_int >> 8) & 0xFF
    can_data[1] = p_int & 0xFF
    can_data[2] = (v_int >> 4) & 0xFF
    can_data[3] = ((v_int & 0x0F) << 4) | ((kp_int >> 8) & 0x0F)
    can_data[4] = kp_int & 0xFF
    can_data[5] = (kd_int >> 4) & 0xFF
    can_data[6] = ((kd_int & 0x0F) << 4) | ((torque_int >> 8) & 0x0F)
    can_data[7] = torque_int & 0xFF
    return make_adapter_frame(profile.node_id, bytes(can_data))


def encoded_torque(command: MotorCommand, profile: ActuatorProfile) -> float:
    raw = float_to_uint(
        command.torque_nm, -profile.torque_max_nm, profile.torque_max_nm, 12
    )
    return uint_to_float(
        raw, -profile.torque_max_nm, profile.torque_max_nm, 12
    )


def decode_feedback(
    data: bytes, received_monotonic_s: float, profile: ActuatorProfile
) -> Feedback:
    if len(data) != 8:
        raise IdentificationError(f"Expected 8 feedback bytes, received {len(data)}")

    state = (data[0] >> 4) & 0x0F
    motor_id = data[0] & 0x0F
    position_raw = (data[2] << 8) | data[3]
    velocity_raw = (data[4] << 8) | data[5]
    torque_raw = (data[6] << 8) | data[7]
    return Feedback(
        received_monotonic_s=received_monotonic_s,
        state=state,
        motor_id=motor_id,
        fault_code=data[1],
        position_rad=uint_to_float(
            position_raw, POSITION_MIN, POSITION_MAX, 16
        ),
        velocity_rad_s=uint_to_float(
            velocity_raw, FEEDBACK_VELOCITY_MIN, FEEDBACK_VELOCITY_MAX, 16
        ),
        torque_nm=uint_to_float(
            torque_raw, -profile.torque_max_nm, profile.torque_max_nm, 16
        ),
    )


def fault_description(code: int) -> str:
    if code == 0:
        return "none"
    names = [name for bit, name in FAULT_NAMES.items() if code & bit]
    return ", ".join(names) if names else f"unknown 0x{code:02X}"


class WaveshareTransport:
    def __init__(self, port: str, profile: ActuatorProfile, timeout_s: float):
        try:
            import serial
        except ImportError as exc:
            raise IdentificationError(
                "pyserial is required for acquisition: pip install pyserial"
            ) from exc

        self._profile = profile
        self._timeout_s = timeout_s
        self._rx_buffer = bytearray()
        self._serial = serial.Serial(
            port=port,
            baudrate=SERIAL_BAUD,
            timeout=0,
            write_timeout=max(0.1, timeout_s),
        )
        self._serial.reset_input_buffer()
        self._serial.write(make_adapter_init_frame())
        self._serial.flush()
        time.sleep(0.2)
        self._serial.reset_input_buffer()

    def close(self) -> None:
        if self._serial.is_open:
            self._serial.close()

    def flush(self) -> None:
        self._serial.flush()

    def send(self, frame: bytes) -> float:
        sent_at = time.monotonic()
        written = self._serial.write(frame)
        if written != len(frame):
            raise IdentificationError(
                f"Serial write was incomplete: {written}/{len(frame)} bytes"
            )
        return sent_at

    def exchange(self, frame: bytes) -> tuple[Feedback, float]:
        sent_at = self.send(frame)
        deadline = sent_at + self._timeout_s

        while time.monotonic() < deadline:
            waiting = self._serial.in_waiting
            if waiting:
                self._rx_buffer.extend(self._serial.read(waiting))

            while len(self._rx_buffer) >= 13:
                if self._rx_buffer[0] != 0xAA or self._rx_buffer[12] != 0x55:
                    del self._rx_buffer[0]
                    continue

                frame_data = bytes(self._rx_buffer[:13])
                del self._rx_buffer[:13]
                can_id = frame_data[2] | (frame_data[3] << 8)
                if can_id != self._profile.feedback_id:
                    continue

                received_at = time.monotonic()
                feedback = decode_feedback(
                    frame_data[4:12], received_at, self._profile
                )
                return feedback, (received_at - sent_at) * 1000.0

            time.sleep(0.0002)

        raise IdentificationError(
            f"No feedback from CAN ID 0x{self._profile.feedback_id:03X} "
            f"within {self._timeout_s * 1000.0:.0f} ms"
        )

    def control(self, command_byte: int) -> Feedback:
        feedback, _ = self.exchange(
            make_adapter_frame(self._profile.control_id, bytes([command_byte]))
        )
        return feedback


class PositionUnwrapper:
    def __init__(self) -> None:
        self._previous: float | None = None
        self._unwrapped = 0.0

    def update(self, position_rad: float) -> float:
        if self._previous is None:
            self._previous = position_rad
            self._unwrapped = position_rad
            return self._unwrapped

        delta = position_rad - self._previous
        if delta > math.pi:
            delta -= 2.0 * math.pi
        elif delta < -math.pi:
            delta += 2.0 * math.pi
        self._unwrapped += delta
        self._previous = position_rad
        return self._unwrapped


class ExperimentRunner:
    def __init__(
        self,
        transport: WaveshareTransport,
        profile: ActuatorProfile,
        csv_path: Path,
        sample_rate_hz: float,
        max_velocity_rad_s: float,
        movement_threshold_rad_s: float,
        brake_timeout_s: float,
        rest_confirm_s: float,
        zero_torque_settle_s: float,
    ):
        self.transport = transport
        self.profile = profile
        self.csv_path = csv_path
        self.period_s = 1.0 / sample_rate_hz
        self.max_velocity_rad_s = max_velocity_rad_s
        self.movement_threshold_rad_s = movement_threshold_rad_s
        self.brake_timeout_s = brake_timeout_s
        self.rest_confirm_s = rest_confirm_s
        self.zero_torque_settle_s = zero_torque_settle_s
        self._started_at = time.monotonic()
        self._unwrapper = PositionUnwrapper()
        self._csv_file = csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._row_count = 0

    def close(self) -> None:
        self._csv_file.flush()
        self._csv_file.close()

    def start_motor(self, timeout_s: float = 8.0) -> None:
        print("Starting motor state machine...")
        feedback = self.transport.control(0x01)
        deadline = time.monotonic() + timeout_s
        while feedback.state != 1 and time.monotonic() < deadline:
            if feedback.fault_code:
                raise IdentificationError(
                    "Motor fault while starting: "
                    f"{fault_description(feedback.fault_code)}"
                )
            time.sleep(0.05)
            feedback = self.transport.control(0x04)

        if feedback.state != 1:
            state_name = STATE_NAMES.get(feedback.state, str(feedback.state))
            raise IdentificationError(
                f"Motor did not reach RUN state; last state was {state_name}"
            )

        self.sample(MotorCommand(), "preflight", 0)
        print("Motor is in RUN state.")

    def safe_shutdown(self) -> None:
        zero_frame = pack_motor_command(MotorCommand(), self.profile)
        try:
            for _ in range(5):
                try:
                    self.transport.send(zero_frame)
                except (IdentificationError, OSError) as exc:
                    print(
                        f"Warning: could not send zero torque: {exc}",
                        file=sys.stderr,
                    )
                time.sleep(0.01)
        finally:
            try:
                self.transport.send(
                    make_adapter_frame(self.profile.control_id, bytes([0x02]))
                )
                self.transport.flush()
            except (IdentificationError, OSError) as exc:
                print(f"Warning: could not send motor stop: {exc}", file=sys.stderr)

    def _wait_for_slot(self, target_s: float) -> None:
        remaining = target_s - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    def sample(
        self,
        command: MotorCommand,
        experiment: str,
        trial: int,
        *,
        direction: int = 0,
        target_velocity_rad_s: float | None = None,
        chirp_frequency_hz: float | None = None,
    ) -> Feedback:
        if abs(command.torque_nm) > self.profile.torque_max_nm:
            raise IdentificationError(
                f"Requested torque {command.torque_nm:.4f} Nm exceeds the "
                f"{self.profile.torque_max_nm:.4f} Nm firmware limit"
            )
        if abs(command.velocity_rad_s) > self.profile.command_velocity_max_rad_s:
            raise IdentificationError(
                f"Requested velocity {command.velocity_rad_s:.2f} rad/s exceeds "
                f"the {self.profile.command_velocity_max_rad_s:.2f} rad/s "
                "firmware command range"
            )

        feedback, round_trip_ms = self.transport.exchange(
            pack_motor_command(command, self.profile)
        )
        if feedback.motor_id != self.profile.node_id:
            raise IdentificationError(
                f"Feedback motor ID {feedback.motor_id} does not match expected "
                f"ID {self.profile.node_id}"
            )
        if feedback.fault_code:
            raise IdentificationError(
                f"Motor fault 0x{feedback.fault_code:02X}: "
                f"{fault_description(feedback.fault_code)}"
            )
        if feedback.state != 1:
            state_name = STATE_NAMES.get(feedback.state, str(feedback.state))
            raise IdentificationError(f"Motor left RUN state: {state_name}")
        if abs(feedback.velocity_rad_s) > self.max_velocity_rad_s:
            raise IdentificationError(
                f"Overspeed: {feedback.velocity_rad_s:.2f} rad/s exceeds "
                f"the {self.max_velocity_rad_s:.2f} rad/s safety limit"
            )

        session_time = feedback.received_monotonic_s - self._started_at
        row = {
            "host_time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "session_time_s": f"{session_time:.9f}",
            "experiment": experiment,
            "trial": trial,
            "direction": direction,
            "target_velocity_rad_s": (
                "" if target_velocity_rad_s is None else target_velocity_rad_s
            ),
            "chirp_frequency_hz": (
                "" if chirp_frequency_hz is None else chirp_frequency_hz
            ),
            "command_position_rad": command.position_rad,
            "command_velocity_rad_s": command.velocity_rad_s,
            "command_kp_nm_rad": command.kp_nm_rad,
            "command_kd_nm_s_rad": command.kd_nm_s_rad,
            "command_torque_nm": command.torque_nm,
            "encoded_torque_nm": encoded_torque(command, self.profile),
            "position_rad": feedback.position_rad,
            "unwrapped_position_rad": self._unwrapper.update(
                feedback.position_rad
            ),
            "velocity_rad_s": feedback.velocity_rad_s,
            "measured_torque_nm": feedback.torque_nm,
            "motor_state": feedback.state,
            "fault_code": feedback.fault_code,
            "round_trip_ms": f"{round_trip_ms:.6f}",
        }
        self._writer.writerow(row)
        self._row_count += 1
        if self._row_count % 100 == 0:
            self._csv_file.flush()
        return feedback

    def run_timed(
        self,
        duration_s: float,
        experiment: str,
        trial: int,
        command_at: Callable[[float], MotorCommand],
        *,
        direction: int = 0,
        target_velocity_rad_s: float | None = None,
        frequency_at: Callable[[float], float] | None = None,
    ) -> list[Feedback]:
        started = time.monotonic()
        next_sample = started
        feedback_samples = []
        while True:
            now = time.monotonic()
            elapsed = now - started
            if elapsed >= duration_s:
                break
            self._wait_for_slot(next_sample)
            elapsed = time.monotonic() - started
            feedback_samples.append(
                self.sample(
                    command_at(elapsed),
                    experiment,
                    trial,
                    direction=direction,
                    target_velocity_rad_s=target_velocity_rad_s,
                    chirp_frequency_hz=(
                        None if frequency_at is None else frequency_at(elapsed)
                    ),
                )
            )
            next_sample += self.period_s
            if next_sample < time.monotonic() - self.period_s:
                next_sample = time.monotonic()
        return feedback_samples

    def brake_to_rest(self, kd_nm_s_rad: float) -> None:
        stable_since: float | None = None
        started = time.monotonic()
        next_sample = started
        command = MotorCommand(velocity_rad_s=0.0, kd_nm_s_rad=kd_nm_s_rad)
        previous_speed_abs: float | None = None
        runaway_growth_margin = max(
            0.05, self.movement_threshold_rad_s * 0.1
        )
        runaway_confirm_samples = max(3, math.ceil(0.15 / self.period_s))
        runaway_samples = 0

        while time.monotonic() - started < self.brake_timeout_s:
            self._wait_for_slot(next_sample)
            feedback = self.sample(command, "rest", 0)
            now = time.monotonic()
            speed_abs = abs(feedback.velocity_rad_s)

            if (
                speed_abs > self.movement_threshold_rad_s
                and previous_speed_abs is not None
                and speed_abs > previous_speed_abs + runaway_growth_margin
            ):
                runaway_samples += 1
            else:
                runaway_samples = 0

            previous_speed_abs = speed_abs

            if runaway_samples >= runaway_confirm_samples:
                raise IdentificationError(
                    "Controlled stop appears to be accelerating instead of braking "
                    f"(speed reached {feedback.velocity_rad_s:+.2f} rad/s while "
                    "targeting 0 rad/s)."
                )
            if speed_abs <= self.movement_threshold_rad_s:
                stable_since = now if stable_since is None else stable_since
                if now - stable_since >= self.rest_confirm_s:
                    self.run_timed(
                        self.zero_torque_settle_s,
                        "rest",
                        0,
                        lambda _: MotorCommand(),
                    )
                    return
            else:
                stable_since = None
            next_sample += self.period_s

        raise IdentificationError(
            "Actuator did not settle before the next experiment"
        )

    def run_stiction(
        self,
        repeats: int,
        maximum_torque_nm: float,
        ramp_rate_nm_s: float,
        movement_confirm_s: float,
        brake_kd_nm_s_rad: float,
    ) -> None:
        print("\nStiction test: slow bidirectional torque ramps")
        required_moving_samples = max(
            1, math.ceil(movement_confirm_s / self.period_s)
        )
        trial = 0
        for repeat in range(repeats):
            for direction in (1, -1):
                trial += 1
                self.brake_to_rest(brake_kd_nm_s_rad)
                print(
                    f"  ramp {trial}/{2 * repeats}: "
                    f"{'positive' if direction > 0 else 'negative'}"
                )
                started = time.monotonic()
                next_sample = started
                moving_samples = 0
                breakaway_torque: float | None = None
                while True:
                    self._wait_for_slot(next_sample)
                    elapsed = time.monotonic() - started
                    torque = min(maximum_torque_nm, elapsed * ramp_rate_nm_s)
                    command = MotorCommand(torque_nm=direction * torque)
                    feedback = self.sample(
                        command,
                        "stiction",
                        trial,
                        direction=direction,
                    )
                    if (
                        abs(feedback.velocity_rad_s)
                        >= self.movement_threshold_rad_s
                    ):
                        moving_samples += 1
                    else:
                        moving_samples = 0

                    if moving_samples >= required_moving_samples:
                        breakaway_torque = direction * torque
                        break
                    if torque >= maximum_torque_nm:
                        break
                    next_sample += self.period_s

                self.sample(MotorCommand(), "stiction_release", trial)
                if breakaway_torque is None:
                    print(
                        f"    no breakaway by {direction * maximum_torque_nm:+.4f} Nm"
                    )
                else:
                    print(f"    breakaway near {breakaway_torque:+.4f} Nm")

    def run_friction(
        self,
        speeds_rad_s: Sequence[float],
        settle_s: float,
        sample_s: float,
        kd_nm_s_rad: float,
    ) -> None:
        print("\nFriction test: bidirectional constant-speed plateaus")
        trial = 0
        for speed_magnitude in speeds_rad_s:
            for direction in (1, -1):
                trial += 1
                self.brake_to_rest(kd_nm_s_rad)
                target = direction * speed_magnitude
                print(f"  plateau {trial}/{2 * len(speeds_rad_s)}: {target:+.2f} rad/s")
                command = MotorCommand(
                    velocity_rad_s=target,
                    kd_nm_s_rad=kd_nm_s_rad,
                )
                self.run_timed(
                    settle_s,
                    "friction_settle",
                    trial,
                    lambda _, cmd=command: cmd,
                    direction=direction,
                    target_velocity_rad_s=target,
                )
                samples = self.run_timed(
                    sample_s,
                    "friction",
                    trial,
                    lambda _, cmd=command: cmd,
                    direction=direction,
                    target_velocity_rad_s=target,
                )
                mean_speed = statistics.fmean(s.velocity_rad_s for s in samples)
                mean_torque = statistics.fmean(s.torque_nm for s in samples)
                print(
                    f"    measured {mean_speed:+.2f} rad/s, "
                    f"{mean_torque:+.4f} Nm"
                )
        self.brake_to_rest(kd_nm_s_rad)

    def run_chirp(
        self,
        repeats: int,
        amplitude_nm: float,
        start_hz: float,
        end_hz: float,
        duration_s: float,
        chirp_kind: str,
        brake_kd_nm_s_rad: float,
    ) -> None:
        print(
            f"\nChirp test: {start_hz:.2f}-{end_hz:.2f} Hz "
            f"{chirp_kind} sweeps"
        )

        def phase_and_frequency(elapsed_s: float) -> tuple[float, float]:
            if chirp_kind == "linear":
                slope = (end_hz - start_hz) / duration_s
                frequency = start_hz + slope * elapsed_s
                phase = 2.0 * math.pi * (
                    start_hz * elapsed_s + 0.5 * slope * elapsed_s**2
                )
                return phase, frequency

            ratio = end_hz / start_hz
            log_ratio = math.log(ratio)
            frequency = start_hz * ratio ** (elapsed_s / duration_s)
            phase = (
                2.0
                * math.pi
                * start_hz
                * duration_s
                / log_ratio
                * (ratio ** (elapsed_s / duration_s) - 1.0)
            )
            return phase, frequency

        def taper(elapsed_s: float) -> float:
            edge_s = min(1.0, duration_s * 0.1)
            if elapsed_s < edge_s:
                return 0.5 * (1.0 - math.cos(math.pi * elapsed_s / edge_s))
            if elapsed_s > duration_s - edge_s:
                remaining = duration_s - elapsed_s
                return 0.5 * (1.0 - math.cos(math.pi * remaining / edge_s))
            return 1.0

        for repeat in range(repeats):
            direction = 1 if repeat % 2 == 0 else -1
            self.brake_to_rest(brake_kd_nm_s_rad)
            print(
                f"  sweep {repeat + 1}/{repeats}: "
                f"{'positive' if direction > 0 else 'negative'} phase"
            )

            def command_at(elapsed_s: float) -> MotorCommand:
                phase, _ = phase_and_frequency(elapsed_s)
                torque = (
                    direction
                    * amplitude_nm
                    * taper(elapsed_s)
                    * math.sin(phase)
                )
                return MotorCommand(torque_nm=torque)

            self.run_timed(
                duration_s,
                "chirp",
                repeat + 1,
                command_at,
                direction=direction,
                frequency_at=lambda elapsed: phase_and_frequency(elapsed)[1],
            )
            self.sample(MotorCommand(), "chirp_release", repeat + 1)
        self.brake_to_rest(brake_kd_nm_s_rad)


def parse_positive_speeds(value: str) -> tuple[float, ...]:
    try:
        speeds = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Speeds must be a comma-separated list of numbers"
        ) from exc
    if not speeds or any(speed <= 0 for speed in speeds):
        raise argparse.ArgumentTypeError("All friction speeds must be positive")
    return speeds


def validate_run_arguments(args: argparse.Namespace, profile: ActuatorProfile) -> None:
    numeric_arguments = {
        "rate-hz": args.rate_hz,
        "feedback-timeout-s": args.feedback_timeout_s,
        "max-velocity-rad-s": args.max_velocity_rad_s,
        "movement-threshold-rad-s": args.movement_threshold_rad_s,
        "movement-confirm-s": args.movement_confirm_s,
        "brake-timeout-s": args.brake_timeout_s,
        "rest-confirm-s": args.rest_confirm_s,
        "zero-torque-settle-s": args.zero_torque_settle_s,
        "velocity-kd": args.velocity_kd,
        "stiction-max-nm": args.stiction_max_nm,
        "stiction-ramp-nm-s": args.stiction_ramp_nm_s,
        "friction-settle-s": args.friction_settle_s,
        "friction-sample-s": args.friction_sample_s,
        "chirp-amplitude-nm": args.chirp_amplitude_nm,
        "chirp-start-hz": args.chirp_start_hz,
        "chirp-end-hz": args.chirp_end_hz,
        "chirp-duration-s": args.chirp_duration_s,
        "friction-velocity-scale": args.friction_velocity_scale,
        "smooth-window-s": args.smooth_window_s,
    }
    invalid = [
        name
        for name, value in numeric_arguments.items()
        if not math.isfinite(value)
    ]
    if invalid or any(not math.isfinite(speed) for speed in args.friction_speeds):
        names = invalid or ["friction-speeds"]
        raise IdentificationError(
            "Numeric arguments must be finite: " + ", ".join(names)
        )
    if not 10.0 <= args.rate_hz <= 500.0:
        raise IdentificationError("--rate-hz must be between 10 and 500")
    if args.feedback_timeout_s <= 0:
        raise IdentificationError("--feedback-timeout-s must be positive")
    if args.max_velocity_rad_s <= 0:
        raise IdentificationError("--max-velocity-rad-s must be positive")
    if args.movement_threshold_rad_s <= 0:
        raise IdentificationError("--movement-threshold-rad-s must be positive")
    if args.movement_confirm_s <= 0:
        raise IdentificationError("--movement-confirm-s must be positive")
    if (
        args.brake_timeout_s <= 0
        or args.rest_confirm_s <= 0
        or args.zero_torque_settle_s <= 0
    ):
        raise IdentificationError(
            "--brake-timeout-s, --rest-confirm-s, and "
            "--zero-torque-settle-s must be positive"
        )
    if args.stiction_repeats < 1:
        raise IdentificationError("--stiction-repeats must be at least 1")
    if not 0 < args.stiction_max_nm <= profile.torque_max_nm:
        raise IdentificationError(
            f"--stiction-max-nm must be in (0, {profile.torque_max_nm}]"
        )
    if args.stiction_ramp_nm_s <= 0:
        raise IdentificationError("--stiction-ramp-nm-s must be positive")
    if args.friction_settle_s <= 0 or args.friction_sample_s <= 0:
        raise IdentificationError(
            "--friction-settle-s and --friction-sample-s must be positive"
        )
    if not 0 < args.chirp_amplitude_nm <= profile.torque_max_nm:
        raise IdentificationError(
            f"--chirp-amplitude-nm must be in (0, {profile.torque_max_nm}]"
        )
    if not 0 < args.chirp_start_hz < args.chirp_end_hz:
        raise IdentificationError(
            "Chirp frequencies must satisfy 0 < start < end"
        )
    if args.chirp_end_hz >= 0.4 * args.rate_hz:
        raise IdentificationError(
            "--chirp-end-hz must be below 40% of the sample rate"
        )
    if args.chirp_repeats < 1 or args.chirp_duration_s <= 0:
        raise IdentificationError(
            "--chirp-repeats must be at least 1 and --chirp-duration-s positive"
        )
    if not 0 < args.velocity_kd <= KD_MAX:
        raise IdentificationError(f"--velocity-kd must be in (0, {KD_MAX}]")
    if any(
        speed > profile.command_velocity_max_rad_s
        for speed in args.friction_speeds
    ):
        raise IdentificationError(
            "A friction speed exceeds the firmware command range of "
            f"{profile.command_velocity_max_rad_s:g} rad/s"
        )
    if args.friction_velocity_scale <= 0 or args.smooth_window_s <= 0:
        raise IdentificationError(
            "--friction-velocity-scale and --smooth-window-s must be positive"
        )


def serializable_arguments(args: argparse.Namespace) -> dict[str, object]:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def create_session_directory(root: Path, actuator_name: str) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = root / f"{actuator_name}_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def run_acquisition(args: argparse.Namespace) -> Path:
    profile = ACTUATORS[args.actuator]
    args.stiction_max_nm = (
        profile.default_stiction_max_nm
        if args.stiction_max_nm is None
        else args.stiction_max_nm
    )
    args.stiction_ramp_nm_s = (
        profile.default_stiction_ramp_nm_s
        if args.stiction_ramp_nm_s is None
        else args.stiction_ramp_nm_s
    )
    args.chirp_amplitude_nm = (
        profile.default_chirp_amplitude_nm
        if args.chirp_amplitude_nm is None
        else args.chirp_amplitude_nm
    )
    args.max_velocity_rad_s = (
        profile.default_max_velocity_rad_s
        if args.max_velocity_rad_s is None
        else args.max_velocity_rad_s
    )
    args.velocity_kd = (
        profile.default_velocity_kd_nm_s_rad
        if args.velocity_kd is None
        else args.velocity_kd
    )
    args.friction_speeds = (
        profile.default_friction_speeds_rad_s
        if args.friction_speeds is None
        else args.friction_speeds
    )
    validate_run_arguments(args, profile)

    selected_tests = (
        ("stiction", "friction", "chirp")
        if "all" in args.tests
        else tuple(dict.fromkeys(args.tests))
    )
    session_dir = create_session_directory(args.output_root, profile.name)
    csv_path = session_dir / "raw_samples.csv"
    metadata_path = session_dir / "metadata.json"
    metadata = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": (
            "torque = inertia*acceleration + damping*velocity + "
            "coulomb_friction*tanh(velocity/friction_velocity_scale) + "
            "torque_bias"
        ),
        "actuator_profile": dataclasses.asdict(profile),
        "arguments": serializable_arguments(args),
        "selected_tests": list(selected_tests),
        "safety_note": (
            "Host shutdown sends zero torque and STOP, but the current firmware "
            "does not implement a CAN command timeout."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Actuator: {profile.name} (CAN ID 0x{profile.node_id:02X})")
    print(f"Output:   {session_dir}")
    print(f"Rate:     {args.rate_hz:g} Hz")
    print(f"Tests:    {', '.join(selected_tests)}")
    print(
        f"Limits:   {args.max_velocity_rad_s:g} rad/s, "
        f"{profile.torque_max_nm:g} Nm firmware torque"
    )
    print(
        "WARNING: The firmware has no CAN timeout. Keep a hardware power cutoff "
        "within reach."
    )
    if not args.yes:
        response = input("Type RUN to enable the motor and begin: ").strip()
        if response != "RUN":
            raise IdentificationError("Run cancelled")

    transport: WaveshareTransport | None = None
    runner: ExperimentRunner | None = None
    try:
        transport = WaveshareTransport(
            args.port, profile, args.feedback_timeout_s
        )
        runner = ExperimentRunner(
            transport=transport,
            profile=profile,
            csv_path=csv_path,
            sample_rate_hz=args.rate_hz,
            max_velocity_rad_s=args.max_velocity_rad_s,
            movement_threshold_rad_s=args.movement_threshold_rad_s,
            brake_timeout_s=args.brake_timeout_s,
            rest_confirm_s=args.rest_confirm_s,
            zero_torque_settle_s=args.zero_torque_settle_s,
        )
        runner.start_motor()
        if "stiction" in selected_tests:
            runner.run_stiction(
                repeats=args.stiction_repeats,
                maximum_torque_nm=args.stiction_max_nm,
                ramp_rate_nm_s=args.stiction_ramp_nm_s,
                movement_confirm_s=args.movement_confirm_s,
                brake_kd_nm_s_rad=args.velocity_kd,
            )
        if "friction" in selected_tests:
            runner.run_friction(
                speeds_rad_s=args.friction_speeds,
                settle_s=args.friction_settle_s,
                sample_s=args.friction_sample_s,
                kd_nm_s_rad=args.velocity_kd,
            )
        if "chirp" in selected_tests:
            runner.run_chirp(
                repeats=args.chirp_repeats,
                amplitude_nm=args.chirp_amplitude_nm,
                start_hz=args.chirp_start_hz,
                end_hz=args.chirp_end_hz,
                duration_s=args.chirp_duration_s,
                chirp_kind=args.chirp_kind,
                brake_kd_nm_s_rad=args.velocity_kd,
            )
    finally:
        try:
            if runner is not None:
                print("\nSending zero torque and stopping motor...")
                runner.safe_shutdown()
        finally:
            try:
                if runner is not None:
                    runner.close()
            finally:
                if transport is not None:
                    transport.close()

    if not args.skip_analysis:
        analyze_csv(
            csv_path,
            session_dir,
            friction_velocity_scale=args.friction_velocity_scale,
            smooth_window_s=args.smooth_window_s,
        )
    return session_dir


def load_numeric_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise IdentificationError(f"No samples found in {csv_path}")
    missing = set(CSV_FIELDS) - set(rows[0])
    if missing:
        raise IdentificationError(
            f"CSV is missing required columns: {', '.join(sorted(missing))}"
        )
    return rows


def numeric(rows: Sequence[dict[str, str]], field: str):
    import numpy as np

    return np.asarray([float(row[field]) for row in rows], dtype=float)


def coefficient_of_determination(actual, predicted) -> float:
    import numpy as np

    residual = np.sum((actual - predicted) ** 2)
    total = np.sum((actual - np.mean(actual)) ** 2)
    return float(1.0 - residual / total) if total > 0 else float("nan")


def analyze_stiction(
    rows: Sequence[dict[str, str]],
    movement_threshold_rad_s: float,
    movement_confirmation_samples: int,
) -> dict[str, object] | None:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["experiment"] == "stiction":
            groups[(row["trial"], row["direction"])].append(row)
    if not groups:
        return None

    breakaways: dict[str, list[float]] = {"positive": [], "negative": []}
    trials = []
    for (trial, direction_text), samples in sorted(groups.items()):
        samples.sort(key=lambda sample: float(sample["session_time_s"]))
        direction = int(direction_text)
        consecutive = 0
        breakaway_index: int | None = None
        for index, sample in enumerate(samples):
            if abs(float(sample["velocity_rad_s"])) >= movement_threshold_rad_s:
                consecutive += 1
                if consecutive >= movement_confirmation_samples:
                    breakaway_index = index - movement_confirmation_samples + 1
                    break
            else:
                consecutive = 0
        detected = breakaway_index is not None
        breakaway_sample = (
            samples[breakaway_index] if breakaway_index is not None else samples[-1]
        )
        measured_torque = float(breakaway_sample["measured_torque_nm"])
        command_torque = float(breakaway_sample["command_torque_nm"])
        key = "positive" if direction > 0 else "negative"
        if detected:
            breakaways[key].append(measured_torque)
        trials.append(
            {
                "trial": int(trial),
                "direction": direction,
                "detected": detected,
                "breakaway_torque_nm": measured_torque,
                "breakaway_command_torque_nm": command_torque,
            }
        )

    result: dict[str, object] = {"trials": trials}
    for key, values in breakaways.items():
        result[key] = {
            "count": len(values),
            "median_breakaway_torque_nm": (
                statistics.median(values) if values else None
            ),
            "mean_breakaway_torque_nm": (
                statistics.fmean(values) if values else None
            ),
            "standard_deviation_nm": (
                statistics.stdev(values) if len(values) > 1 else 0.0
            ),
        }
    return result


def analyze_friction(
    rows: Sequence[dict[str, str]], friction_velocity_scale: float
) -> tuple[dict[str, object] | None, object | None]:
    import numpy as np
    from scipy.optimize import lsq_linear

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["experiment"] == "friction":
            groups[(row["trial"], row["target_velocity_rad_s"])].append(row)
    if len(groups) < 3:
        return None, None

    plateau_rows = []
    for (trial, target), samples in sorted(groups.items()):
        velocity = statistics.fmean(
            float(sample["velocity_rad_s"]) for sample in samples
        )
        torque = statistics.fmean(
            float(sample["measured_torque_nm"]) for sample in samples
        )
        plateau_rows.append(
            {
                "trial": int(trial),
                "target_velocity_rad_s": float(target),
                "mean_velocity_rad_s": velocity,
                "mean_torque_nm": torque,
                "torque_standard_deviation_nm": (
                    statistics.stdev(
                        float(sample["measured_torque_nm"]) for sample in samples
                    )
                    if len(samples) > 1
                    else 0.0
                ),
            }
        )

    velocity = np.asarray(
        [row["mean_velocity_rad_s"] for row in plateau_rows], dtype=float
    )
    torque = np.asarray([row["mean_torque_nm"] for row in plateau_rows], dtype=float)
    design = np.column_stack(
        [
            velocity,
            np.tanh(velocity / friction_velocity_scale),
            np.ones_like(velocity),
        ]
    )
    fit_result = lsq_linear(
        design,
        torque,
        bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
    )
    if not fit_result.success:
        raise IdentificationError(f"Friction fit failed: {fit_result.message}")
    coefficients = fit_result.x
    predicted = design @ coefficients
    damping, coulomb, bias = (float(value) for value in coefficients)
    fit = {
        "damping_nm_s_rad": damping,
        "coulomb_friction_nm": coulomb,
        "torque_bias_nm": bias,
        "friction_velocity_scale_rad_s": friction_velocity_scale,
        "rmse_nm": float(np.sqrt(np.mean((torque - predicted) ** 2))),
        "r_squared": coefficient_of_determination(torque, predicted),
        "plateaus": plateau_rows,
    }
    return fit, (velocity, torque, predicted)


def odd_window(sample_count: int, requested: int, polynomial_order: int) -> int:
    window = min(sample_count if sample_count % 2 else sample_count - 1, requested)
    window = max(window, polynomial_order + 2)
    if window % 2 == 0:
        window += 1
    if window > sample_count:
        window = sample_count if sample_count % 2 else sample_count - 1
    return window


def analyze_chirp(
    rows: Sequence[dict[str, str]],
    friction_fit: dict[str, object] | None,
    friction_velocity_scale: float,
    smooth_window_s: float,
) -> tuple[dict[str, object] | None, object | None]:
    import numpy as np
    from scipy.optimize import lsq_linear
    from scipy.signal import savgol_filter

    chirp_rows = [row for row in rows if row["experiment"] == "chirp"]
    if len(chirp_rows) < 20:
        return None, None

    trial_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in chirp_rows:
        trial_rows[row["trial"]].append(row)
    usable_trials = [
        samples
        for _, samples in sorted(trial_rows.items(), key=lambda item: int(item[0]))
        if len(samples) >= 7
    ]
    if not usable_trials:
        return None, None

    time_steps = np.concatenate(
        [np.diff(numeric(samples, "session_time_s")) for samples in usable_trials]
    )
    dt_s = float(np.median(time_steps))
    sample_rate_hz = 1.0 / dt_s
    requested_window = max(5, round(smooth_window_s * sample_rate_hz))
    processed_trials = []
    windows = []
    for samples in usable_trials:
        sample_time = numeric(samples, "session_time_s")
        velocity_raw = numeric(samples, "velocity_rad_s")
        torque_raw = numeric(samples, "measured_torque_nm")
        window = odd_window(len(samples), requested_window, polynomial_order=3)
        if window < 5:
            continue
        windows.append(window)
        processed_trials.append(
            (
                sample_time,
                savgol_filter(velocity_raw, window, 3, mode="interp"),
                savgol_filter(
                    velocity_raw,
                    window,
                    3,
                    deriv=1,
                    delta=float(np.median(np.diff(sample_time))),
                    mode="interp",
                ),
                savgol_filter(torque_raw, window, 3, mode="interp"),
                numeric(samples, "encoded_torque_nm"),
                numeric(samples, "chirp_frequency_hz"),
            )
        )
    if not processed_trials:
        raise IdentificationError("Too few chirp samples for derivative estimation")

    lag_limit = min(10, min(len(trial[0]) for trial in processed_trials) // 20)
    best = None

    for lag in range(-lag_limit, lag_limit + 1):
        aligned_parts = []
        for (
            trial_time,
            trial_velocity,
            trial_acceleration,
            trial_torque,
            trial_command,
            trial_frequency,
        ) in processed_trials:
            if lag < 0:
                aligned_parts.append(
                    (
                        trial_time[:lag],
                        trial_velocity[:lag],
                        trial_acceleration[:lag],
                        trial_torque[-lag:],
                        trial_command[:lag],
                        trial_frequency[:lag],
                    )
                )
            elif lag > 0:
                aligned_parts.append(
                    (
                        trial_time[lag:],
                        trial_velocity[lag:],
                        trial_acceleration[lag:],
                        trial_torque[:-lag],
                        trial_command[lag:],
                        trial_frequency[lag:],
                    )
                )
            else:
                aligned_parts.append(
                    (
                        trial_time,
                        trial_velocity,
                        trial_acceleration,
                        trial_torque,
                        trial_command,
                        trial_frequency,
                    )
                )

        aligned_time, vel, accel, torque, aligned_command, aligned_frequency = (
            np.concatenate(parts) for parts in zip(*aligned_parts)
        )
        train = np.arange(len(torque)) % 5 != 0
        validation = ~train

        if friction_fit is not None:
            damping = float(friction_fit["damping_nm_s_rad"])
            coulomb = float(friction_fit["coulomb_friction_nm"])
            bias = float(friction_fit["torque_bias_nm"])
            friction = (
                damping * vel
                + coulomb * np.tanh(vel / friction_velocity_scale)
                + bias
            )
            residual = torque - friction
            denominator = float(accel[train] @ accel[train])
            if denominator <= 0:
                continue
            inertia = float(accel[train] @ residual[train] / denominator)
            predicted = inertia * accel + friction
            coefficients = (inertia, damping, coulomb, bias)
            model_source = "friction plateaus fixed; inertia fitted from chirp"
        else:
            design = np.column_stack(
                [
                    accel,
                    vel,
                    np.tanh(vel / friction_velocity_scale),
                    np.ones_like(vel),
                ]
            )
            fit_result = lsq_linear(
                design[train],
                torque[train],
                bounds=(
                    [0.0, 0.0, 0.0, -np.inf],
                    [np.inf, np.inf, np.inf, np.inf],
                ),
            )
            if not fit_result.success:
                continue
            fitted = fit_result.x
            predicted = design @ fitted
            coefficients = tuple(float(value) for value in fitted)
            model_source = "all parameters fitted jointly from chirp"

        train_rmse = float(
            np.sqrt(np.mean((torque[train] - predicted[train]) ** 2))
        )
        validation_rmse = float(
            np.sqrt(np.mean((torque[validation] - predicted[validation]) ** 2))
        )
        if coefficients[0] <= 0:
            continue
        candidate = (
            train_rmse,
            validation_rmse,
            lag,
            coefficients,
            predicted,
            torque,
            vel,
            accel,
            aligned_time,
            aligned_command,
            aligned_frequency,
            validation,
            model_source,
        )
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        raise IdentificationError(
            "Chirp fit did not produce a positive inertia estimate"
        )

    (
        train_rmse,
        validation_rmse,
        lag,
        coefficients,
        predicted,
        torque,
        velocity,
        acceleration,
        aligned_time,
        aligned_command,
        aligned_frequency,
        validation,
        model_source,
    ) = best
    inertia, damping, coulomb, bias = coefficients
    result = {
        "inertia_kg_m2": inertia,
        "damping_nm_s_rad": damping,
        "coulomb_friction_nm": coulomb,
        "torque_bias_nm": bias,
        "friction_velocity_scale_rad_s": friction_velocity_scale,
        "model_source": model_source,
        "savgol_window_samples": windows,
        "estimated_sample_rate_hz": sample_rate_hz,
        "selected_torque_lag_samples": lag,
        "selected_torque_lag_s": lag * dt_s,
        "training_rmse_nm": train_rmse,
        "fit_rmse_nm": float(np.sqrt(np.mean((torque - predicted) ** 2))),
        "validation_rmse_nm": validation_rmse,
        "validation_r_squared": coefficient_of_determination(
            torque[validation], predicted[validation]
        ),
    }
    plot_data = (
        aligned_time,
        aligned_command,
        torque,
        predicted,
        velocity,
        acceleration,
        aligned_frequency,
    )
    return result, plot_data


def save_plots(
    output_dir: Path,
    rows: Sequence[dict[str, str]],
    stiction_fit: dict[str, object] | None,
    friction_plot_data: object | None,
    friction_fit: dict[str, object] | None,
    chirp_plot_data: object | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if stiction_fit is not None:
        stiction_rows = [row for row in rows if row["experiment"] == "stiction"]
        figure, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
        for trial in sorted({row["trial"] for row in stiction_rows}, key=int):
            trial_rows = [row for row in stiction_rows if row["trial"] == trial]
            time_values = numeric(trial_rows, "session_time_s")
            time_values -= time_values[0]
            command_line = axes[0].plot(
                time_values,
                numeric(trial_rows, "command_torque_nm"),
                label=f"trial {trial}",
            )[0]
            axes[0].plot(
                time_values,
                numeric(trial_rows, "measured_torque_nm"),
                color=command_line.get_color(),
                linestyle="--",
                alpha=0.7,
            )
            axes[1].plot(
                time_values,
                numeric(trial_rows, "velocity_rad_s"),
                label=f"trial {trial}",
            )
        axes[0].set_ylabel("Command torque [N m]")
        axes[1].set_ylabel("Velocity [rad/s]")
        axes[1].set_xlabel("Ramp time [s]")
        axes[0].grid(True, alpha=0.3)
        axes[1].grid(True, alpha=0.3)
        axes[0].legend(ncol=2, fontsize=8)
        figure.suptitle("Static friction torque ramps")
        figure.tight_layout()
        figure.savefig(output_dir / "stiction_ramps.png", dpi=160)
        plt.close(figure)

    if friction_plot_data is not None and friction_fit is not None:
        velocity, torque, _ = friction_plot_data
        velocity_curve = np.linspace(float(np.min(velocity)), float(np.max(velocity)), 400)
        predicted_curve = (
            float(friction_fit["damping_nm_s_rad"]) * velocity_curve
            + float(friction_fit["coulomb_friction_nm"])
            * np.tanh(
                velocity_curve
                / float(friction_fit["friction_velocity_scale_rad_s"])
            )
            + float(friction_fit["torque_bias_nm"])
        )
        figure, axis = plt.subplots(figsize=(9, 6))
        axis.scatter(velocity, torque, color="tab:blue", label="plateau means")
        axis.plot(velocity_curve, predicted_curve, color="tab:orange", label="fit")
        axis.set_xlabel("Velocity [rad/s]")
        axis.set_ylabel("Measured motor torque [N m]")
        axis.set_title("Coulomb friction and viscous damping fit")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "friction_fit.png", dpi=160)
        plt.close(figure)

    if chirp_plot_data is not None:
        (
            time_s,
            command_torque,
            measured_torque,
            predicted_torque,
            velocity,
            acceleration,
            frequency,
        ) = chirp_plot_data
        time_s = time_s - time_s[0]
        figure, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 10))
        axes[0].plot(time_s, command_torque, label="encoded command", alpha=0.7)
        axes[0].plot(time_s, measured_torque, label="measured q-current torque")
        axes[0].plot(time_s, predicted_torque, label="model fit", linewidth=1.2)
        axes[0].set_ylabel("Torque [N m]")
        axes[0].legend(ncol=3, fontsize=8)
        axes[1].plot(time_s, velocity)
        axes[1].set_ylabel("Velocity [rad/s]")
        axes[2].plot(time_s, acceleration)
        axes[2].set_ylabel("Acceleration [rad/s²]")
        axes[3].plot(time_s, frequency)
        axes[3].set_ylabel("Frequency [Hz]")
        axes[3].set_xlabel("Chirp time [s]")
        for axis in axes:
            axis.grid(True, alpha=0.3)
        figure.suptitle("Chirp response and free-rotor model fit")
        figure.tight_layout()
        figure.savefig(output_dir / "chirp_model_fit.png", dpi=160)
        plt.close(figure)


def analyze_csv(
    csv_path: Path,
    output_dir: Path,
    *,
    friction_velocity_scale: float,
    smooth_window_s: float,
) -> dict[str, object]:
    if friction_velocity_scale <= 0 or smooth_window_s <= 0:
        raise IdentificationError(
            "Friction velocity scale and smoothing window must be positive"
        )
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise IdentificationError(
            "Analysis requires numpy, scipy, and matplotlib: "
            "pip install numpy scipy matplotlib"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_numeric_rows(csv_path)
    metadata_path = csv_path.parent / "metadata.json"
    movement_threshold = 0.5
    movement_confirmation_samples = 1
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_arguments = metadata.get("arguments", {})
        movement_threshold = float(
            metadata_arguments.get("movement_threshold_rad_s", movement_threshold)
        )
        movement_confirmation_samples = max(
            1,
            math.ceil(
                float(metadata_arguments.get("movement_confirm_s", 0.0))
                * float(metadata_arguments.get("rate_hz", 1.0))
            ),
        )

    stiction_fit = analyze_stiction(
        rows, movement_threshold, movement_confirmation_samples
    )
    friction_fit, friction_plot_data = analyze_friction(
        rows, friction_velocity_scale
    )
    chirp_fit, chirp_plot_data = analyze_chirp(
        rows,
        friction_fit,
        friction_velocity_scale,
        smooth_window_s,
    )
    results = {
        "source_csv": str(csv_path.resolve()),
        "analyzed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stiction": stiction_fit,
        "steady_state_friction": friction_fit,
        "dynamic_model": chirp_fit,
    }
    results_path = output_dir / "identified_parameters.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_plots(
        output_dir,
        rows,
        stiction_fit,
        friction_plot_data,
        friction_fit,
        chirp_plot_data,
    )

    print("\nAnalysis complete")
    if stiction_fit is not None:
        positive = stiction_fit["positive"]["median_breakaway_torque_nm"]
        negative = stiction_fit["negative"]["median_breakaway_torque_nm"]
        print(f"  median breakaway: {positive} Nm / {negative} Nm")
    if friction_fit is not None:
        print(
            "  steady friction: "
            f"Coulomb={friction_fit['coulomb_friction_nm']:.6g} Nm, "
            f"damping={friction_fit['damping_nm_s_rad']:.6g} Nms/rad"
        )
    if chirp_fit is not None:
        print(
            f"  rotor inertia: {chirp_fit['inertia_kg_m2']:.6g} kg m^2 "
            f"(validation R^2={chirp_fit['validation_r_squared']:.4f})"
        )
    print(f"  parameters: {results_path}")
    return results


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--actuator", choices=ACTUATORS, required=True)
    parser.add_argument("--port", required=True, help="Waveshare serial port, e.g. COM11")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("system_identification_results"),
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=("all", "stiction", "friction", "chirp"),
        default=("all",),
    )
    parser.add_argument("--rate-hz", type=float, default=200.0)
    parser.add_argument("--feedback-timeout-s", type=float, default=0.05)
    parser.add_argument("--max-velocity-rad-s", type=float)
    parser.add_argument("--movement-threshold-rad-s", type=float, default=0.5)
    parser.add_argument("--movement-confirm-s", type=float, default=0.03)
    parser.add_argument("--brake-timeout-s", type=float, default=20.0)
    parser.add_argument("--rest-confirm-s", type=float, default=1.0)
    parser.add_argument("--zero-torque-settle-s", type=float, default=1.0)
    parser.add_argument("--velocity-kd", type=float)
    parser.add_argument("--stiction-repeats", type=int, default=3)
    parser.add_argument("--stiction-max-nm", type=float)
    parser.add_argument("--stiction-ramp-nm-s", type=float)
    parser.add_argument("--friction-speeds", type=parse_positive_speeds)
    parser.add_argument("--friction-settle-s", type=float, default=10.0)
    parser.add_argument("--friction-sample-s", type=float, default=3.0)
    parser.add_argument("--chirp-repeats", type=int, default=2)
    parser.add_argument("--chirp-amplitude-nm", type=float)
    parser.add_argument("--chirp-start-hz", type=float, default=0.5)
    parser.add_argument("--chirp-end-hz", type=float, default=15.0)
    parser.add_argument("--chirp-duration-s", type=float, default=30.0)
    parser.add_argument("--chirp-kind", choices=("linear", "log"), default="log")
    parser.add_argument("--friction-velocity-scale", type=float, default=0.5)
    parser.add_argument("--smooth-window-s", type=float, default=0.075)
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the typed RUN confirmation",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Identify free-spinning actuator friction, damping, and inertia."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser(
        "run", help="Acquire data and analyze the completed experiment"
    )
    add_run_arguments(run_parser)

    analyze_parser = subparsers.add_parser(
        "analyze", help="Reanalyze an existing raw_samples.csv file"
    )
    analyze_parser.add_argument("csv_path", type=Path)
    analyze_parser.add_argument("--output-dir", type=Path)
    analyze_parser.add_argument("--friction-velocity-scale", type=float, default=0.5)
    analyze_parser.add_argument("--smooth-window-s", type=float, default=0.075)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            output = run_acquisition(args)
            print(f"\nFinished. Results are in {output}")
        else:
            output_dir = args.output_dir or args.csv_path.parent
            analyze_csv(
                args.csv_path,
                output_dir,
                friction_velocity_scale=args.friction_velocity_scale,
                smooth_window_s=args.smooth_window_s,
            )
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except (IdentificationError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
