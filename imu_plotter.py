"""
imu_plotter.py  —  Real-time UART plotter
==========================================
Handles UART formats automatically:

  IMU-only test (IMU_Only_Test):
    <Hz>Hz  th=<rad>  th_enc=<rad>  th_a=<rad>  gz=<r/s>  |a|=<g>

  LQR controller (IMU_Test_rig):
    <Hz>Hz  theta=<rad>  err=<rad>  gz=<r/s> r/s  fw=<r/s> r/s  Tcmd=<Nm> Nm

  Dual-axis LQR (Jumping_Sensors):
    <Hz>Hz  thx=<rad> thy=<rad> gx=<r/s> gy=<r/s> fwx=<r/s> fwy=<r/s> tx=<Nm> ty=<Nm> ...

Usage:
    python imu_plotter.py [PORT] [BAUD]
    python imu_plotter.py --can [PORT] [BAUD]
    python imu_plotter.py COM3 115200
Defaults: first available COM port, 115200 baud.
"""

import sys
import re
import collections
import threading
import time
import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import RadioButtons, Button, TextBox

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW = 500
UART_BAUD = 115200
CAN_BAUD  = 2_000_000
BAUD   = UART_BAUD
CAN_SENSOR_FB_ID      = 0x104
CAN_SENSOR_AUX_FB_ID  = 0x105
CAN_FLYWHEEL_X_FB_ID  = 0x101
CAN_FLYWHEEL_Y_FB_ID  = 0x102
CAN_FLYWHEEL_X_CMD_ID = 0x001
CAN_FLYWHEEL_Y_CMD_ID = 0x002
CAN_FLYWHEEL_X_CTRL_ID = 0x201
CAN_FLYWHEEL_Y_CTRL_ID = 0x202
CAN_ANGLE_MIN = -3.14159265
CAN_ANGLE_MAX = 3.14159265
CAN_RATE_MIN  = -1500.0
CAN_RATE_MAX  = 1500.0
CAN_TRQ_MIN   = -1.0
CAN_TRQ_MAX   = 1.0
CMD_MOTOR_START = 0x01
CMD_MOTOR_STOP  = 0x02
CMD_MOTOR_ALIGN = 0x03
CTRL_RETRY_COUNT = 3
CTRL_RETRY_DELAY_S = 0.01
ALIGN_STOP_SETTLE_S = 0.05
PLOT_SAMPLE_PERIOD_S = 0.02

# ── Regex patterns ────────────────────────────────────────────────────────────
# IMU-only test format: th=  th_enc=  th_a=  gz=  |a|=
RE_IMU_ONLY = re.compile(
    r"([\d.]+)Hz\s+th=([-\d.]+)\s+th_enc=([-\d.]+)\s+th_a=([-\d.]+)"
    r"\s+gz=([-\d.]+)\s+\|a\|=([-\d.]+)(?:\s+ac=(\d+))?"
)

# LQR controller format: theta=  th_enc=  err=  gz=  fw=  Tcmd=  ac=
RE_LQR = re.compile(
    r"([\d.]+)Hz\s+theta=([-\d.]+)\s+th_enc=([-\d.]+)\s+err=([-\d.]+)\s+"
    r"gz=([-\d.]+)\s+r/s\s+fw=([-\d.]+)\s+r/s\s+Tcmd=([-\d.]+)(?:\s+Nm\s+ac=(\d+))?"
)

RE_DUAL_LQR = re.compile(
    r"([\d.]+)Hz\s+thx=([-\d.]+)\s+thy=([-\d.]+)\s+gx=([-\d.]+)\s+gy=([-\d.]+)\s+"
    r"fwx=([-\d.]+)\s+fwy=([-\d.]+)\s+tx=([-\d.]+)\s+ty=([-\d.]+)(?:\s+tof=(\d+)mm\s+\|a\|=([-\d.]+))?"
)

RE_MOTOR = re.compile(
    r"Motor\[(\d+)\].*pos=([-\d.]+)\s+rad\s+vel=([-\d.]+)\s+r/s\s+trq=([-\d.]+)"
)

# ── Shared data ───────────────────────────────────────────────────────────────
def _dq(): return collections.deque([0.0] * WINDOW, maxlen=WINDOW)

data = {
    "hz":      _dq(),
    "theta":   _dq(),   # gyro+accel estimate (both modes)
    "th_enc":  _dq(),   # encoder angle (IMU-only mode)
    "th_a":    _dq(),   # accel-only angle (IMU-only mode)
    "gz":      _dq(),
    "accel_mag": _dq(),
    "err":     _dq(),   # LQR mode
    "fw":      _dq(),   # LQR mode
    "Tcmd":    _dq(),   # LQR mode
    "thx":     _dq(),   # Dual-axis LQR mode
    "thy":     _dq(),
    "gx":      _dq(),
    "gy":      _dq(),
    "fwx":     _dq(),
    "fwy":     _dq(),
    "tx":      _dq(),
    "ty":      _dq(),
    "x_state": _dq(),
    "y_state": _dq(),
    "tof_mm":  _dq(),
    "mot_pos": _dq(),
    "mot_vel": _dq(),
    "mot_trq": _dq(),
    "ac":      _dq(),   # accel blend count per print interval (0 = gated out)
}
lock = threading.Lock()
mode = {"val": "unknown"}  # "imu_only", "lqr", or "dual_lqr"
axis_select = {"val": "x"}
STATE_NAMES = {
    0: "IDLE",
    1: "RUN",
    2: "STOP",
    3: "FAULT",
    4: "FAULT_OVER",
}

def _u16_to_float(u, x_min, x_max):
    return x_min + (float(u) * (x_max - x_min) / 65535.0)

def _u12_to_float(u, x_min, x_max):
    return x_min + (float(u) * (x_max - x_min) / 4095.0)

def _checksum(data):
    return sum(data) & 0xFF

def make_can_init_frame():
    core = [0x12, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00]
    return bytearray([0xAA, 0x55] + core + [_checksum(core)])

def pack_ctrl_frame(cmd_byte, ctrl_id):
    id_bytes = [ctrl_id & 0xFF, (ctrl_id >> 8) & 0xFF, 0x00, 0x00]
    core = [0x01, 0x01, 0x00] + id_bytes + [0x01] + [cmd_byte] + [0x00] * 7 + [0x00]
    return bytearray([0xAA, 0x55] + core + [_checksum(core)])

# ── Reader thread + parser ─────────────────────────────────────────────────────
conn = {
    "thread": None,
    "stop_event": None,
    "serial": None,
    "connected": False,
    "protocol": "UART",
}
tx_queue = collections.deque()
tx_lock = threading.Lock()
can_stats = {"frames": 0, "last_id": None, "error": ""}
sample_times = {
    CAN_FLYWHEEL_X_FB_ID: 0.0,
    CAN_FLYWHEEL_Y_FB_ID: 0.0,
    CAN_FLYWHEEL_X_CMD_ID: 0.0,
    CAN_FLYWHEEL_Y_CMD_ID: 0.0,
}

def _parse_uart_line(raw):
    m = RE_IMU_ONLY.search(raw)
    if m:
        groups = m.groups()
        hz, th, th_enc, th_a, gz, amag = (float(x) for x in groups[:6])
        ac = int(groups[6]) if groups[6] is not None else 0
        with lock:
            mode["val"] = "imu_only"
            data["hz"].append(hz)
            data["theta"].append(th)
            data["th_enc"].append(th_enc)
            data["th_a"].append(th_a)
            data["gz"].append(gz)
            data["accel_mag"].append(amag)
            data["ac"].append(ac)
        return

    m = RE_LQR.search(raw)
    if m:
        groups = m.groups()
        hz, theta, th_enc, err, gz, fw, Tcmd = (float(x) for x in groups[:7])
        ac = int(groups[7]) if groups[7] is not None else 0
        with lock:
            mode["val"] = "lqr"
            data["hz"].append(hz)
            data["theta"].append(theta)
            data["th_enc"].append(th_enc)
            data["err"].append(err)
            data["gz"].append(gz)
            data["fw"].append(fw)
            data["Tcmd"].append(Tcmd)
            data["ac"].append(ac)
        return

    m = RE_DUAL_LQR.search(raw)
    if m:
        groups = m.groups()
        hz, thx, thy, gx, gy, fwx, fwy, tx, ty = (float(x) for x in groups[:9])
        tof_mm = float(groups[9]) if groups[9] is not None else 0.0
        amag = float(groups[10]) if groups[10] is not None else 0.0
        with lock:
            mode["val"] = "dual_lqr"
            data["hz"].append(hz)
            data["thx"].append(thx)
            data["thy"].append(thy)
            data["gx"].append(gx)
            data["gy"].append(gy)
            data["fwx"].append(fwx)
            data["fwy"].append(fwy)
            data["tx"].append(tx)
            data["ty"].append(ty)
            data["tof_mm"].append(tof_mm)
            data["accel_mag"].append(amag)
        return

    m = RE_MOTOR.search(raw)
    if m:
        _, pos, vel, trq = (float(x) for x in m.groups())
        with lock:
            data["mot_pos"].append(pos)
            data["mot_vel"].append(vel)
            data["mot_trq"].append(trq)

def _parse_can_frame(can_id, d):
    if can_id == CAN_SENSOR_FB_ID:
        thx_u = (d[2] << 8) | d[3]
        gx_u = (d[4] << 8) | d[5]
        with lock:
            mode["val"] = "dual_lqr"
            data["thx"].append(_u16_to_float(thx_u, CAN_ANGLE_MIN, CAN_ANGLE_MAX))
            data["gx"].append(_u16_to_float(gx_u, CAN_RATE_MIN, CAN_RATE_MAX))
    elif can_id == CAN_SENSOR_AUX_FB_ID:
        thy_u = (d[2] << 8) | d[3]
        gy_u = (d[4] << 8) | d[5]
        with lock:
            mode["val"] = "dual_lqr"
            data["thy"].append(_u16_to_float(thy_u, CAN_ANGLE_MIN, CAN_ANGLE_MAX))
            data["gy"].append(_u16_to_float(gy_u, CAN_RATE_MIN, CAN_RATE_MAX))
    elif can_id == CAN_FLYWHEEL_X_FB_ID:
        vx_u = (d[4] << 8) | d[5]
        now = time.monotonic()
        if (now - sample_times[can_id]) >= PLOT_SAMPLE_PERIOD_S:
            sample_times[can_id] = now
            with lock:
                data["fwx"].append(_u16_to_float(vx_u, CAN_RATE_MIN, CAN_RATE_MAX))
                data["x_state"].append(float((d[0] >> 4) & 0x0F))
    elif can_id == CAN_FLYWHEEL_Y_FB_ID:
        vy_u = (d[4] << 8) | d[5]
        now = time.monotonic()
        if (now - sample_times[can_id]) >= PLOT_SAMPLE_PERIOD_S:
            sample_times[can_id] = now
            with lock:
                data["fwy"].append(_u16_to_float(vy_u, CAN_RATE_MIN, CAN_RATE_MAX))
                data["y_state"].append(float((d[0] >> 4) & 0x0F))
    elif can_id == CAN_FLYWHEEL_X_CMD_ID:
        tx_u12 = ((d[6] & 0x0F) << 8) | d[7]
        now = time.monotonic()
        if (now - sample_times[can_id]) >= PLOT_SAMPLE_PERIOD_S:
            sample_times[can_id] = now
            with lock:
                data["tx"].append(_u12_to_float(tx_u12, CAN_TRQ_MIN, CAN_TRQ_MAX))
    elif can_id == CAN_FLYWHEEL_Y_CMD_ID:
        ty_u12 = ((d[6] & 0x0F) << 8) | d[7]
        now = time.monotonic()
        if (now - sample_times[can_id]) >= PLOT_SAMPLE_PERIOD_S:
            sample_times[can_id] = now
            with lock:
                data["ty"].append(_u12_to_float(ty_u12, CAN_TRQ_MIN, CAN_TRQ_MAX))

def serial_reader_loop(ser, protocol, stop_event):
    if protocol == "CAN":
        buf = bytearray()
        while not stop_event.is_set():
            try:
                now = time.monotonic()
                with tx_lock:
                    while tx_queue and tx_queue[0][0] <= now:
                        _, frame = tx_queue.popleft()
                        ser.write(frame)

                waiting = ser.in_waiting
                if waiting:
                    buf.extend(ser.read(waiting))
                else:
                    time.sleep(0.001)
            except (serial.SerialException, OSError) as exc:
                can_stats["error"] = str(exc)
                break

            while len(buf) >= 13:
                if not (buf[0] == 0xAA and buf[1] == 0xC8):
                    del buf[0]
                    continue
                if buf[12] != 0x55:
                    del buf[0]
                    continue
                frame = bytes(buf[:13])
                del buf[:13]
                can_id = frame[2] | (frame[3] << 8)
                can_stats["frames"] += 1
                can_stats["last_id"] = can_id
                _parse_can_frame(can_id, frame[4:12])
    else:
        while not stop_event.is_set():
            try:
                raw = ser.readline().decode("utf-8", errors="replace").strip()
            except Exception:
                continue
            if raw:
                _parse_uart_line(raw)

# ── Plot setup ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle("IMU Plotter", fontsize=13)

ax_angle, ax_rate, ax_extra = axes

# Angle plot — theta (IMU estimate) + encoder on same axes
ax_angle.set_ylabel("rad")
ax_angle.set_title("Angle: IMU estimate vs Encoder  (green bars = accel sync active)")
ax_angle.axhline(0, color="gray", lw=0.5)
line_theta,  = ax_angle.plot([], [], label="th (IMU)",  lw=1.5, color="tab:blue")
line_th_enc, = ax_angle.plot([], [], label="th_enc",    lw=1.5, color="tab:red",    linestyle="--")
line_th_a,   = ax_angle.plot([], [], label="th_a (acc)",lw=1,   color="tab:green",  linestyle=":")

# Twin axis for accel-sync activity bar
ax_ac = ax_angle.twinx()
ax_ac.set_ylabel("sync count", fontsize=7, color="tab:green")
ax_ac.tick_params(axis="y", labelcolor="tab:green", labelsize=7)
ax_ac.set_ylim(0, 15)
line_ac, = ax_ac.plot([], [], lw=1.5, color="tab:green", alpha=0.5)

ax_angle.legend(loc="upper left", fontsize=8)

# Rate plot
ax_rate.set_ylabel("rad/s")
ax_rate.set_title("Angular Rate (gz)")
ax_rate.axhline(0, color="gray", lw=0.5)
line_gz, = ax_rate.plot([], [], label="gz", lw=1, color="tab:orange")
ax_rate.legend(loc="upper left", fontsize=8)

# Extra plot — changes content based on detected mode
ax_extra.set_ylabel("(mode dependent)")
ax_extra.set_title("Flywheel & Torque (LQR) / Accel magnitude (IMU-only)")
ax_extra.axhline(0, color="gray", lw=0.5)
line_fw,      = ax_extra.plot([], [], label="fw (r/s)",   lw=1,   color="tab:blue")
line_Tcmd,    = ax_extra.plot([], [], label="Tcmd (Nm)",  lw=1.5, color="black")
line_amag,    = ax_extra.plot([], [], label="|a| (g)",    lw=1,   color="tab:purple")
ax_extra.legend(loc="upper left", fontsize=8)
ax_extra.set_xlabel("Samples")

rax = fig.add_axes([0.86, 0.72, 0.12, 0.16])
radio_axis = RadioButtons(rax, ("X axis", "Y axis"), active=0)
rax.set_title("Dual LQR\naxis", fontsize=8)
def _on_axis_change(label):
    axis_select["val"] = "x" if label.startswith("X") else "y"
radio_axis.on_clicked(_on_axis_change)

status_text = fig.text(0.86, 0.69, "Disconnected", fontsize=8, color="tab:red")
can_stats_text = fig.text(0.86, 0.675, "RX: 0  ID: --", fontsize=7, color="tab:gray")

rax_mode = fig.add_axes([0.86, 0.54, 0.12, 0.13])
radio_mode = RadioButtons(rax_mode, ("UART", "CAN"), active=0)
rax_mode.set_title("Source", fontsize=8)
def _on_mode_change(label):
    if label == "CAN":
        tb_baud.set_val(str(CAN_BAUD))
    else:
        tb_baud.set_val(str(UART_BAUD))
radio_mode.on_clicked(_on_mode_change)

ax_port = fig.add_axes([0.86, 0.48, 0.12, 0.05])
default_ports = [p.device for p in serial.tools.list_ports.comports()]
default_port = default_ports[0] if default_ports else ""
for _p in serial.tools.list_ports.comports():
    if "CH340" in (_p.description or ""):
        default_port = _p.device
        break
tb_port = TextBox(ax_port, "Port", initial=default_port)
tb_port.text_disp.set_fontsize(8)

ax_baud = fig.add_axes([0.86, 0.42, 0.12, 0.05])
tb_baud = TextBox(ax_baud, "Baud", initial=str(BAUD))
tb_baud.text_disp.set_fontsize(8)

ax_connect = fig.add_axes([0.86, 0.35, 0.055, 0.05])
btn_connect = Button(ax_connect, "Connect")
ax_disconnect = fig.add_axes([0.925, 0.35, 0.055, 0.05])
btn_disconnect = Button(ax_disconnect, "Stop")

ax_refresh = fig.add_axes([0.86, 0.29, 0.12, 0.05])
btn_refresh = Button(ax_refresh, "Refresh Port")

fig.text(0.86, 0.25, "X DOF", fontsize=8)
ax_x_enable = fig.add_axes([0.86, 0.21, 0.038, 0.045])
btn_x_enable = Button(ax_x_enable, "On")
ax_x_disable = fig.add_axes([0.902, 0.21, 0.038, 0.045])
btn_x_disable = Button(ax_x_disable, "Off")
ax_x_align = fig.add_axes([0.944, 0.21, 0.038, 0.045])
btn_x_align = Button(ax_x_align, "Align")
state_x_text = fig.text(0.86, 0.235, "state: --", fontsize=8, color="tab:blue")

fig.text(0.86, 0.16, "Y DOF", fontsize=8)
ax_y_enable = fig.add_axes([0.86, 0.12, 0.038, 0.045])
btn_y_enable = Button(ax_y_enable, "On")
ax_y_disable = fig.add_axes([0.902, 0.12, 0.038, 0.045])
btn_y_disable = Button(ax_y_disable, "Off")
ax_y_align = fig.add_axes([0.944, 0.12, 0.038, 0.045])
btn_y_align = Button(ax_y_align, "Align")
state_y_text = fig.text(0.86, 0.145, "state: --", fontsize=8, color="tab:blue")

plt.tight_layout(rect=[0.0, 0.0, 0.84, 1.0])
xs = list(range(WINDOW))

def _set_status(msg, ok=False):
    status_text.set_text(msg)
    status_text.set_color("tab:green" if ok else "tab:red")
    fig.canvas.draw_idle()

def _queue_frames(frames, label):
    due = time.monotonic()
    scheduled = []
    for frame, delay_s in frames:
        scheduled.append((due, frame))
        due += delay_s
    with tx_lock:
        tx_queue.extend(scheduled)
        ordered = sorted(tx_queue, key=lambda item: item[0])
        tx_queue.clear()
        tx_queue.extend(ordered)
    _set_status(f"Queued {label}", ok=True)

def disconnect_reader(_event=None):
    if conn["connected"]:
        conn["stop_event"].set()
        if conn["thread"] is not None:
            conn["thread"].join(timeout=0.5)
        if conn["serial"] is not None:
            try:
                conn["serial"].close()
            except Exception:
                pass
    conn["thread"] = None
    conn["stop_event"] = None
    conn["serial"] = None
    conn["connected"] = False
    with tx_lock:
        tx_queue.clear()
    _set_status("Disconnected", ok=False)

def connect_reader(_event=None):
    disconnect_reader()
    port = tb_port.text.strip()
    if not port:
        _set_status("No port selected", ok=False)
        return
    protocol = radio_mode.value_selected
    try:
        baud = int(tb_baud.text.strip())
    except ValueError:
        _set_status("Bad baud", ok=False)
        return
    timeout = 0.01 if protocol == "CAN" else 1.0
    try:
        ser = serial.Serial(port, baud, timeout=timeout)
    except serial.SerialException as e:
        _set_status(f"Open failed: {e}", ok=False)
        return

    if protocol == "CAN":
        try:
            ser.write(make_can_init_frame())
            time.sleep(0.2)
        except Exception as e:
            try:
                ser.close()
            except Exception:
                pass
            _set_status(f"Init failed: {e}", ok=False)
            return

    stop_event = threading.Event()
    t = threading.Thread(target=serial_reader_loop, args=(ser, protocol, stop_event), daemon=True)
    conn["serial"] = ser
    conn["stop_event"] = stop_event
    conn["thread"] = t
    conn["connected"] = True
    conn["protocol"] = protocol
    can_stats["frames"] = 0
    can_stats["last_id"] = None
    can_stats["error"] = ""
    t.start()
    _set_status(f"Connected {protocol} {port}@{baud}", ok=True)

def refresh_port(_event=None):
    ports = serial.tools.list_ports.comports()
    names = [p.device for p in ports]
    ch340 = [p.device for p in ports if "CH340" in (p.description or "")]
    if ch340:
        tb_port.set_val(ch340[0])
        _set_status(f"Port {ch340[0]}", ok=False)
    elif names:
        tb_port.set_val(names[0])
        _set_status(f"Port {names[0]}", ok=False)
    else:
        _set_status("No COM ports", ok=False)

def send_axis_ctrl(ctrl_id, cmd_byte, label):
    if not conn["connected"] or conn["protocol"] != "CAN" or conn["serial"] is None:
        _set_status(f"{label}: connect CAN first", ok=False)
        return
    frame = pack_ctrl_frame(cmd_byte, ctrl_id)
    frames = [(frame, CTRL_RETRY_DELAY_S) for _ in range(CTRL_RETRY_COUNT)]
    if frames:
        frames[-1] = (frames[-1][0], 0.0)
    _queue_frames(frames, label)

def send_axis_align(ctrl_id, axis_label):
    if not conn["connected"] or conn["protocol"] != "CAN" or conn["serial"] is None:
        _set_status(f"{axis_label} align: connect CAN first", ok=False)
        return
    stop_frame = pack_ctrl_frame(CMD_MOTOR_STOP, ctrl_id)
    align_frame = pack_ctrl_frame(CMD_MOTOR_ALIGN, ctrl_id)
    frames = [(stop_frame, CTRL_RETRY_DELAY_S) for _ in range(CTRL_RETRY_COUNT)]
    frames.append((stop_frame, ALIGN_STOP_SETTLE_S))
    frames.extend((align_frame, CTRL_RETRY_DELAY_S) for _ in range(CTRL_RETRY_COUNT))
    if frames:
        frames[-1] = (frames[-1][0], 0.0)
    _queue_frames(frames, f"{axis_label} align sequence")

btn_connect.on_clicked(connect_reader)
btn_disconnect.on_clicked(disconnect_reader)
btn_refresh.on_clicked(refresh_port)
btn_x_enable.on_clicked(lambda _e: send_axis_ctrl(CAN_FLYWHEEL_X_CTRL_ID, CMD_MOTOR_START, "X enable"))
btn_x_disable.on_clicked(lambda _e: send_axis_ctrl(CAN_FLYWHEEL_X_CTRL_ID, CMD_MOTOR_STOP, "X disable"))
btn_x_align.on_clicked(lambda _e: send_axis_align(CAN_FLYWHEEL_X_CTRL_ID, "X"))
btn_y_enable.on_clicked(lambda _e: send_axis_ctrl(CAN_FLYWHEEL_Y_CTRL_ID, CMD_MOTOR_START, "Y enable"))
btn_y_disable.on_clicked(lambda _e: send_axis_ctrl(CAN_FLYWHEEL_Y_CTRL_ID, CMD_MOTOR_STOP, "Y disable"))
btn_y_align.on_clicked(lambda _e: send_axis_align(CAN_FLYWHEEL_Y_CTRL_ID, "Y"))

def _on_close(_event):
    disconnect_reader()

fig.canvas.mpl_connect("close_event", _on_close)

def update(_frame):
    with lock:
        snap = {k: list(v) for k, v in data.items()}
        cur_mode = mode["val"]

    x_state = int(snap["x_state"][-1]) if snap["x_state"] else -1
    y_state = int(snap["y_state"][-1]) if snap["y_state"] else -1
    state_x_text.set_text(f"state: {STATE_NAMES.get(x_state, '--')}")
    state_y_text.set_text(f"state: {STATE_NAMES.get(y_state, '--')}")
    last_id = can_stats["last_id"]
    last_id_text = "--" if last_id is None else f"0x{last_id:03X}"
    can_stats_text.set_text(f"RX: {can_stats['frames']}  ID: {last_id_text}")
    if can_stats["error"]:
        status_text.set_text(f"I/O error: {can_stats['error']}")
        status_text.set_color("tab:red")

    line_ac.set_data(xs,     snap["ac"])

    if cur_mode == "imu_only":
        line_theta.set_data(xs,  snap["theta"])
        line_th_enc.set_data(xs, snap["th_enc"])
        line_gz.set_data(xs,     snap["gz"])
        line_th_a.set_data(xs,  snap["th_a"])
        line_amag.set_data(xs,  snap["accel_mag"])
        line_fw.set_data(xs,    [0.0] * WINDOW)
        line_Tcmd.set_data(xs,  [0.0] * WINDOW)
        line_theta.set_label("th (IMU)")
        line_gz.set_label("gz")
        ax_angle.set_title("Angle: IMU estimate vs Encoder  (green bars = accel sync active)")
        ax_rate.set_title("Angular Rate (gz)")
        ax_extra.set_title("|a| magnitude (1.0 = static)")
    elif cur_mode == "dual_lqr":
        ax_key = axis_select["val"]
        theta_key = "thx" if ax_key == "x" else "thy"
        gyro_key = "gx" if ax_key == "x" else "gy"
        fw_key = "fwx" if ax_key == "x" else "fwy"
        cmd_key = "tx" if ax_key == "x" else "ty"
        line_theta.set_data(xs,  snap[theta_key])
        line_th_enc.set_data(xs, [0.0] * WINDOW)
        line_th_a.set_data(xs,   [0.0] * WINDOW)
        line_gz.set_data(xs,     snap[gyro_key])
        line_fw.set_data(xs,     snap[fw_key])
        line_Tcmd.set_data(xs,   snap[cmd_key])
        line_amag.set_data(xs,   snap["accel_mag"])
        line_theta.set_label(f"th{ax_key}")
        line_gz.set_label(f"g{ax_key}")
        ax_angle.set_title(f"Estimated Angle (Dual LQR, {ax_key.upper()} axis)")
        ax_rate.set_title(f"Angular Rate (g{ax_key})")
        ax_extra.set_title(f"Flywheel Speed & Torque Command ({ax_key.upper()} axis)")
    else:
        line_theta.set_data(xs,  snap["theta"])
        line_th_enc.set_data(xs, snap["th_enc"])
        line_gz.set_data(xs,     snap["gz"])
        line_th_a.set_data(xs,  [0.0] * WINDOW)
        line_amag.set_data(xs,  [0.0] * WINDOW)
        line_fw.set_data(xs,    snap["fw"])
        line_Tcmd.set_data(xs,  snap["Tcmd"])
        line_theta.set_label("theta")
        line_gz.set_label("gz")
        ax_angle.set_title("Angle: IMU estimate vs Encoder  (green bars = accel sync active)")
        ax_rate.set_title("Angular Rate (gz)")
        ax_extra.set_title("Flywheel Speed & Torque Command")

    ax_angle.legend(loc="upper left", fontsize=8)
    ax_rate.legend(loc="upper left", fontsize=8)

    for ax in [ax_angle, ax_rate, ax_extra]:
        ax.relim()
        ax.autoscale_view()
    ax_ac.relim()
    ax_ac.autoscale_view()

    return (line_theta, line_th_enc, line_th_a, line_gz,
            line_fw, line_Tcmd, line_amag, line_ac)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a]
    can_mode = False
    if "--can" in args:
        can_mode = True
        args.remove("--can")

    if len(args) >= 1:
        port = args[0]
        tb_port.set_val(port)

    baud = int(args[1]) if len(args) >= 2 else BAUD
    tb_baud.set_val(str(baud))
    if can_mode:
        radio_mode.set_active(1)
    else:
        radio_mode.set_active(0)

    ani = animation.FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    plt.show()
