"""
Motor Control GUI
Sliders for all MIT-protocol command fields, live feedback display.
Requires: pyserial  (pip install pyserial)
"""

import tkinter as tk
from tkinter import ttk
import serial
import serial.tools.list_ports
import threading
import time
import math
import collections

# ── Protocol constants (must match STM32 firmware) ────────────────────────────
P_MIN,  P_MAX  = -math.pi, math.pi   # ±π rad (one full revolution)
V_MIN,  V_MAX  = -1500, 1500
KP_MIN, KP_MAX =   0.0, 500.0
KD_MIN, KD_MAX =   0.0, 15.0

# Torque range — 0.5 Nm allows peak headroom beyond steady-state limit (IQMAX_A × KT = 0.19 Nm)
MOTOR_KT       =  0.0095   # Nm/A
IQMAX_A        = 20.0      # must match IQMAX_A in drive_parameters.h
TORQUE_MAX_NM  = 0.5       # Nm
T_MIN,  T_MAX  = -TORQUE_MAX_NM, TORQUE_MAX_NM

# ── Motor node table  (name → command CAN ID) ─────────────────────────────────
NODES = {
    "flywheel_x (0x01)": {"mode": "motor", "id": 0x01},
    "flywheel_y (0x02)": {"mode": "motor", "id": 0x02},
    "spring_act (0x03)": {"mode": "motor", "id": 0x03},
    "jumping_sensor (0x04)": {"mode": "sensor", "id": 0x04},
}

MOTOR_ID    = 0x02            # default; overridden at runtime by dropdown
FEEDBACK_ID = MOTOR_ID + 0x100
CTRL_ID     = MOTOR_ID + 0x200
BAUD        = 2_000_000
SEND_HZ     = 50        # command rate
IO_HZ       = 500       # serial service rate
UI_HZ       = 20        # display refresh rate
SENSOR_LED_CMD_MAGIC = 0xA5
SENSOR_FB_X_ID = 0x104
SENSOR_FB_Y_ID = 0x105
SENSOR_ACCEL_FB_ID = 0x106
SENSOR_GAIN_X_FB_ID = 0x108
SENSOR_GAIN_Y_FB_ID = 0x109
SENSOR_CTRL_ID = 0x204
SENSOR_GAIN_CMD_MAGIC = 0x4B
SENSOR_GAIN_VERSION = 0x01
SENSOR_GAIN_SCALES = (100.0, 1000.0, 10000.0)
FLYWHEEL_X_FB_ID = 0x101
FLYWHEEL_Y_FB_ID = 0x102
FLYWHEEL_X_CMD_ID = 0x001
FLYWHEEL_Y_CMD_ID = 0x002
LQR_T_MIN, LQR_T_MAX = -1.0, 1.0
PLOT_HISTORY = 400

# ── MCSDK fault bit decode ───────────────────────────────────────────────────
FAULT_NAMES = {
    0x01: "FOC RATE",
    0x02: "OVER VOLT",
    0x04: "UNDER VOLT",
    0x08: "OVER TEMP",
    0x10: "STARTUP",
    0x20: "SPEED FB",
    0x40: "OVER CURR",
    0x80: "SW ERROR",
}

def decode_faults(code: int) -> str:
    """Return a short human-readable description of fault byte."""
    if code == 0:
        return "OK"
    names = [name for bit, name in FAULT_NAMES.items() if code & bit]
    return "  ".join(names) if names else f"0x{code:02X}"

def decode_sensor_faults(code: int) -> str:
    """Return sensor-specific CAN/IMU fault decode."""
    if code == 0:
        return "OK"
    names = []
    if code & 0x08:
        names.append("IMU DATA")
    if code & 0x10:
        names.append("TOF DATA")
    if code & 0x20:
        names.append("TOF INIT")
    if code & 0x40:
        names.append("CAN TX")
    if code & 0x80:
        names.append("IMU ID")
    return "  ".join(names) if names else f"0x{code:02X}"

# ── MCSDK state machine decode ─────────────────────────────────────────────────────
STATE_NAMES  = {0: "IDLE", 1: "RUN", 2: "STOP", 3: "FAULT", 4: "FAULT OVER", 5: "?"}
STATE_COLORS = {0: "#f9e2af", 1: "#a6e3a1", 2: "#6c7086", 3: "#f38ba8", 4: "#f38ba8", 5: "#6c7086"}

# ── Low-level protocol ────────────────────────────────────────────────────────

def _checksum(data):
    return sum(data) & 0xFF

def float_to_uint(x, x_min, x_max, bits):
    x = max(min(x, x_max), x_min)
    return int((x - x_min) * ((1 << bits) - 1) / (x_max - x_min))

def uint_to_float(x_int, x_min, x_max, bits):
    return x_min + float(x_int) * (x_max - x_min) / ((1 << bits) - 1)

def pack_command(p, v, kp, kd, t, motor_id=MOTOR_ID):
    p_int  = float_to_uint(p,  P_MIN,  P_MAX,  16)
    v_int  = float_to_uint(v,  V_MIN,  V_MAX,  12)
    kp_int = float_to_uint(kp, KP_MIN, KP_MAX, 12)
    kd_int = float_to_uint(kd, KD_MIN, KD_MAX, 12)
    t_int  = float_to_uint(t,  T_MIN,  T_MAX,  12)

    can_data = bytearray(8)
    can_data[0] = (p_int >> 8) & 0xFF
    can_data[1] =  p_int & 0xFF
    can_data[2] = (v_int >> 4) & 0xFF
    can_data[3] = ((v_int  & 0x0F) << 4) | ((kp_int >> 8) & 0x0F)
    can_data[4] =  kp_int & 0xFF
    can_data[5] = (kd_int >> 4) & 0xFF
    can_data[6] = ((kd_int & 0x0F) << 4) | ((t_int >> 8) & 0x0F)
    can_data[7] =  t_int & 0xFF

    core = [0x01, 0x01, 0x00, motor_id, 0x00, 0x00, 0x00, 0x08] + list(can_data) + [0x00]
    return bytearray([0xAA, 0x55] + core + [_checksum(core)])

def pack_ctrl_frame(cmd_byte: int, ctrl_id=CTRL_ID):
    """Pack a 1-byte state machine command (START=0x01, STOP=0x02) on ctrl_id."""
    id_bytes = [ctrl_id & 0xFF, (ctrl_id >> 8) & 0xFF, 0x00, 0x00]
    core = [0x01, 0x01, 0x00] + id_bytes + [0x01] + [cmd_byte] + [0x00]*7 + [0x00]
    return bytearray([0xAA, 0x55] + core + [_checksum(core)])

def pack_sensor_led_frame(r: int, g: int, b: int, sensor_ctrl_id: int):
    id_bytes = [sensor_ctrl_id & 0xFF, (sensor_ctrl_id >> 8) & 0xFF, 0x00, 0x00]
    can_data = [
        SENSOR_LED_CMD_MAGIC,
        max(0, min(255, int(r))),
        max(0, min(255, int(g))),
        max(0, min(255, int(b))),
        0x00, 0x00, 0x00, 0x00,
    ]
    core = [0x01, 0x01, 0x00] + id_bytes + [0x08] + can_data + [0x00]
    return bytearray([0xAA, 0x55] + core + [_checksum(core)])

def _gain_to_i16(value: float, scale: float) -> bytes:
    raw = round(float(value) * scale)
    if not -32768 <= raw <= 32767:
        raise ValueError(f"Gain {value} is outside the signed CAN range")
    return int(raw).to_bytes(2, byteorder="big", signed=True)

def pack_lqr_gain_frame(axis: str, k_theta: float, k_rate: float, k_wheel: float):
    if axis not in ("x", "y"):
        raise ValueError(f"Unknown LQR axis: {axis}")
    can_data = bytearray([SENSOR_GAIN_CMD_MAGIC, 0 if axis == "x" else 1])
    for value, scale in zip((k_theta, k_rate, k_wheel), SENSOR_GAIN_SCALES):
        can_data.extend(_gain_to_i16(value, scale))
    id_bytes = [
        SENSOR_CTRL_ID & 0xFF,
        (SENSOR_CTRL_ID >> 8) & 0xFF,
        0x00,
        0x00,
    ]
    core = [0x01, 0x01, 0x00] + id_bytes + [0x08] + list(can_data) + [0x00]
    return bytearray([0xAA, 0x55] + core + [_checksum(core)])

def make_init_frame():
    core = [0x12, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00]
    return bytearray([0xAA, 0x55] + core + [_checksum(core)])


# ── Colour palette (Catppuccin Mocha) ─────────────────────────────────────────
BG      = "#1e1e2e"
SURFACE = "#313244"
OVERLAY = "#45475a"
MUTED   = "#6c7086"
TEXT    = "#cdd6f4"
BLUE    = "#89b4fa"
GREEN   = "#a6e3a1"
RED     = "#f38ba8"
YELLOW  = "#f9e2af"
MAUVE   = "#cba6f7"


# ── Main application ──────────────────────────────────────────────────────────

class MotorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Motor Control Panel")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self.ser: serial.Serial | None = None
        self.running = False
        self._io_thread = None
        self._rx_buf = bytearray()

        # Active node selection — updated by the dropdown
        self._node_mode  = "motor"
        self._motor_id   = 0x02
        self._feedback_id = self._motor_id + 0x100
        self._ctrl_id    = self._motor_id + 0x200

        # Feedback state (written by IO thread, read by UI thread — floats are GIL-safe)
        self.fb_pos   = 0.0
        self.fb_vel   = 0.0
        self.fb_torq  = 0.0
        self.fb_id    = 0
        self.fb_err   = 0
        self.fb_state = 0  # compressed state nibble from firmware
        self.fb_count = 0
        self.tx_count = 0

        # Plain Python attrs mirroring slider/enable state — safe to read from IO thread
        self._enabled = False
        self._ctrl_queue = []
        self._ctrl_lock = threading.Lock()
        self._cmd = {"p": 0.0, "v": 0.0, "kp": 0.0, "kd": 0.0, "t": 0.0}
        self._sensor_led = {"r": 0, "g": 0, "b": 0}
        self._entry_updating = False  # re-entrancy guard for _entry_commit
        self._io_error = ""

        self._telemetry_lock = threading.Lock()
        self._telemetry = {
            "x": {"angle": 0.0, "wrapped_angle": 0.0, "rate": 0.0, "speed": 0.0, "command": 0.0,
                  "gains": (48.3391, 0.0, -0.2236), "gains_seen": False,
                  "accel_angle": 0.0, "accel_wrapped_angle": 0.0, "accel_seen": False,
                  "accel_valid": False, "gyro_rejected": False, "resync_count": None,
                  "state": 0, "error": 0, "sensor_seen": False, "motor_seen": False},
            "y": {"angle": 0.0, "wrapped_angle": 0.0, "rate": 0.0, "speed": 0.0, "command": 0.0,
                  "gains": (48.3391, 0.0, -0.2236), "gains_seen": False,
                  "accel_angle": 0.0, "accel_wrapped_angle": 0.0, "accel_seen": False,
                  "accel_valid": False, "gyro_rejected": False, "resync_count": None,
                  "state": 0, "error": 0, "sensor_seen": False, "motor_seen": False},
        }
        self._plot_history = {
            "x": {"angle": collections.deque(maxlen=PLOT_HISTORY),
                  "accel_angle": collections.deque(maxlen=PLOT_HISTORY),
                  "command": collections.deque(maxlen=PLOT_HISTORY)},
            "y": {"angle": collections.deque(maxlen=PLOT_HISTORY),
                  "accel_angle": collections.deque(maxlen=PLOT_HISTORY),
                  "command": collections.deque(maxlen=PLOT_HISTORY)},
        }
        self._lqr_window = None
        self._lqr_axis_var = None
        self._lqr_angle_canvas = None
        self._lqr_command_canvas = None
        self._lqr_state_vars = {}
        self._lqr_gain_input_vars = {}
        self._lqr_gain_readback_vars = {}

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self._style_ttk()

        # Connection bar
        self._build_conn_bar()

        # Separator
        tk.Frame(self.root, bg=OVERLAY, height=1).pack(fill="x", padx=10)

        # Command sliders
        self._build_sliders()

        # Sensor LED controls
        self._build_sensor_led_controls()

        tk.Frame(self.root, bg=OVERLAY, height=1).pack(fill="x", padx=10)

        # Feedback panel
        self._build_feedback()

        tk.Frame(self.root, bg=OVERLAY, height=1).pack(fill="x", padx=10)

        # Log line
        self._log_var = tk.StringVar(value="Not connected.")
        tk.Label(self.root, textvariable=self._log_var, bg=BG, fg=MUTED,
                 font=("Consolas", 9), anchor="w").pack(fill="x", padx=14, pady=(4, 8))

    def _style_ttk(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TFrame",       background=BG)
        s.configure("TLabel",       background=BG,      foreground=TEXT, font=("Consolas", 10))
        s.configure("TButton",      background=SURFACE, foreground=TEXT, font=("Consolas", 10),
                    relief="flat", borderwidth=0)
        s.map("TButton",            background=[("active", OVERLAY)])
        s.configure("TCheckbutton", background=BG, foreground=TEXT, font=("Consolas", 10))
        s.configure("Horizontal.TScale", background=BG, troughcolor=SURFACE,
                    sliderlength=16, sliderrelief="flat")
        s.configure("TCombobox",    fieldbackground=SURFACE, background=SURFACE,
                    foreground=TEXT, selectbackground=OVERLAY)

    def _build_conn_bar(self):
        f = tk.Frame(self.root, bg=BG)
        f.pack(fill="x", padx=10, pady=10)

        tk.Label(f, text="Port:", bg=BG, fg=MUTED, font=("Consolas", 10)).pack(side="left")

        self._port_var = tk.StringVar()
        self._port_cb  = ttk.Combobox(f, textvariable=self._port_var, width=10,
                                      state="readonly")
        self._port_cb.pack(side="left", padx=(4, 8))
        self._refresh_ports()

        ttk.Button(f, text="⟳ Refresh", command=self._refresh_ports,
                   width=10).pack(side="left", padx=2)

        self._conn_btn = ttk.Button(f, text="Connect", command=self._toggle_connect,
                                    width=12)
        self._conn_btn.pack(side="left", padx=6)

        # Node selector dropdown
        tk.Label(f, text="Node:", bg=BG, fg=MUTED,
                 font=("Consolas", 10)).pack(side="left", padx=(10, 2))
        self._motor_var = tk.StringVar()
        motor_names = list(NODES.keys())
        default_name = next(k for k, v in NODES.items() if v["id"] == 0x02)
        self._motor_var.set(default_name)
        motor_cb = ttk.Combobox(f, textvariable=self._motor_var,
                                values=motor_names, width=18, state="readonly")
        motor_cb.pack(side="left", padx=(0, 8))
        self._motor_var.trace_add("write", self._on_motor_change)

        self._status_lbl = tk.Label(f, text="● Disconnected", bg=BG, fg=RED,
                                    font=("Consolas", 10, "bold"))
        self._status_lbl.pack(side="left", padx=6)

        # Motor ON / OFF buttons
        self._motor_on_btn = ttk.Button(f, text="Motor ON",  command=self._motor_on,  width=10)
        self._motor_on_btn.pack(side="left", padx=(16, 2))
        self._motor_off_btn = ttk.Button(f, text="Motor OFF", command=self._motor_off, width=10)
        self._motor_off_btn.pack(side="left", padx=2)
        self._realign_btn = ttk.Button(f, text="Re-Align",  command=self._force_align, width=10)
        self._realign_btn.pack(side="left", padx=2)
        ttk.Button(f, text="LQR Monitor", command=self._open_lqr_monitor,
                   width=12).pack(side="left", padx=(8, 2))

        # State badge — updates from feedback
        self._state_var = tk.StringVar(value="STATE: --")
        self._state_lbl = tk.Label(f, textvariable=self._state_var, bg=BG,
                                   fg=MUTED, font=("Consolas", 10, "bold"))
        self._state_lbl.pack(side="left", padx=12)

        self._enable_var = tk.BooleanVar(value=False)
        self._enable_check = ttk.Checkbutton(f, text="Enable Output", variable=self._enable_var,
                                             command=self._on_enable_toggle)
        self._enable_check.pack(side="right", padx=4)

    def _build_sliders(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="x", padx=10, pady=8)

        tk.Label(outer, text="COMMAND", bg=BG, fg=BLUE,
                 font=("Consolas", 11, "bold")).grid(row=0, column=0, columnspan=5,
                                                      sticky="w", padx=4, pady=(0, 6))

        # (display name, key, min, max, default, unit, entry width)
        params = [
            ("Position",  "p",  P_MIN,  P_MAX,  0.0,  "rad",   7),
            ("Velocity",  "v",  V_MIN,  V_MAX,  0.0,  "rad/s", 7),
            ("Kp",        "kp", KP_MIN, KP_MAX, 0.0,  "Nm/rad",7),
            ("Kd",        "kd", KD_MIN, KD_MAX, 0.0,  "Nm·s/r",7),
            ("Torque FF", "t",  T_MIN,  T_MAX,  0.0,  "Nm",    7),
        ]

        self._slider_vars = {}

        for row, (name, key, mn, mx, default, unit, ew) in enumerate(params, start=1):
            var = tk.DoubleVar(value=default)
            self._slider_vars[key] = var

            tk.Label(outer, text=name, bg=BG, fg=TEXT, font=("Consolas", 10),
                     width=10, anchor="e").grid(row=row, column=0, padx=(4, 6),
                                                pady=3, sticky="e")

            slider = ttk.Scale(outer, from_=mn, to=mx, variable=var,
                               orient="horizontal", length=300)
            slider.grid(row=row, column=1, padx=4, pady=3, sticky="ew")

            # Numeric entry — bidirectionally linked to the slider
            entry_var = tk.StringVar(value=f"{default:.3f}")
            entry = tk.Entry(outer, textvariable=entry_var, width=ew,
                             bg=SURFACE, fg=BLUE, insertbackground=BLUE,
                             font=("Consolas", 10), relief="flat",
                             highlightthickness=1, highlightbackground=OVERLAY,
                             highlightcolor=BLUE, justify="right")
            entry.grid(row=row, column=2, padx=(6, 2), pady=3)

            tk.Label(outer, text=unit, bg=BG, fg=MUTED,
                     font=("Consolas", 9), width=7, anchor="w").grid(
                         row=row, column=3, padx=(2, 8), pady=3, sticky="w")

            # Slider → entry + mirror to thread-safe plain attr
            def _slider_moved(*args, v=var, ev=entry_var, k=key):
                val = v.get()
                ev.set(f"{val:.3f}")
                self._cmd[k] = val
            var.trace_add("write", _slider_moved)

            # Entry → slider + mirror to thread-safe plain attr
            def _entry_commit(event, v=var, ev=entry_var, lo=mn, hi=mx, k=key):
                if self._entry_updating:
                    return
                try:
                    val = float(ev.get())
                    val = max(lo, min(hi, val))
                    self._entry_updating = True
                    self._cmd[k] = val        # update motor command FIRST
                    ev.set(f"{val:.3f}")      # update display
                    v.set(val)                # move slider (fires _slider_moved but _cmd already correct)
                except ValueError:
                    ev.set(f"{v.get():.3f}")
                finally:
                    self._entry_updating = False
            entry.bind("<Return>",    _entry_commit)
            entry.bind("<FocusOut>",  _entry_commit)

        outer.columnconfigure(1, weight=1)

        # Zero / Stop buttons
        btn_row = len(params) + 1
        btn_frame = tk.Frame(outer, bg=BG)
        btn_frame.grid(row=btn_row, column=0, columnspan=5, pady=(8, 2))

        ttk.Button(btn_frame, text="Zero All",
                   command=self._zero_all, width=12).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="Zero Torque",
                   command=self._zero_torque, width=14).pack(side="left", padx=6)

    def _build_feedback(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="x", padx=10, pady=8)

        tk.Label(outer, text="FEEDBACK", bg=BG, fg=GREEN,
                 font=("Consolas", 11, "bold")).grid(row=0, column=0, columnspan=6,
                                                      sticky="w", padx=4, pady=(0, 6))

        fields = [
            ("Position",  "pos",  "rad",   P_MIN, P_MAX,  BLUE),
            ("Velocity",  "vel",  "rad/s", V_MIN, V_MAX,  MAUVE),
            ("Torque",    "torq", "Nm",    T_MIN, T_MAX,  YELLOW),
        ]

        self._fb_val_vars   = {}
        self._fb_name_vars  = {}
        self._fb_unit_vars  = {}
        self._fb_val_labels = {}   # kept for dynamic fg colour changes
        self._bar_canvases  = {}
        self._fb_ranges     = {}

        for col, (name, key, unit, lo, hi, color) in enumerate(fields):
            c = tk.Frame(outer, bg=SURFACE, bd=0)
            c.grid(row=1, column=col, padx=6, pady=4, sticky="nsew")

            name_var = tk.StringVar(value=name)
            self._fb_name_vars[key] = name_var
            tk.Label(c, textvariable=name_var, bg=SURFACE, fg=MUTED,
                     font=("Consolas", 9)).pack(anchor="w", padx=8, pady=(6, 0))

            val_var = tk.StringVar(value="---")
            self._fb_val_vars[key] = val_var
            lbl = tk.Label(c, textvariable=val_var, bg=SURFACE, fg=color,
                           font=("Consolas", 22, "bold"), width=8, anchor="e")
            lbl.pack(padx=8, pady=(0, 2))
            self._fb_val_labels[key] = lbl

            unit_var = tk.StringVar(value=unit)
            self._fb_unit_vars[key] = unit_var
            tk.Label(c, textvariable=unit_var, bg=SURFACE, fg=MUTED,
                     font=("Consolas", 9)).pack(anchor="e", padx=8, pady=(0, 4))

            bar = tk.Canvas(c, bg=OVERLAY, height=8, highlightthickness=0, width=170)
            bar.pack(fill="x", padx=6, pady=(0, 8))
            self._bar_canvases[key] = (bar, color)
            self._fb_ranges[key] = (lo, hi)

        # ── Error code panel (4th column, no bar, colour-coded) ──────────────
        err_frame = tk.Frame(outer, bg=SURFACE, bd=0)
        err_frame.grid(row=1, column=3, padx=6, pady=4, sticky="nsew")

        tk.Label(err_frame, text="Error", bg=SURFACE, fg=MUTED,
                 font=("Consolas", 9)).pack(anchor="w", padx=8, pady=(6, 0))

        self._fb_err_code_var = tk.StringVar(value="---")
        self._fb_err_lbl = tk.Label(err_frame, textvariable=self._fb_err_code_var,
                                    bg=SURFACE, fg=GREEN,
                                    font=("Consolas", 14, "bold"), width=10, anchor="center",
                                    wraplength=130, justify="center")
        self._fb_err_lbl.pack(padx=8, pady=(4, 2))

        self._fb_err_hex_var = tk.StringVar(value="0x00")
        tk.Label(err_frame, textvariable=self._fb_err_hex_var,
                 bg=SURFACE, fg=MUTED,
                 font=("Consolas", 9)).pack(anchor="e", padx=8, pady=(0, 4))

        # Spacer where the bar would be, so heights match
        tk.Frame(err_frame, bg=SURFACE, height=16).pack(pady=(0, 8))

        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
        outer.columnconfigure(2, weight=1)
        outer.columnconfigure(3, weight=1)

        # Status strip
        stat = tk.Frame(outer, bg=BG)
        stat.grid(row=2, column=0, columnspan=6, sticky="w", padx=4, pady=(2, 0))

        self._fb_id_var    = tk.StringVar(value="Node ID: —")
        self._fb_rx_var    = tk.StringVar(value="RX: 0  frames")
        self._fb_tx_var    = tk.StringVar(value="TX: 0  cmds")

        for v in (self._fb_id_var, self._fb_rx_var, self._fb_tx_var):
            tk.Label(stat, textvariable=v, bg=BG, fg=MUTED,
                     font=("Consolas", 9)).pack(side="left", padx=12)

    def _build_sensor_led_controls(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="x", padx=10, pady=(2, 8))

        tk.Label(outer, text="SENSOR LED (RGB)", bg=BG, fg=MAUVE,
                 font=("Consolas", 11, "bold")).grid(row=0, column=0, columnspan=5,
                                                      sticky="w", padx=4, pady=(0, 6))

        self._led_slider_vars = {}
        led_fields = [("Red", "r", RED), ("Green", "g", GREEN), ("Blue", "b", BLUE)]
        for row, (name, key, color) in enumerate(led_fields, start=1):
            var = tk.IntVar(value=0)
            self._led_slider_vars[key] = var
            tk.Label(outer, text=name, bg=BG, fg=TEXT, font=("Consolas", 10),
                     width=10, anchor="e").grid(row=row, column=0, padx=(4, 6),
                                                pady=3, sticky="e")
            slider = ttk.Scale(outer, from_=0, to=255, variable=var,
                               orient="horizontal", length=300)
            slider.grid(row=row, column=1, padx=4, pady=3, sticky="ew")
            val_lbl = tk.Label(outer, text="0", bg=BG, fg=color, font=("Consolas", 10, "bold"),
                               width=4, anchor="e")
            val_lbl.grid(row=row, column=2, padx=(6, 2), pady=3, sticky="e")

            def _led_changed(*args, k=key, v=var, lbl=val_lbl):
                val = max(0, min(255, int(v.get())))
                self._sensor_led[k] = val
                lbl.config(text=str(val))
            var.trace_add("write", _led_changed)

        ttk.Button(outer, text="LED Off",
                   command=lambda: [self._led_slider_vars["r"].set(0),
                                    self._led_slider_vars["g"].set(0),
                                    self._led_slider_vars["b"].set(0)],
                   width=10).grid(row=1, column=3, rowspan=3, padx=(10, 2), pady=2)
        tk.Label(outer, text="(Enable Output + select jumping_sensor to send)", bg=BG, fg=MUTED,
                 font=("Consolas", 9)).grid(row=4, column=0, columnspan=5, sticky="w", padx=4, pady=(2, 0))
        outer.columnconfigure(1, weight=1)

    def _open_lqr_monitor(self):
        if self._lqr_window is not None and self._lqr_window.winfo_exists():
            self._lqr_window.lift()
            return

        win = tk.Toplevel(self.root)
        win.title("LQR Telemetry")
        win.configure(bg=BG)
        win.geometry("1040x760")
        self._lqr_window = win
        self._lqr_axis_var = tk.StringVar(value="x")

        header = tk.Frame(win, bg=BG)
        header.pack(fill="x", padx=12, pady=10)
        tk.Label(header, text="Axis:", bg=BG, fg=MUTED,
                 font=("Consolas", 10)).pack(side="left")
        for axis in ("x", "y"):
            tk.Radiobutton(header, text=axis.upper(), value=axis,
                           variable=self._lqr_axis_var, bg=BG, fg=TEXT,
                           selectcolor=SURFACE, activebackground=BG,
                           activeforeground=TEXT, font=("Consolas", 10, "bold")).pack(
                               side="left", padx=4)

        for axis in ("x", "y"):
            group = tk.Frame(header, bg=BG)
            group.pack(side="left", padx=(18, 0))
            tk.Label(group, text=f"{axis.upper()}:", bg=BG, fg=BLUE,
                     font=("Consolas", 10, "bold")).pack(side="left")
            ttk.Button(group, text="ON", width=5,
                       command=lambda a=axis: self._queue_axis_control(a, 0x01, "ON")).pack(
                           side="left", padx=2)
            ttk.Button(group, text="OFF", width=5,
                       command=lambda a=axis: self._queue_axis_control(a, 0x02, "OFF")).pack(
                           side="left", padx=2)
            ttk.Button(group, text="ALIGN", width=7,
                       command=lambda a=axis: self._queue_axis_control(a, 0x03, "ALIGN")).pack(
                           side="left", padx=2)
            state_var = tk.StringVar(value="--")
            self._lqr_state_vars[axis] = state_var
            tk.Label(group, textvariable=state_var, bg=BG, fg=GREEN,
                     font=("Consolas", 9, "bold"), width=16, anchor="w").pack(
                         side="left", padx=6)

        gain_panel = tk.Frame(win, bg=BG)
        gain_panel.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(
            gain_panel,
            text="SIGNED LQR GAINS K  (firmware applies u = -Kx)",
            bg=BG,
            fg=MAUVE,
            font=("Consolas", 10, "bold"),
        ).grid(row=0, column=0, columnspan=7, sticky="w", pady=(0, 4))
        for column, label in enumerate(("Axis", "K theta", "K rate", "K wheel", "", "Readback K / applied -K")):
            tk.Label(
                gain_panel, text=label, bg=BG, fg=MUTED,
                font=("Consolas", 9, "bold")
            ).grid(row=1, column=column, padx=4, sticky="w")

        defaults = ("48.3391", "0.0", "-0.2236")
        for row, axis in enumerate(("x", "y"), start=2):
            tk.Label(
                gain_panel, text=axis.upper(), bg=BG, fg=BLUE,
                font=("Consolas", 10, "bold")
            ).grid(row=row, column=0, padx=4, pady=2)
            variables = [tk.StringVar(value=value) for value in defaults]
            self._lqr_gain_input_vars[axis] = variables
            for column, variable in enumerate(variables, start=1):
                ttk.Entry(gain_panel, textvariable=variable, width=11).grid(
                    row=row, column=column, padx=4, pady=2
                )
            ttk.Button(
                gain_panel,
                text="Set",
                width=7,
                command=lambda a=axis: self._queue_lqr_gains(a),
            ).grid(row=row, column=4, padx=6)
            readback_var = tk.StringVar(value="waiting for CAN")
            self._lqr_gain_readback_vars[axis] = readback_var
            tk.Label(
                gain_panel, textvariable=readback_var, bg=BG, fg=GREEN,
                font=("Consolas", 9), anchor="w", width=48
            ).grid(row=row, column=5, padx=4, sticky="w")
        ttk.Button(
            gain_panel,
            text="Set Both",
            width=10,
            command=self._queue_lqr_gains_both,
        ).grid(row=2, column=6, rowspan=2, padx=(8, 0))

        self._lqr_angle_canvas = tk.Canvas(win, bg=SURFACE, highlightthickness=0)
        self._lqr_angle_canvas.pack(fill="both", expand=True, padx=12, pady=(2, 6))
        self._lqr_command_canvas = tk.Canvas(win, bg=SURFACE, highlightthickness=0)
        self._lqr_command_canvas.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        def _closed():
            self._lqr_window = None
            self._lqr_angle_canvas = None
            self._lqr_command_canvas = None
            self._lqr_state_vars = {}
            self._lqr_gain_input_vars = {}
            self._lqr_gain_readback_vars = {}
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _closed)

    def _queue_axis_control(self, axis: str, command: int, label: str):
        ctrl_id = 0x201 if axis == "x" else 0x202
        self._queue_control(ctrl_id, command, f"{axis.upper()} {label}")

    def _queue_control(self, ctrl_id: int, command: int, label: str):
        if not self.running:
            self._log("Connect CAN before sending axis controls.")
            return

        frame = pack_ctrl_frame(command, ctrl_id)
        now = time.monotonic()
        repeats = 3 if command == 0x03 else 2
        with self._ctrl_lock:
            for index in range(repeats):
                self._ctrl_queue.append((now + index * 0.02, frame))
            self._ctrl_queue.sort(key=lambda item: item[0])
        self._log(f"{label} queued.")

    def _gain_values_from_ui(self, axis: str):
        variables = self._lqr_gain_input_vars.get(axis)
        if variables is None:
            raise ValueError(f"{axis.upper()} gain controls are not open")
        try:
            return tuple(float(variable.get().strip()) for variable in variables)
        except ValueError as exc:
            raise ValueError("LQR gains must be signed numbers") from exc

    def _queue_lqr_gains(self, axis: str):
        if not self.running:
            self._log("Connect CAN before sending LQR gains.")
            return
        try:
            frame = pack_lqr_gain_frame(axis, *self._gain_values_from_ui(axis))
        except ValueError as exc:
            self._log(str(exc))
            return
        now = time.monotonic()
        with self._ctrl_lock:
            for index in range(2):
                self._ctrl_queue.append((now + index * 0.02, frame))
            self._ctrl_queue.sort(key=lambda item: item[0])
        self._log(f"{axis.upper()} signed LQR gains queued.")

    def _queue_lqr_gains_both(self):
        self._queue_lqr_gains("x")
        self._queue_lqr_gains("y")

    def _refresh_lqr_monitor(self):
        if self._lqr_window is None or not self._lqr_window.winfo_exists():
            return

        with self._telemetry_lock:
            snapshot = {axis: values.copy() for axis, values in self._telemetry.items()}

        for axis, values in snapshot.items():
            if values["sensor_seen"]:
                self._plot_history[axis]["angle"].append(values["angle"])
                self._plot_history[axis]["command"].append(values["command"])
            if values["accel_seen"]:
                self._plot_history[axis]["accel_angle"].append(values["accel_angle"])
            state_name = STATE_NAMES.get(values["state"], "?") if values["motor_seen"] else "--"
            accel_state = "ACC OK" if values["accel_valid"] else "ACC REJECT"
            gyro_state = "GYRO REJECT" if values["gyro_rejected"] else "GYRO OK"
            self._lqr_state_vars[axis].set(
                f"{state_name} {accel_state} {gyro_state} err=0x{values['error']:02X}")
            gain_var = self._lqr_gain_readback_vars.get(axis)
            if gain_var is not None:
                if values["gains_seen"]:
                    gains = values["gains"]
                    gain_var.set(
                        f"K [{gains[0]:+.2f}, {gains[1]:+.3f}, {gains[2]:+.4f}]  "
                        f"-K [{-gains[0]:+.2f}, {-gains[1]:+.3f}, {-gains[2]:+.4f}]"
                    )
                else:
                    gain_var.set("waiting for CAN")

        axis = self._lqr_axis_var.get()
        self._draw_trace(self._lqr_angle_canvas,
                         self._plot_history[axis]["angle"],
                         f"{axis.upper()} angle: Madgwick estimate vs filtered accel",
                         "rad",
                         BLUE,
                         self._plot_history[axis]["accel_angle"],
                         GREEN)
        self._draw_trace(self._lqr_command_canvas,
                         self._plot_history[axis]["command"],
                         f"{axis.upper()} torque command", "Nm", YELLOW)

    @staticmethod
    def _draw_trace(canvas: tk.Canvas, values, title: str, unit: str, color: str,
                    secondary_values=None, secondary_color=GREEN):
        canvas.update_idletasks()
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width < 20 or height < 20:
            return

        canvas.delete("all")
        margin_x = 48
        margin_y = 24
        plot_w = max(1, width - margin_x - 10)
        plot_h = max(1, height - 2 * margin_y)
        canvas.create_text(8, 8, text=title, fill=TEXT, anchor="nw",
                           font=("Consolas", 10, "bold"))

        samples = list(values)
        secondary_samples = list(secondary_values) if secondary_values is not None else []
        all_samples = samples + secondary_samples
        if not all_samples:
            canvas.create_text(width / 2, height / 2, text="Waiting for CAN telemetry",
                               fill=MUTED, font=("Consolas", 11))
            return

        lo = min(min(all_samples), 0.0)
        hi = max(max(all_samples), 0.0)
        span = hi - lo
        if span < 1e-6:
            span = 1.0
            lo -= 0.5
            hi += 0.5
        else:
            pad = span * 0.1
            lo -= pad
            hi += pad
            span = hi - lo

        for fraction in (0.0, 0.5, 1.0):
            y = margin_y + plot_h * fraction
            canvas.create_line(margin_x, y, width - 10, y, fill=OVERLAY)
            value = hi - span * fraction
            canvas.create_text(margin_x - 5, y, text=f"{value:+.2f}",
                               fill=MUTED, anchor="e", font=("Consolas", 8))

        if lo <= 0.0 <= hi:
            zero_y = margin_y + (hi / span) * plot_h
            canvas.create_line(margin_x, zero_y, width - 10, zero_y,
                               fill=MUTED, dash=(3, 3))

        def draw_samples(series, line_color):
            count = len(series)
            coords = []
            for index, value in enumerate(series):
                x = margin_x + (index / max(1, count - 1)) * plot_w
                y = margin_y + ((hi - value) / span) * plot_h
                coords.extend((x, y))
            if len(coords) >= 4:
                canvas.create_line(*coords, fill=line_color, width=2, smooth=False)

        draw_samples(samples, color)
        draw_samples(secondary_samples, secondary_color)
        if samples:
            canvas.create_text(width - 12, 8, text=f"Madgwick {samples[-1]:+.4f} {unit}",
                               fill=color, anchor="ne", font=("Consolas", 10, "bold"))
        if secondary_samples:
            canvas.create_text(width - 12, 24,
                               text=f"accel {secondary_samples[-1]:+.4f} {unit}",
                               fill=secondary_color, anchor="ne",
                               font=("Consolas", 10, "bold"))

    # ── Connection helpers ────────────────────────────────────────────────────

    def _refresh_ports(self):
        ports = serial.tools.list_ports.comports()
        names = [p.device for p in ports]
        self._port_cb["values"] = names
        ch340 = [p.device for p in ports if "CH340" in (p.description or "")]
        if ch340:
            self._port_var.set(ch340[0])
        elif names:
            self._port_var.set(names[0])

    def _toggle_connect(self):
        if self.running:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        port = self._port_var.get()
        if not port:
            self._log("No port selected.")
            return
        try:
            self.ser = serial.Serial(port, BAUD, timeout=0.01)
            self.ser.write(make_init_frame())
            time.sleep(0.2)
            self._rx_buf.clear()
            self._io_error = ""
            self.running = True
            self._conn_btn.config(text="Disconnect")
            self._status_lbl.config(text=f"● {port}", fg=GREEN)
            self._log(f"Connected to {port} at {BAUD:,} baud.")
            self._io_thread = threading.Thread(target=self._io_loop, daemon=True)
            self._io_thread.start()
            self._schedule_ui()
        except Exception as exc:
            self._log(f"Connection failed: {exc}")
            self._status_lbl.config(text="● Error", fg=RED)

    def _disconnect(self):
        self.running = False
        if self._io_thread is not None and self._io_thread is not threading.current_thread():
            self._io_thread.join(timeout=0.5)
        self._io_thread = None
        if self.ser and self.ser.is_open:
            try:
                if self._node_mode == "motor":
                    self.ser.write(pack_command(0, 0, 0, 0, 0, self._motor_id))
                    time.sleep(0.05)
                self.ser.close()
            except Exception:
                pass
        with self._ctrl_lock:
            self._ctrl_queue.clear()
        self._enable_var.set(False)
        self._enabled = False
        self._conn_btn.config(text="Connect")
        self._status_lbl.config(text="● Disconnected", fg=RED)
        self._log("Disconnected. Zero torque sent.")

    def _on_motor_change(self, *_):
        name = self._motor_var.get()
        node = NODES.get(name, {"mode": "motor", "id": 0x02})
        mid = node["id"]
        self._node_mode   = node["mode"]
        self._motor_id    = mid
        self._feedback_id = mid + 0x100
        self._ctrl_id     = mid + 0x200
        self._set_feedback_profile(self._node_mode)
        controls_state = "normal" if self._node_mode == "motor" else "disabled"
        self._motor_on_btn.config(state=controls_state)
        self._motor_off_btn.config(state=controls_state)
        self._realign_btn.config(state=controls_state)
        self._enable_check.config(state="normal")
        self._log(f"Node → {name}  (mode={self._node_mode} cmd 0x{mid:02X} fb 0x{mid+0x100:03X})")

    def _set_feedback_profile(self, mode: str):
        if mode == "sensor":
            self._fb_name_vars["pos"].set("Body Angle")
            self._fb_name_vars["vel"].set("Gyro Rate")
            self._fb_name_vars["torq"].set("TOF Dist")
            self._fb_unit_vars["pos"].set("rad")
            self._fb_unit_vars["vel"].set("rad/s")
            self._fb_unit_vars["torq"].set("mm")
            self._fb_ranges["pos"] = (P_MIN, P_MAX)
            self._fb_ranges["vel"] = (V_MIN, V_MAX)
            self._fb_ranges["torq"] = (0.0, 4000.0)
        else:
            self._fb_name_vars["pos"].set("Position")
            self._fb_name_vars["vel"].set("Velocity")
            self._fb_name_vars["torq"].set("Torque")
            self._fb_unit_vars["pos"].set("rad")
            self._fb_unit_vars["vel"].set("rad/s")
            self._fb_unit_vars["torq"].set("Nm")
            self._fb_ranges["pos"] = (P_MIN, P_MAX)
            self._fb_ranges["vel"] = (V_MIN, V_MAX)
            self._fb_ranges["torq"] = (T_MIN, T_MAX)

    def _on_close(self):
        self._disconnect()
        self.root.destroy()

    def _on_enable_toggle(self):
        if self._enable_var.get() and not self.running:
            self._enable_var.set(False)
            self._log("Connect first before enabling output.")
        self._enabled = bool(self._enable_var.get())

    def _motor_on(self):
        """Send a START command to the motor state machine."""
        if self._node_mode != "motor":
            self._log("Motor commands are disabled for jumping_sensor.")
            return
        if self.running:
            self._queue_control(self._ctrl_id, 0x01, "Motor ON")
        else:
            self._log("Connect first.")

    def _motor_off(self):
        """Send a STOP command to the motor state machine."""
        if self._node_mode != "motor":
            self._log("Motor commands are disabled for jumping_sensor.")
            return
        if self.running:
            self._queue_control(self._ctrl_id, 0x02, "Motor OFF")
        else:
            self._log("Connect first.")

    def _force_align(self):
        """Stop the motor, clear encoder alignment, and re-run alignment."""
        if self._node_mode != "motor":
            self._log("Alignment command is disabled for jumping_sensor.")
            return
        if self.running:
            self._queue_control(self._ctrl_id, 0x03, "Motor ALIGN")
        else:
            self._log("Connect first.")

    def _log(self, msg: str):
        self._log_var.set(msg)

    def _parse_lqr_telemetry(self, can_id: int, data: bytes):
        if can_id in (SENSOR_GAIN_X_FB_ID, SENSOR_GAIN_Y_FB_ID):
            axis = "x" if can_id == SENSOR_GAIN_X_FB_ID else "y"
            expected_axis = 0 if axis == "x" else 1
            if data[0] != SENSOR_GAIN_VERSION or data[1] != expected_axis:
                return
            gains = tuple(
                int.from_bytes(
                    data[offset:offset + 2], byteorder="big", signed=True
                ) / scale
                for offset, scale in zip((2, 4, 6), SENSOR_GAIN_SCALES)
            )
            with self._telemetry_lock:
                self._telemetry[axis]["gains"] = gains
                self._telemetry[axis]["gains_seen"] = True
            return

        if can_id == SENSOR_ACCEL_FB_ID:
            accel_values = {
                "x": (uint_to_float((data[2] << 8) | data[3], P_MIN, P_MAX, 16), data[6]),
                "y": (uint_to_float((data[4] << 8) | data[5], P_MIN, P_MAX, 16), data[7]),
            }
            accel_valid = (data[1] & 0x01) != 0
            gyro_rejected = (data[1] & 0x02) != 0
            with self._telemetry_lock:
                for axis, (wrapped_angle, resync_count) in accel_values.items():
                    telemetry = self._telemetry[axis]
                    if telemetry["accel_seen"]:
                        delta = wrapped_angle - telemetry["accel_wrapped_angle"]
                        if delta > math.pi:
                            delta -= 2.0 * math.pi
                        elif delta < -math.pi:
                            delta += 2.0 * math.pi
                        accel_angle = telemetry["accel_angle"] + delta
                    else:
                        accel_angle = wrapped_angle

                    previous_resync_count = telemetry["resync_count"]
                    telemetry["accel_angle"] = accel_angle
                    telemetry["accel_wrapped_angle"] = wrapped_angle
                    telemetry["accel_seen"] = True
                    telemetry["accel_valid"] = accel_valid
                    telemetry["gyro_rejected"] = gyro_rejected
                    telemetry["resync_count"] = resync_count
                    if (previous_resync_count is not None and
                            resync_count != previous_resync_count):
                        telemetry["angle"] = accel_angle
                        telemetry["wrapped_angle"] = wrapped_angle
            return

        axis = None
        updates = {}
        if can_id == SENSOR_FB_X_ID:
            axis = "x"
            updates = {
                "wrapped_angle": uint_to_float((data[2] << 8) | data[3], P_MIN, P_MAX, 16),
                "rate": uint_to_float((data[4] << 8) | data[5], V_MIN, V_MAX, 16),
                "sensor_seen": True,
            }
        elif can_id == SENSOR_FB_Y_ID:
            axis = "y"
            updates = {
                "wrapped_angle": uint_to_float((data[2] << 8) | data[3], P_MIN, P_MAX, 16),
                "rate": uint_to_float((data[4] << 8) | data[5], V_MIN, V_MAX, 16),
                "sensor_seen": True,
            }
        elif can_id in (FLYWHEEL_X_FB_ID, FLYWHEEL_Y_FB_ID):
            axis = "x" if can_id == FLYWHEEL_X_FB_ID else "y"
            updates = {
                "speed": uint_to_float((data[4] << 8) | data[5], V_MIN, V_MAX, 16),
                "state": (data[0] >> 4) & 0x0F,
                "error": data[1],
                "motor_seen": True,
            }
        elif can_id in (FLYWHEEL_X_CMD_ID, FLYWHEEL_Y_CMD_ID):
            axis = "x" if can_id == FLYWHEEL_X_CMD_ID else "y"
            torque_raw = ((data[6] & 0x0F) << 8) | data[7]
            updates = {
                "command": uint_to_float(torque_raw, LQR_T_MIN, LQR_T_MAX, 12),
            }

        if axis is not None:
            with self._telemetry_lock:
                telemetry = self._telemetry[axis]
                if "wrapped_angle" in updates:
                    wrapped_angle = updates["wrapped_angle"]
                    if telemetry["sensor_seen"]:
                        delta = wrapped_angle - telemetry["wrapped_angle"]
                        if delta > math.pi:
                            delta -= 2.0 * math.pi
                        elif delta < -math.pi:
                            delta += 2.0 * math.pi
                        updates["angle"] = telemetry["angle"] + delta
                    else:
                        updates["angle"] = wrapped_angle
                telemetry.update(updates)

    # ── Background I/O thread ─────────────────────────────────────────────────

    def _io_loop(self):
        io_interval = 1.0 / IO_HZ
        send_interval = 1.0 / SEND_HZ
        next_periodic_send = time.monotonic()
        while self.running:
            t0 = time.monotonic()
            try:
                with self._ctrl_lock:
                    due_frames = []
                    while self._ctrl_queue and self._ctrl_queue[0][0] <= t0:
                        _, frame = self._ctrl_queue.pop(0)
                        due_frames.append(frame)
                for frame in due_frames:
                    self.ser.write(frame)
                    self.tx_count += 1

                if t0 >= next_periodic_send:
                    next_periodic_send = t0 + send_interval
                    if self._enabled and self._node_mode == "motor":
                        c = self._cmd
                        self.ser.write(pack_command(c["p"], c["v"], c["kp"], c["kd"], c["t"],
                                                    self._motor_id))
                        self.tx_count += 1
                    elif self._enabled and self._node_mode == "sensor":
                        s = self._sensor_led
                        self.ser.write(pack_sensor_led_frame(s["r"], s["g"], s["b"], self._ctrl_id))
                        self.tx_count += 1

                if self.ser.in_waiting:
                    chunk = self.ser.read(self.ser.in_waiting)
                    self._rx_buf.extend(chunk)

                # Waveshare USB-CAN RX frame: AA C8 [ID_L] [ID_H] [8 data bytes] 55 = 13 bytes
                while len(self._rx_buf) >= 13:
                    if self._rx_buf[0] == 0xAA and self._rx_buf[12] == 0x55:
                        frame = self._rx_buf[:13]
                        del self._rx_buf[:13]
                        fb_id = frame[2] | (frame[3] << 8)   # CAN ID, little-endian
                        d = frame[4:12]                       # 8 CAN data bytes
                        self._parse_lqr_telemetry(fb_id, d)
                        if fb_id == self._feedback_id:
                            self.fb_id    = d[0] & 0x0F          # lower nibble = motor ID
                            self.fb_state = (d[0] >> 4) & 0x0F   # upper nibble = state code
                            self.fb_err   = d[1]
                            self.fb_pos   = uint_to_float((d[2] << 8) | d[3], P_MIN, P_MAX, 16)
                            self.fb_vel   = uint_to_float((d[4] << 8) | d[5], V_MIN, V_MAX, 16)
                            if self._node_mode == "sensor":
                                self.fb_torq = float((d[6] << 8) | d[7])
                            else:
                                self.fb_torq = uint_to_float((d[6] << 8) | d[7], T_MIN, T_MAX, 16)
                            self.fb_count += 1
                    else:
                        del self._rx_buf[0]

            except (serial.SerialException, OSError) as exc:
                self._io_error = str(exc)
                self.running = False
                break

            elapsed = time.monotonic() - t0
            rem = io_interval - elapsed
            if rem > 0:
                time.sleep(rem)

    # ── UI refresh (main thread) ──────────────────────────────────────────────

    def _schedule_ui(self):
        if self.running:
            self._refresh_display()
            self.root.after(1000 // UI_HZ, self._schedule_ui)

    def _refresh_display(self):
        if self._io_error:
            self._log(f"IO error: {self._io_error}")
            self._status_lbl.config(text="● Error", fg=RED)
            self._io_error = ""

        self._fb_val_vars["pos"].set(f"{self.fb_pos:+7.3f}")
        self._fb_val_vars["vel"].set(f"{self.fb_vel:+7.3f}")
        if self._node_mode == "sensor":
            self._fb_val_vars["torq"].set(f"{int(self.fb_torq):7d}")
        else:
            self._fb_val_vars["torq"].set(f"{self.fb_torq:+7.3f}")

        # Error panel: green = OK, red = fault(s)
        err = self.fb_err
        if self._node_mode == "sensor":
            self._fb_err_code_var.set(decode_sensor_faults(err))
        else:
            self._fb_err_code_var.set(decode_faults(err))
        self._fb_err_hex_var.set(f"0x{err:02X}")
        self._fb_err_lbl.config(fg=RED if err else GREEN)

        self._fb_id_var.set(f"Node ID: {self.fb_id}")
        self._fb_rx_var.set(f"RX: {self.fb_count}  frames")
        self._fb_tx_var.set(f"TX: {self.tx_count}  cmds")

        # State badge in toolbar
        s = self.fb_state
        name  = STATE_NAMES.get(s, "?")
        color = STATE_COLORS.get(s, MUTED)
        if self._node_mode == "sensor" and s == 1:
            name = "SENSOR RUN"
        self._state_var.set(f"STATE: {name}")
        self._state_lbl.config(fg=color)

        for key, (canvas, color) in self._bar_canvases.items():
            lo, hi = self._fb_ranges[key]
            val = {"pos": self.fb_pos, "vel": self.fb_vel, "torq": self.fb_torq}[key]
            self._draw_bar(canvas, val, lo, hi, color)

        self._refresh_lqr_monitor()

    @staticmethod
    def _draw_bar(canvas: tk.Canvas, val: float, lo: float, hi: float, color: str):
        canvas.update_idletasks()
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 2:
            return
        canvas.delete("all")
        canvas.create_rectangle(0, 0, w, h, fill=OVERLAY, outline="")

        if lo < 0 < hi:
            # Bipolar: fill from centre
            cx = w * (-lo) / (hi - lo)
            val_px = w * (val - lo) / (hi - lo)
            x0, x1 = (min(cx, val_px), max(cx, val_px))
            fill = color if val >= 0 else RED
            canvas.create_rectangle(x0, 1, x1, h - 1, fill=fill, outline="")
            canvas.create_line(cx, 0, cx, h, fill=MUTED, width=1)
        else:
            x1 = max(0, w * (val - lo) / (hi - lo))
            canvas.create_rectangle(0, 1, x1, h - 1, fill=color, outline="")

    # ── Slider helpers ────────────────────────────────────────────────────────

    def _zero_all(self):
        for v in self._slider_vars.values():
            v.set(0.0)

    def _zero_torque(self):
        self._slider_vars["t"].set(0.0)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    MotorGUI(root)
    root.mainloop()
