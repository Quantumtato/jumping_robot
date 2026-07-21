import serial
import time
import serial.tools.list_ports

# --- Motor Parameters (Must match STM32 firmware) ---
P_MIN, P_MAX = -12.5, 12.5
V_MIN, V_MAX = -45.0, 45.0
KP_MIN, KP_MAX = 0.0, 500.0
KD_MIN, KD_MAX = 0.0, 15.0
T_MIN, T_MAX = -18.0, 18.0

MOTOR_ID = 0x01
FEEDBACK_ID = 0x101

def calculate_checksum(data):
    """Calculates the 8-bit sum checksum for the 20-byte protocol."""
    return sum(data) & 0xFF

def float_to_uint(x, x_min, x_max, bits):
    """Maps a float value to an unsigned integer of specified bit depth."""
    x = max(min(x, x_max), x_min)
    span = x_max - x_min
    return int((x - x_min) * ((1 << bits) - 1) / span)

def uint_to_float(x_int, x_min, x_max, bits):
    """Maps an unsigned integer back to a float value."""
    span = x_max - x_min
    return x_min + float(x_int) * span / ((1 << bits) - 1)

def pack_waveshare_frame(p, v, kp, kd, t):
    """Packs the MIT-style command into a 20-byte Waveshare frame."""
    p_int = float_to_uint(p, P_MIN, P_MAX, 16)
    v_int = float_to_uint(v, V_MIN, V_MAX, 12)
    kp_int = float_to_uint(kp, KP_MIN, KP_MAX, 12)
    kd_int = float_to_uint(kd, KD_MIN, KD_MAX, 12)
    t_int = float_to_uint(t, T_MIN, T_MAX, 12)
    
    can_data = bytearray(8)
    can_data[0] = (p_int >> 8) & 0xFF
    can_data[1] = p_int & 0xFF
    can_data[2] = (v_int >> 4) & 0xFF
    can_data[3] = ((v_int & 0x0F) << 4) | ((kp_int >> 8) & 0x0F)
    can_data[4] = kp_int & 0xFF
    can_data[5] = (kd_int >> 4) & 0xFF
    can_data[6] = ((kd_int & 0x0F) << 4) | ((t_int >> 8) & 0x0F)
    can_data[7] = t_int & 0xFF
    
    # ID is 4 bytes at indices 5-8 in the 20-byte packet
    # core = [Type, FrameType, FrameFormat, ID_0, ID_1, ID_2, ID_3, DLC, D0...D7, Reserved]
    core = [0x01, 0x01, 0x00, 0x00, 0x00, 0x00, MOTOR_ID, 0x08] + list(can_data) + [0x00]
    checksum = calculate_checksum(core)
    return bytearray([0xAA, 0x55] + core + [checksum])

def run_motor_test():
    ports = [p.device for p in serial.tools.list_ports.comports() if 'CH340' in p.description]
    port = ports[0] if ports else 'COM11'
    
    print(f"--- MOTOR IMPEDANCE TEST (WAVESHARE V1) ---")
    print(f"Connecting to {port} at 2,000,000 baud...")

    try:
        ser = serial.Serial(port, 2000000, timeout=0.01)
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # INITIALIZATION (Set 1Mbps and Start)
    config_core = [
        0x12, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x01, 0x00, 0x00, 0x00, 0x00
    ]
    checksum = calculate_checksum(config_core)
    ser.write(bytearray([0xAA, 0x55] + config_core + [checksum]))
    time.sleep(0.2)

    print(f"Connected to Motor ID {MOTOR_ID}. Sending commands...")
    print("Press Ctrl+C to stop.")

    rx_buffer = bytearray()
    try:
        while True:
            # Send command
            cmd_packet = pack_waveshare_frame(0.0, 0.0, 0.0, 0.0, 5000.0)  # 1 Nm FF torque, no impedance
            ser.write(cmd_packet)
            time.sleep(0.005)

            # Robust parser for 13-byte binary frames: AA [type] [ID_H] [ID_L] [D0..D7] 55
            if ser.in_waiting > 0:
                rx_buffer.extend(ser.read(ser.in_waiting))

                while len(rx_buffer) >= 13:
                    if rx_buffer[0] == 0xAA and rx_buffer[12] == 0x55:
                        frame = rx_buffer[:13]
                        del rx_buffer[:13]

                        fb_id = (frame[2] << 8) | frame[3]

                        if fb_id == FEEDBACK_ID:
                            d = frame[4:12]  # 8 CAN data bytes
                            # Firmware layout: [motor_id, error_code, p_hi, p_lo, v_hi, v_lo, t_hi, t_lo]
                            motor_id  = d[0]
                            err_code  = d[1]
                            p_int = (d[2] << 8) | d[3]
                            v_int = (d[4] << 8) | d[5]
                            t_int = (d[6] << 8) | d[7]

                            p = uint_to_float(p_int, P_MIN, P_MAX, 16)
                            v = uint_to_float(v_int, V_MIN, V_MAX, 16)
                            t = uint_to_float(t_int, T_MIN, T_MAX, 16)

                            print(f"ID:{motor_id} Err:{err_code:3d} | POS: {p:6.3f} rad | VEL: {v:6.3f} rad/s | TORQ: {t:6.3f} Nm  raw={d.hex()}", end='\r')
                    else:
                        del rx_buffer[0]

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping Motor...")
        stop_packet = pack_waveshare_frame(0.0, 0.0, 0.0, 0.0, 0.0)
        ser.write(stop_packet)
        ser.close()

if __name__ == "__main__":
    run_motor_test()
