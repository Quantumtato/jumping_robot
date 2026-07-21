import serial
import time
import serial.tools.list_ports

def calculate_checksum(data):
    """Calculates the 8-bit sum checksum for the 20-byte protocol."""
    return sum(data) & 0xFF

def run_waveshare_test():
    # 1. Find the CH340 port
    ports = [p.device for p in serial.tools.list_ports.comports() if 'CH340' in p.description]
    port = ports[0] if ports else 'COM11'
    
    print(f"--- WAVESHARE USB-CAN-A (V1 PROTOCOL) TEST ---")
    print(f"Connecting to {port} at 2,000,000 baud...")

    try:
        ser = serial.Serial(port, 2000000, timeout=0.1)
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # 2. INITIALIZATION (20-byte Configuration Frame)
    # Header: AA 55
    # Cmd: 12
    # Speed: 01 (1Mbps)
    # Format: 01 (Standard)
    # Mask/Filter: 00*8
    # Mode: 00 (Normal)
    # Suffix: 01 00 00 00 00
    # Checksum: Sum of bytes index 2 to 18
    
    config_core = [
        0x12, # Command ID
        0x01, # 1 Mbps
        0x01, # Standard Frame
        0x00, 0x00, 0x00, 0x00, # Filter
        0x00, 0x00, 0x00, 0x00, # Mask
        0x00, # Normal Mode
        0x01, 0x00, 0x00, 0x00, 0x00 # Fixed Suffix
    ]
    checksum = calculate_checksum(config_core)
    config_packet = bytearray([0xAA, 0x55] + config_core + [checksum])
    
    print(f"Initializing Adapter: {config_packet.hex().upper()}")
    ser.write(config_packet)
    time.sleep(0.2)

    # 3. SPAM DATA (Fixed 20-byte Data Protocol)
    # Header: AA 55
    # Type: 01 (Data)
    # Frame Type: 01 (Standard)
    # Frame Format: 00 (Data)
    # ID: 00 00 00 01 (ID 1, 4 bytes)
    # DLC: 08
    # Data: 11 22 33 44 55 66 77 88
    # Reserved: 00
    # Checksum: Sum of bytes index 2 to 18
    
    print("Sending CAN Frames... Watch the scope!")
    
    msg_count = 0
    try:
        while True:
            data_core = [
                0x01, # Type: Data
                0x01, # Frame Type: Standard
                0x00, # Frame Format: Data
                0x00, 0x00, 0x00, 0x01, # ID: 1
                0x08, # DLC: 8
                0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, # Payload
                0x00  # Reserved
            ]
            checksum = calculate_checksum(data_core)
            data_packet = bytearray([0xAA, 0x55] + data_core + [checksum])
            
            ser.write(data_packet)
            msg_count += 1
            
            if msg_count % 100 == 0:
                print(f"Sent {msg_count} packets...", end='\r')
            
            # Check for incoming feedback (20 byte packets)
            if ser.in_waiting >= 20:
                rx = ser.read(20)
                if rx[0] == 0xAA and rx[1] == 0x55:
                    # ID is at index 6-9
                    fb_id = (rx[6] << 24) | (rx[7] << 16) | (rx[8] << 8) | rx[9]
                    fb_data = rx[11:19]
                    print(f"\n[FEEDBACK]: ID={hex(fb_id)} Data={fb_data.hex()}")

            time.sleep(0.01) # 100 Hz

    except KeyboardInterrupt:
        print(f"\nStopping. Sent {msg_count} messages.")
        ser.close()

if __name__ == "__main__":
    run_waveshare_test()
