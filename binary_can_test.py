import serial
import time
import serial.tools.list_ports

def run_binary_test():
    # 1. Find the CH340 port
    ports = [p.device for p in serial.tools.list_ports.comports() if 'CH340' in p.description]
    port = ports[0] if ports else 'COM11'
    
    print(f"--- WAVESHARE BINARY CAN TEST ---")
    print(f"Connecting to {port} at 2,000,000 baud...")

    try:
        ser = serial.Serial(port, 2000000, timeout=0.1)
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # 2. INITIALIZATION (The "Set and Start" command)
    # Most Waveshare CH340 adapters use this 13-byte binary config packet
    # AA 55 12 [BaudCode] [Mode] 00 00 00 00 00 00 00 [CheckSum]
    # Code for 1Mbps is typically 0x01 or 0x0A depending on the specific model
    print("Configuring Adapter for 1Mbps...")
    
    # We will try the most common "Set 1M and Start" binary frame
    # This is a common pattern for the Waveshare USB-Serial-CAN adapters
    config_packet = bytearray([0xAA, 0x55, 0x12, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x14])
    ser.write(config_packet)
    time.sleep(0.2)

    print("Sending Binary CAN Frames... Watch the scope!")
    
    msg_count = 0
    try:
        while True:
            # 3. Construct a standard CAN Data Frame (13 bytes)
            # [0]   0xAA (Frame Start)
            # [1]   0x08 (Std Frame, Data Frame, DLC=8)
            # [2-3] 0x00 0x01 (CAN ID: 1)
            # [4-11] Data bytes
            # [12]  0x55 (Frame End)
            
            can_frame = bytearray([0xAA, 0x08, 0x00, 0x01, 
                                   0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 
                                   0x55])
            
            ser.write(can_frame)
            msg_count += 1
            
            if msg_count % 100 == 0:
                print(f"Sent {msg_count} binary frames...", end='\r')
            
            # Check for binary feedback (usually 13 byte packets)
            if ser.in_waiting >= 13:
                rx = ser.read(13)
                if rx[0] == 0xAA:
                    print(f"\n[BINARY FEEDBACK RECEIVED]: ID={hex(rx[2]<<8 | rx[3])} Data={rx[4:12].hex()}")

            time.sleep(0.01) # 100 Hz

    except KeyboardInterrupt:
        print(f"\nStopping. Sent {msg_count} messages.")
        ser.close()

if __name__ == "__main__":
    run_binary_test()
