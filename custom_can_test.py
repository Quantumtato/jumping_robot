import serial
import time
import serial.tools.list_ports

def run_custom_test():
    # 1. Auto-detect port
    ports = [p.device for p in serial.tools.list_ports.comports() if 'CH340' in p.description]
    port = ports[0] if ports else 'COM11'
    
    print(f"--- CUSTOM RAW SERIAL CAN TEST ---")
    print(f"Connecting to {port} at 2,000,000 baud...")

    try:
        # Open port at the high-speed TTY rate
        ser = serial.Serial(port, 2000000, timeout=0.1)
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # 2. INITIALIZATION SEQUENCE
    # Many CH340 adapters use simple binary or ASCII commands to start.
    # We will try a few common "Start" sequences.
    
    # Sequence A: Typical SLCAN Start (ASCII 'S8' = 1Mbps, 'O' = Open)
    print("Initializing Adapter...")
    ser.write(b'C\r')      # Close just in case
    time.sleep(0.1)
    ser.write(b'S8\r')     # Set bitrate to 1Mbps
    time.sleep(0.1)
    ser.write(b'O\r')      # Open channel
    time.sleep(0.1)

    print("Sending CAN Data (Spam Mode)... Watch the scope!")
    
    msg_count = 0
    try:
        while True:
            # 3. Construct a raw SLCAN data frame
            # Format: t[ID][Length][Data]\r
            # t = Standard frame
            # 001 = ID 0x01
            # 8 = length 8
            # data = 16 hex chars
            can_frame = b't00181122334455667788\r'
            
            ser.write(can_frame)
            msg_count += 1
            
            if msg_count % 100 == 0:
                print(f"Sent {msg_count} raw frames...", end='\r')
            
            # Check for incoming data (feedback)
            if ser.in_waiting:
                rx = ser.read(ser.in_waiting)
                # Raw feedback usually starts with 't' or 'T'
                if b't' in rx:
                    print(f"\n[FEEDBACK RECEIVED]: {rx.decode(errors='ignore').strip()}")

            time.sleep(0.01) # 100 Hz

    except KeyboardInterrupt:
        print(f"\nStopping. Sent {msg_count} messages.")
        ser.write(b'C\r') # Close command
        ser.close()

if __name__ == "__main__":
    run_custom_test()
