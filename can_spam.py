import can
import time
import serial.tools.list_ports

def spam_can():
    # 1. Auto-detect the CH340 adapter
    ports = [p.device for p in serial.tools.list_ports.comports() if 'CH340' in p.description]
    port = ports[0] if ports else 'COM11'
    
    print(f"--- CAN LED BLINK TEST ---")
    print(f"Targeting Adapter on {port} at 2Mbps serial...")
    
    try:
        # Initialize at 1Mbps CAN with 2Mbps Serial
        bus = can.interface.Bus(interface='slcan', channel=port, bitrate=1000000, tty_baudrate=2000000)
    except Exception as e:
        print(f"ERROR: Could not open port {port}. Is another program (Motor Pilot) using it?")
        print(f"Details: {e}")
        return

    print("SPAMMING START: Watch your adapter LEDs!")
    print("Press Ctrl+C to stop.")

    msg_count = 0
    try:
        while True:
            # Create a dummy message with a recognizable ID
            # We use a 1ms delay (1000 Hz) to make the LED stay solid/bright
            msg = can.Message(
                arbitration_id=0x7FF, 
                data=[0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11], 
                is_extended_id=False
            )
            
            try:
                bus.send(msg)
                msg_count += 1
                if msg_count % 100 == 0:
                    print(f"Sent {msg_count} messages...", end='\r')
            except can.CanError:
                print("\nTransmission Error: Is the CAN bus terminated or shorted?")
                time.sleep(1)
            
            time.sleep(0.001) 

    except KeyboardInterrupt:
        print(f"\nStopped. Total sent: {msg_count}")
        bus.shutdown()

if __name__ == "__main__":
    spam_can()
