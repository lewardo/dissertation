import serial

def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("ERROR: No serial ports found")
        return None

    print("Available ports:")
    for i, port in enumerate(ports):
        print(f"  [{i}] {port.device}: {port.description}")

    while True:
        try:
            choice = int(input("Enter the port: "))
            if 0 <= choice < len(ports):
                return ports[choice].device
            else:
                print(f"Invalid. Enter a number between 0 and {len(ports)-1}")
        except ValueError:
            print("Invalid. Enter a number")
        except KeyboardInterrupt:
            print("\nExiting")
            return None