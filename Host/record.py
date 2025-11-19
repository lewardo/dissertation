import serial
import csv
import time
import sys
import os

def record_session(port, filename):    
    ser = None 
    
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)
        
        ser.reset_input_buffer()
        while ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            if line:
                print(f"[Arduino] {line}")
        
        print("\n--- Arduino Ready ---")
        print("Press ENTER to START calibrating")
        input()
        
        ser.write(b'S')

        print("--- Waiting for Calibration ---")

        while True:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    if not line:
                        continue
                    
                    if line.startswith("CAL:") or line.startswith("INFO:"):
                        print(f"[Arduino] {line}")
                        sys.stdout.flush()
                    
                    if line.find("High Accuracy") > -1:
                        break

                except Exception as e:
                    print(f"\nError processing line: {line}\n{e}")

        print("--- Calibrated, press ENTER to start recording ---")
        input()
        
        start_time = time.time()
        
        print("RECORDING. Press Ctrl+C to STOP")
        print("-------------------------------")

        print("WA_X, WA_Y, WA_Z, WG_X, WG_Y, WG_Z", end="")
        sys.stdout.flush()

        filepath = os.path.join(os.path.curdir, os.pardir, "Data", filename)            
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", 
                "world_accel_x", "world_accel_y", "world_accel_z",
                "world_gyro_x", "world_gyro_y", "world_gyro_z"
            ])

            while True:                
                if ser.in_waiting > 0:
                    try:
                        line = ser.readline().decode('utf-8').strip()
                        if not line:
                            continue
                        
                        if line.startswith("CAL:") or line.startswith("INFO:"):
                            sys.stdout.write('\r' + ' ' * 80 + '\r')
                            print(f"[Arduino] {line}")
                            print("WA_X, WA_Y, WA_Z, WG_X, WG_Y, WG_Z", end="")
                            sys.stdout.flush()
                        
                        else:
                            values = line.split(',')
                            if len(values) == 6:
                                timestamp = time.time() - start_time
                                writer.writerow([timestamp] + values)
                                
                                f_vals = [f"{float(v):>8.3f}" for v in values]
                                out_str = ", ".join(f_vals)

                                sys.stdout.write('\r' + out_str)
                                sys.stdout.flush()
                                
                    except UnicodeDecodeError:
                        sys.stdout.write('\r' + '!' * 80 + '\r') # Show error
                        sys.stdout.flush()
                    except Exception as e:
                        print(f"\nError processing line: {line}\n{e}")

    except KeyboardInterrupt:
        print("\n\n--- Stopping... ---")
        if ser:
            ser.write(b'X')
        print(f"Recording STOPPED. Data saved to {filepath}")

    except serial.SerialException as e:
        print(f"\n--- ERROR ---")
        print(f"Failed to connect to {port}: {e}")
        print("Please check the port and ensure the Arduino is not in use.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    
    finally:
        if ser and ser.is_open:
            ser.write(b'X') # Ensure stop command is sent
            ser.close()
            print("Serial port closed.")
