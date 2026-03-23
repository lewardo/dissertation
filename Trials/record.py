import serial
import csv
import time
import sys
import os

def record_session(port, filename):    
    ser = None 
    
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        # time.sleep(1)
        
        ser.reset_input_buffer()
        while ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            if line:
                print(f"[Arduino] {line}")
        
        print("\n--- Arduino Ready ---")
        print("Press ENTER to START recording")
        input()
        
        ser.write(b'S')

        start_time = time.time()
        
        print("RECORDING. Press Ctrl+C to STOP")
        print("-------------------------------")

        print("PS, LA_X, LA_Y, LA_Z, G_X, G_Y, G_Z, Q_R, Q_I, Q_J, Q_K", end="")
        sys.stdout.flush()

        filepath = os.path.join(os.path.curdir, os.pardir, "Data", "Series", "00-Raw", filename)            
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "pressure_value",
                "linear_accel_x", "linear_accel_y", "linear_accel_z",
                "gyro_x", "gyro_y", "gyro_z",
                "orientation_r", "orientation_i", "orientation_j", "orientation_k"
            ])

            prev_time = time.time()
            sample_num = 0
            while True:
                if time.time() - prev_time > 0.03:
                    prev_time = time.time()
                    sample_num += 1

                    sys.stdout.write('\r' + ' ' * 120 + '\r')
                    print(f"*** Sample {sample_num} Dropped ***")
                    
                    continue

                if ser.in_waiting > 0:
                    try:
                        line = ser.readline().decode('utf-8').strip()
                        if not line:
                            continue

                        prev_time = time.time()
                        
                        if line.startswith("INFO:"):
                            sys.stdout.write('\r' + ' ' * 120 + '\r')
                            print(f"[Arduino] {line}")
                            print("PS, LA_X, LA_Y, LA_Z, G_X, G_Y, G_Z, Q_R, Q_I, Q_J, Q_K", end="")
                            sys.stdout.flush()
                        
                        else:
                            values = line.split(',')
                            if len(values) == 11:
                                sample_num += 1

                                timestamp = time.time() - start_time
                                writer.writerow([timestamp] + values)
                                
                                f_vals = [f"{float(v):>8.3f}" for v in values]
                                out_str = ", ".join([f"{timestamp:>8.3f}"] + f_vals)

                                sys.stdout.write('\r' + out_str)
                                sys.stdout.flush()
                                
                    except UnicodeDecodeError:
                        sys.stdout.write('\r' + '!' * 120 + '\r') # Show error
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
