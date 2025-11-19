#! /usr/bin/env python

import sys

from port import find_arduino_port
from record import record_session
from session import parse_arguments, generate_filename

if __name__ == "__main__":
    # 1. Parse Command Line Arguments
    try:
        args = parse_arguments()
    except SystemExit:
        # argparse automatically prints help/errors, we just exit cleanly
        sys.exit(0)

    print("--- Pen Data Recording Script ---")

    # 2. Generate Filename from Args
    filename = generate_filename(args)
    print(f"Session File: {filename}")

    # 3. Determine Port (CLI argument OR Interactive fallback)
    port = args.port
    if not port:
        print("No port provided via CLI. Searching...")
        port = find_arduino_port()

    # 4. Start Recording
    if port and filename:
        print(f"\nConnecting to {port}...")
        try:
            record_session(port, filename)
        except KeyboardInterrupt:
            print("\nSession cancelled by user.")
    else:
        print("Exiting without recording.")