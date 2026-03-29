#! /usr/bin/env python

# Participant 9, 12, 27 left handed
# Participant 21 has fucked grip
# Participant 29 facing east

import sys

from util.port import find_arduino_port
from util.record import record_session
from util.session import parse_arguments, generate_filename

if __name__ == "__main__":
    # Parse Command Line Arguments
    try:
        args = parse_arguments()
    except SystemExit:
        # argparse automatically prints help/errors, we just exit cleanly
        sys.exit(0)

    print("--- Pen Data Recording Script ---")

    # Generate Filename from Args
    filename = generate_filename(args)
    print(f"Session File: {filename}")

    # Determine Port
    port = args.port
    if not port:
        print("No port provided via CLI. Searching...")
        port = find_arduino_port()

    # Start Recording
    if port and filename:
        print(f"\nConnecting to {port}...")
        try:
            record_session(port, filename)
        except KeyboardInterrupt:
            print("\nSession cancelled by user.")
    else:
        print("Exiting without recording.")