import argparse
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description="Pen Data Recording Script")
    
    # Required Arguments
    parser.add_argument("-n", "--participant", type=int, required=True, help="Participant Number (e.g., 1)")
    parser.add_argument("-t", "--trial", type=int, required=True, help="Trial Number (e.g., 1)")
    parser.add_argument("-d", "--descriptor", type=str, required=True, help="Trial Descriptor (e.g., spiral)")
    parser.add_argument("-s", "--pss", type=int, required=True, help="PSS Score (whole number)")
    parser.add_argument("-f", "--fss", type=int, required=True, help="FSS Score (whole number)")
    parser.add_argument("-p", "--port", type=str, default="/dev/ttyACM0", help="Serial Port (e.g., COM3 or /dev/ttyACM0)")

    return parser.parse_args()

def generate_filename(args):
    timestamp = time.strftime("%m%d%H%M") 
    filename = f"P{args.participant:03d}_T{args.trial:03d}_{args.descriptor}_pss-{args.pss:02d}_fss-{args.fss:02d}_{timestamp}.csv"
    return filename
