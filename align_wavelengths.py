import argparse
import os
from processingmm.addons import align_wavelengths

# Set up argument parsing
parser = argparse.ArgumentParser(description="Align wavelengths for the given database path.")

parser.add_argument("--directory", type=str, required=True, help="Path to the database directory")
parser.add_argument("--mode", type=str, choices=['no', 'pddn', 'both'], default='both', help="PDDN mode: 'no', 'pddn', or 'both'")
parser.add_argument("--run_all", action="store_true", help="Process all folders, even if already processed")

# Parse arguments
args = parser.parse_args()

# Print the parsed arguments
print(f"Aligning wavelengths for the database in path: {args.directory}")
print(f"PDDN mode: {args.mode}")
print(f"Run all: {args.run_all}")

# Run wavelength alignment
align_wavelengths.align_wavelenghts([args.directory], args.mode, args.run_all)