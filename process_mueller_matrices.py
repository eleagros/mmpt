import argparse
from processingmm import batch_processing

# Set up argument parsing
parser = argparse.ArgumentParser(description="Process Mueller Matrices with optional PDDN mode and calibration settings.")

parser.add_argument("--directory", type=str, required=True, help="Path to the main processing directory")
parser.add_argument("--calib", type=str, required=True, help="Path to the calibration directory")
parser.add_argument("--PDDN_mode", type=str, choices=['no', 'pddn', 'both'], default='no', help="Processing mode: 'no', 'pddn', or 'both'")
parser.add_argument("--wavelengths", type=int, nargs='+', default=[550, 600], help="List of wavelengths to process (default: 550, 600)")
parser.add_argument("--processing_mode", type=str, choices=['no_viz', 'default', 'full'], default='default', help="Processing mode")
parser.add_argument("--save_pdf_figs", action="store_true", help="Save PDF figures (default: False)")
parser.add_argument("--run_all", action="store_true", help="Process all folders, even if already processed")
parser.add_argument("--align_wls", action="store_true", help="Aligns the measurements with the 550nm ones before processing")


# Parse arguments
args = parser.parse_args()

# Print the parsed arguments
print(f"Processing Mueller Matrices for the folder: {args.directory}")
print(f"Calibration directory: {args.calib}")
print(f"PDDN mode: {args.PDDN_mode}")
print(f"Wavelengths: {args.wavelengths}")
print(f"Processing mode: {args.processing_mode}")
print(f"Save PDF Figures: {args.save_pdf_figs}")
print(f"Run all: {args.run_all}")

# Get parameters
parameters = batch_processing.get_parameters(
    directories=[args.directory], 
    calib_directory=args.calib, 
    wavelengths=args.wavelengths, 
    parameter_set='default',
    PDDN_mode=args.PDDN_mode, 
    processing_mode=args.processing_mode, 
    run_all=args.run_all, 
    save_pdf_figs=args.save_pdf_figs,
    align_wls = args.align_wls,
)

# Run batch processing
batch_processing.batch_process_master(parameters)