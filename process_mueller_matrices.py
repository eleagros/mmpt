import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Process Mueller Matrices with optional PDDN mode and calibration settings.")

parser.add_argument("--input_dir", type=str, help="Path to the main processing directory")
parser.add_argument("--calib_dir", type=str, help="Path to the calibration directory")
parser.add_argument("--instrument", type=str, choices=['IMP', 'IMPv2'], default='IMP', help="Name of the instrument used (default: 'IMP')")
parser.add_argument("--PDDN_mode", type=str, choices=['no', 'pddn', 'both'], default='no', help="Processing mode: 'no', 'pddn', or 'both'")
parser.add_argument("--wavelengths", type=int, nargs='+', default=[550, 600, 650], help="List of wavelengths to process (default: 550, 600)")
parser.add_argument("--workflow_mode", type=str, choices=['no_viz', 'default', 'full'], default='default', help="Workflow_mode: 'no_viz', 'default', or 'full'")
parser.add_argument("--save_pdf_figs", action="store_true", help="Save PDF figures (default: False)")
parser.add_argument("--force_reprocess", action="store_true", help="Process all folders, even if already processed")
parser.add_argument("--align_wls", action="store_true", help="Aligns the measurements with the 550nm ones before processing")
parser.add_argument("--mm_backend", type=str, choices=['c', 'torch'], default='c', help="Mueller matrix computation backend (default: 'c')")
parser.add_argument("--lu_chipman_backend", type=str, choices=['processing', 'prediction'], default='processing', help="Lu-Chipman backend (default: 'processing')")
parser.add_argument("--binning_factor", type=int, default=1, help="Binning factor for the images (default: 1)")

# Parse arguments
args = parser.parse_args()

# Ensure required arguments are provided
if not args.input_dir or not args.calib_dir:
    parser.error("--input_dir and --calib_dir are required.")

# Print the parsed arguments
print(f"Processing Mueller Matrices for the folder: {args.input_dir}")
print(f"Calibration directory: {args.calib_dir}")
print(f"Instrument: {args.instrument}")
print(f"PDDN mode: {args.PDDN_mode}")
print(f"Wavelengths: {args.wavelengths}")
print(f"Processing mode: {args.workflow_mode}")
print(f"Save PDF Figures: {args.save_pdf_figs}")
print(f"Force reprocess: {args.force_reprocess}")

from mmpt import processingmm


# online sets up processing when the data is already loaded in memory (i.e. when processing "live")
data_source = 'offline'
instrument = args.instrument
input_dirs = [args.input_dir]
calib_dir = args.calib_dir
force_reprocess = args.force_reprocess
PDDN_mode = args.PDDN_mode
mm_computation_backend = args.mm_backend
lu_chipman_backend = args.lu_chipman_backend
wavelengths = args.wavelengths
workflow_mode = args.workflow_mode
save_pdf_figs = args.save_pdf_figs
align_wls = args.align_wls
remove_reflection = False
binning_factor = 1

mmProcessor = processingmm.MuellerMatrixProcessor(
    data_source = data_source,
    wavelengths = wavelengths,
    input_dirs = input_dirs,
    calib_dir = calib_dir,
    PDDN_mode = PDDN_mode,
    instrument = instrument,
    remove_reflection = remove_reflection, 
    workflow_mode = workflow_mode,
    force_reprocess = force_reprocess,
    save_pdf_figs = save_pdf_figs, 
    align_wls = align_wls,
    binning_factor = binning_factor,
    mm_computation_backend = mm_computation_backend,
    lu_chipman_backend = lu_chipman_backend,
    prediction_mode = None
)

mmProcessor.batch_process_master()