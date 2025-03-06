import argparse
import os
from processingmm.addons import plot_polarimetry

# Set up argument parsing
parser = argparse.ArgumentParser(description="Align wavelengths for the given database path.")

parser.add_argument("--path_folder", type=str, required=True, help="Path to the folder in which the data is saved")
parser.add_argument("--processing_mode", type=str, choices=['no_viz', 'default', 'full'], default='default', help="Processing mode")
parser.add_argument("--save_pdf_figs", action="store_true", help="Save pdf files")

# Parse arguments
args = parser.parse_args()

# Call the visualization function
plot_polarimetry.visualize_MM(
    path_save=args.path_folder,
    MM_path=os.path.join(args.path_folder, 'MM.npz'),
    processing_mode=args.processing_mode,  # Set a default or extract from elsewhere
    save_pdf_figs=args.save_pdf_figs
)