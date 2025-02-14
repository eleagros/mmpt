import sys
import os

pathDB = os.path.abspath(sys.argv[0]).split('processingMM')[0]

print('Processing the Mueller Matrices for the folder in path in path: ', pathDB)

import os
from processingmm import batch_processing

# set the parameters to run the script
directories = [pathDB]
calib_directory = f"{pathDB}/calib"
    
# set the parameters to be used for the line visualisation
# NB: parameter file accessible in ./src/processingmm/data/parameters_visualisations.json
parameter_set = 'default'

# set run_all to true in order to run the pipeline on all the folders (even the ones already processed)
run_all = False

# PDDN_mode can be set to:
# 1. 'no': processes without using the PDDN
# 2. 'pddn': processes with PDDN when available (for 550nm and 650nm)
# 3. 'both': processes both with PDDN when available and without PDDN
if len(sys.argv) > 1:
    PDDN_mode = 'no' if sys.argv[1] else 'pddn'
else:
    PDDN_mode = 'no'

# Set the wavelengths to be processed
# 1. 'all': processes all the available wavelenght
# 2. [xxx, yyy]: processes only the wavelenghts 'xxx' and 'yyy'
wavelengths = [550,600,650]

# Processing mode
# 1. 'no_viz': processes only the MM - no visualization at all. useful for fast computation
# 2. 'default': processes the MM and plots the polarimetric parameters maps (i.e. depolarization, azimuth, 
# retardance, diattenuation, azimuth local variability)
# 3. 'full': do like default, and additionally plot the MM components, as well as the line
# visualization

# define if pdf figures should be saved (takes a lot of time) - no impact when processing_mode is set to no_viz

# NB: processing time without PDDN
# 'no_viz': 0.71s
# 'default', save_pdf_figs False: 2.25s
# 'default', save_pdf_figs True: 3.60s
# 'full', save_pdf_figs False: 3.95s
# 'full', save_pdf_figs True: 8.07s
processing_mode = 'default'
save_pdf_figs = False

parameters = batch_processing.get_parameters(directories, calib_directory, wavelengths, parameter_set = parameter_set, 
                                PDDN_mode = PDDN_mode, processing_mode = processing_mode, run_all = run_all, 
                                save_pdf_figs = save_pdf_figs)
batch_processing.batch_process_master(parameters)
