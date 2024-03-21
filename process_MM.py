import os
from processingmm import batch_processing, libmpMuelMat
import sys

if len(sys.argv) != 3 and len(sys.argv) != 4:
    raise ValueError("Usage: python program_name.py data_directory calib_directory force_recompute (optional)")
else:
    directories = [sys.argv[1]]
    calib_directory = sys.argv[2]
    try:
        run_all = sys.argv[3]
    except:
        run_all = False

    # set run_all to true in order to run the pipeline on all the folders (even the ones already processed)
    batch_processing.batch_process(directories, calib_directory, run_all = run_all, parameter_set = 'TheoniPics', PDDN = False)