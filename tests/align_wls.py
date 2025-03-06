import sys
import os

# set the parameters to run the script
directories = os.path.join(os.path.dirname(__file__), "data")

print('Aligning wavelengths for the database in path: ', directories)

from processingmm.addons import align_wavelengths
align_wavelengths.align_wavelengths([directories], 'both', True, [600,650])