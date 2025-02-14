import sys
import os

print(sys.argv)
pathDB = os.path.abspath(sys.argv[0]).split('processingMM')[0]

print('Aligning wavelengths for the database in path: ', pathDB)

from processingmm.addons import align_wavelengths
align_wavelengths.align_wavelenghts([pathDB], 'both', False)