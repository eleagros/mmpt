import os, shutil

from tqdm import tqdm

from processingmm import utils

def add_calibration(directories):
    data_folder, _ = utils.get_all_folders(directories)
    
    for folder in tqdm(data_folder):
        
        file = open(os.path.join(folder, 'MMProcessing.txt'), "r")
        calib_directory=file.readlines()[1].replace('\n','')
        assert os.path.exists(calib_directory), 'Calibration directory does not exist'
        file.close()
        
        os.makedirs(os.path.join(folder, 'calibration'), exist_ok=True)
        
        wls_of_interest = ['550nm', '600nm', '650nm']
        for calib_folder in os.listdir(calib_directory):
            if calib_folder in wls_of_interest:
                if len(os.listdir(os.path.join(calib_directory, calib_folder))) == 0:
                    print('No calibration files in %s' % calib_folder)
                else:
                    wl = calib_folder.split('nm')[0]
                    os.makedirs(os.path.join(folder, 'calibration', calib_folder), exist_ok=True)
                    shutil.copy(os.path.join(calib_directory, calib_folder, wl + '_A.cod'), os.path.join(folder, 'calibration', calib_folder, wl + '_A.cod'))
                    shutil.copy(os.path.join(calib_directory, calib_folder, wl + '_W.cod'), os.path.join(folder, 'calibration', calib_folder, wl + '_W.cod'))
