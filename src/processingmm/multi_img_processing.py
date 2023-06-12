import os
from datetime import datetime

def get_dir_to_compute(measurements_directory, treshold = 21, sanity = False, run_all = False):
    """
    get the directories to compute MM for (i.e. directories missing the polarimetric results)

    Parameters
    ----------
    measurements_directory : str
        the path to the measurement directory

    Returns
    -------
    to_compute : list
        the list of folders to be computed
    """
    if run_all:
        treshold = 1000
    to_compute = remove_already_computed_folders(measurements_directory, os.listdir(measurements_directory), treshold, sanity)
    return to_compute


def remove_already_computed_folders(measurements_directory, to_compute, treshold = 14, sanity = False):
    """
    check for each folder if it was already computed - if yes, remove it from the list of folders to compute

    Parameters
    ----------
    measurements_directory : str
        the path to the measurement directory
    to_compute : list
        the list of folders to be computed
    sanity : bool
        not used anymore

    Returns
    -------
    not_computed : list
        the list of folders to be computed (i.e. the folders that have not been computed yet)
    """
    not_computed = []
    
    # iterate over the list of folders to be computed
    for c in to_compute:
        path = os.path.join(measurements_directory, c)
        
        # get the directories that are not computed yet
        directories = remove_already_computed_directories(path, treshold = treshold, sanity = sanity)
        if len(directories) >= 1:
            not_computed.append(c)
    return not_computed


def remove_already_computed_directories(path, treshold = 18, sanity = False):
    """
    check if all wavelengths for a given folder was already computed - return the directories that needs to be processed

    Parameters
    ----------
    measurements_directory : str
        the path to the measurement directory
    to_compute : list
        the list of folders to be computed
    sanity : bool
        not used anymore

    Returns
    -------
    directories_computable : list
        the list of directories to be computed (i.e. the folders that have not been computed yet)
    """

    # get the directories for which raw data is available
    path_raw_data = os.path.join(path, 'raw_data')
    directories = get_data_containing_folder(path_raw_data)
    
    path_50x50 = os.path.join(path, '50x50_images')
    path_polarimetry = os.path.join(path, 'polarimetry') 
    
    directories_computable = []
    
    # iterate over these directories
    for d in directories:
        if sanity:
            directories_computable.append(os.path.join(path_raw_data, d))
        else:
            
            # check if polarimetry was obtained
            dir_list = os.listdir(os.path.join(path_polarimetry, d))
            if len(dir_list) >= treshold:
                pass
            else:
                directories_computable.append(os.path.join(path_raw_data, d))
              
    directories_computable = list(set(directories_computable))
    return directories_computable


def get_data_containing_folder(path):
    """
    get the wavelength folders containing raw data

    Parameters
    ----------
    path : str
        the path to the folder of interest

    Returns
    -------
    directories : list
        the list of wavelength folders to be computed
    """
    filenames = os.listdir(path)
    directories = []
    
    for filename in filenames:
        if os.path.isdir(os.path.join(path, filename)):
            
            # check if the raw data is available for each wavelength
            dir_empty = len(os.listdir(os.path.join(path, filename))) == 0
            if not dir_empty:
                directories.append(filename)
    return directories


def get_calibration_dates(calib_directory):
    """
    returns the dates of all the calibration folders containing calibration data for 550nm and 650nm in the calibration directory

    Parameters
    ----------
    calib_directory : str
        the path to the calibration directory

    Returns
    -------
    calib_directory_dates_num : list of datetime
        a list containing the dates of the calibration folders
    """
    
    # get all the folders
    calib_directory_dates = os.listdir(calib_directory)
    calib_directory_dates_cleaned = []
    for c in calib_directory_dates:
        
        # check if calibration was obtained for both wavelengths
        if os.path.exists(os.path.join(calib_directory, c, '550nm', '550_W.mat')) and os.path.exists(os.path.join(calib_directory, c, '650nm', '650_W.mat')):
            calib_directory_dates_cleaned.append(c)
        else:
            pass
   
    # convert the folder names to datetimes and return the list containing the dates
    calib_directory_dates_num = []
    for c in calib_directory_dates:
        calib_directory_dates_num.append(datetime.strptime(c.split('_')[0], '%Y-%m-%d'))
    return calib_directory_dates_num