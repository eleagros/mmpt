import os
from datetime import datetime
import warnings

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


def get_calibration_directory(calib_directory_dates_num, path, calib_directory, directories, Flag = False, idx = -1):
    """
    get the calibration directory with the date closest to the folder given as in input

    Parameters
    ----------
    calib_directory_dates_num : list
        a list containing the dates of the calibration
    path : str
        the path to the folder that is currently being processed
    calib_directory : str
        the path to the folder containing the calibration data
    idx : int
        should always be -1, except for quality check
        
    Returns
    -------
    cal_fol_rt : path
        the path to the calibration directory with the date closest to the folder given as in input
    """
    # get the date of the measurement of the folder that is currently being processed
    date_measurement = get_date_measurement(path, idx)
    
    # find the data that is the closest in the calibration directory
    closest_date = find_closest_date(date_measurement, calib_directory_dates_num, directories, calib_directory, Flag = Flag)
    directories_calib = os.listdir(calib_directory)
    calib_folder = []
    
    # iterate over the directories in the calibration folder
    for directory in directories_calib:
        
        # add to the calib_folder list the folders with the date corresponding to the closest date
        if closest_date in directory:
            
            all_wl = True
            for d in directories:
                path_A = os.path.join(calib_directory, directory, d.split('\\')[-1], d.split('\\')[-1].split('nm')[0] + '_B0.cod')
                path_W = os.path.join(calib_directory, directory, d.split('\\')[-1], d.split('\\')[-1].split('nm')[0] + '_Bruit.cod')
                if os.path.isfile(path_A) and os.path.isfile(path_W):
                    pass
                else:
                    all_wl = False
                
            if all_wl:
                calib_folder.append(os.path.join(calib_directory, directory))
            
    # check if there are actually calibration folders - if not, raise an error
    if calib_folder:
        
        # select the last calibration of the day (i.e. if 'C_1', 'C_2' exists - select 'C_2')
        highest = 0
        cal_fol_rt = None
        for cal_fol in calib_folder:
            if int(cal_fol.split('_')[-1]) > highest:
                highest = int(cal_fol.split('_')[-1])
                cal_fol_rt = cal_fol
        if cal_fol_rt:
            return cal_fol_rt
        else:
            raise FileNotFoundError('No calibration was found. Check if calibration folders are present.')
    else:
        raise FileNotFoundError('No calibration was found. Check if calibration folders are present.')

def get_date_measurement(path, idx = -1):
    """
    returns the data of the measured folder

    Parameters
    ----------
    path : str
        the folder name for which we want the date for
    idx : int
        parameter indicating where to split the path string

    Returns
    -------
    date : datetime
        the date corresponding to the folder given as an input
    """
    try:
        date = datetime.strptime(path.split('\\')[idx].split('_')[0], '%Y-%m-%d')
    except:
        date = datetime.strptime(path.split('/')[idx].split('_')[0], '%Y-%m-%d')
    return date


def find_closest_date(date_measurement, calib_dates, directories, calib_directory, Flag = False):
    """
    finds the closest date in a list given a date as an input

    Parameters
    ----------
    date_measurement : datetime
        the date
    calib_dates : list of datetime
        a list of datetimes in which the closest one to date_measurement should be found

    Returns
    -------
    closest_date : datetime
        the closest date to date_measurement in calib_dates
    """
    directories_calib = os.listdir(calib_directory)
    
    found = False
    while not found:
        
        closest = 1000
        date_calibration = None
        idx = None
        
        # iterate over the dates in the list
        for idy, calib in enumerate(calib_dates):
            duration = calib - date_measurement 

            # get the number of days separating the calibration and measurement
            days = abs(duration.days)
            if days < closest:
                date_calibration = calib
                closest = days
                idx = idy
                
        closest_date = date_calibration.strftime('%Y-%m-%d')
        
        # iterate over the directories in the calibration folder
        for directory in directories_calib:

            # add to the calib_folder list the folders with the date corresponding to the closest date
            if closest_date in directory:

                all_wl = True
                for d in directories:
                    path_A = os.path.join(calib_directory, directory, d.split('\\')[-1], d.split('\\')[-1].split('nm')[0] + '_B0.cod')
                    path_W = os.path.join(calib_directory, directory, d.split('\\')[-1], d.split('\\')[-1].split('nm')[0] + '_Bruit.cod')
                    if os.path.isfile(path_A) and os.path.isfile(path_W):
                        pass
                    else:
                        all_wl = False
             
        if all_wl: 
            found = True
        else:
            del calib_dates[idx]
    
    if date_calibration == None:
        raise FileNotFoundError('No calibration was found for the closest 1000 days. Check if calibration folders are present.')
    
    # if no calibration can be found for the same day, raise a warning
    if closest > 0:
        incorrect_date(closest, Flag = Flag)
    closest_date = date_calibration.strftime('%Y-%m-%d')

    return closest_date


def incorrect_date(closest, Flag = False):
    """
    raises a warning indicating that no calibration has been found for the exact date of the measurement

    Parameters
    ----------
    closest : int
        the number of days separating the measurement and the calibration
    """
    if Flag:   
        warnings.warn('No calibration was found for the exact date, the one used was {} day(s) ago.'.format(closest), UserWarning, stacklevel=2)
    else:
        pass
