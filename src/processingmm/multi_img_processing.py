import os
import numpy as np
from datetime import datetime
import warnings
import copy
from processingmm.helpers import load_wavelengths, is_there_data, is_processed


def remove_already_computed_folders(measurements_directory: str, sanity = False, run_all: bool = False):
    """
    check for each folder if it was already computed - if yes, remove it from the list of folders to compute

    Parameters
    ----------
    measurements_directory : str
        the path to the measurement directory
    to_compute : list
        the list of folders to be computed

    Returns
    -------
    not_computed : list
        the list of folders to be computed (i.e. the folders that have not been computed yet)
    """
    to_compute = os.listdir(measurements_directory)
    not_computed = []
    
    # iterate over the list of folders to be computed
    for c in to_compute:
        path = os.path.join(measurements_directory, c)
        
        # get the directories that are not computed yet
        directories = remove_already_computed_directories(path, sanity = sanity, run_all = run_all)
        if len(directories) >= 1:
            not_computed.append(c)
    return not_computed


def remove_already_computed_directories(path: str, sanity = False, run_all: bool = False):
    """
    remove the folders and wl that were previsouly computed

    Parameters
    ----------
    path : str
        the path to the measurement directory
    sanity : bool

    Returns
    -------
    directories_computable : list
        the list of directories to be computed (i.e. the folders that have not been computed yet)
    """
    # get the directories for which raw data is available
    path_raw_data = os.path.join(path, 'raw_data')
    directories = get_data_containing_folder(path_raw_data)

    directories_computable = []
    
    # iterate over these directories
    for d in directories:
        if sanity or run_all:
            directories_computable.append(os.path.join(path_raw_data, d))
        else:
            if is_processed(path, d) or sanity and not run_all:
                pass
            else:
                directories_computable.append(os.path.join(path, d))
    directories_computable = list(set(directories_computable))
    return directories_computable


def get_data_containing_folder(path: str):
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
    for wavelength in load_wavelengths():
        if os.path.isdir(os.path.join(path, wavelength)):
            if is_there_data(os.path.join(path, wavelength)):
                directories.append(wavelength)
    return directories


def get_calibration_dates(calib_directory: str):
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
    wavelenghts = load_wavelengths()

    calib_directory_dates_cleaned = []
    for c in calib_directory_dates:
        path_calib_directory = os.path.join(calib_directory, c)
        calibration_found = False
        for wl in wavelenghts:
            if os.path.exists(os.path.join(path_calib_directory, wl, wl.split('nm')[0] + '_W.mat')) or os.path.exists(os.path.join(path_calib_directory, 
                                                                                                                    wl, wl.split('nm')[0] + '_W.cod')):
                calibration_found = True
            else:
                pass
        if calibration_found:
            calib_directory_dates_cleaned.append(c)

    # convert the folder names to datetimes and return the list containing the dates
    calib_directory_dates_num = []
    for c in calib_directory_dates:
        calib_directory_dates_num.append(datetime.strptime(c.split('_')[0], '%Y-%m-%d'))
    return calib_directory_dates_num


def get_calibration_directory(calib_directory_dates_num: list, path: str, calib_directory: str, directories, folder_eu_time: dict = {}, Flag = False, idx = -1):
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
    directories : list
        the path of the directories containing the raw data
    Flag : boolean
        boolean indicating if the warnings and processed should be displayed (default: False)
    idx : int
        should always be -1, except for quality check
        
    Returns
    -------
    cal_fol_rt : path
        the path to the calibration directory with the date closest to the folder given as in input
    """
    
    print(folder_eu_time, path.split('temp_processing\\')[1])
    # get the date of the measurement of the folder that is currently being processed
    if path.split('temp_processing\\')[1] in folder_eu_time.keys():
        date_measurement = folder_eu_time[path.split('temp_processing\\')[1]]
    else:
        date_measurement = get_date_measurement(path, idx)
    
    print(path, date_measurement)
    
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


def find_closest_date(date_measurement: datetime, calib_dates_complete: list, directories: list, calib_directory: str, Flag = False):
    """
    finds the directory with the closest date to the measurement one

    Parameters
    ----------
    date_measurement : datetime
        the date
    calib_dates_complete : list of datetime
        a list of datetimes in which the closest one to date_measurement should be found
    directories : list
        the path of the directories containing the raw data
    calib_directory : str
        the path to the folder containing the calibration data
    Flag : boolean
        boolean indicating if the warnings and processed should be displayed (default: False)

    Returns
    -------
    date_calibration : str
        the name of the folder with the closest date to the measurement one
    """
    # create a deep copy of the calibration dates
    calib_dates = copy.deepcopy(calib_dates_complete)
    directories_calib = os.listdir(calib_directory)
    found = False

    # get the list of the wavelengths that needs to be checked
    wavelenghts_check = []
    for d in directories:
        wavelenghts_check.append(d.split('\\')[-1])

    while not found:
        date_calibration = None
        
        # compute the time difference between the calibration days and the measurement day 
        durations = []
        for calib in calib_dates:
            durations.append(abs((calib - date_measurement).days))
        
        # get the smallest indexs
        min_idx = np.where(np.array(durations) == np.array(durations).min())[0]
        min_idx_distance = min_idx[0]
        min_duration_distance = durations[min_idx_distance]
        closest_date = calib_dates[min_idx_distance].strftime('%Y-%m-%d')

        # get the calibration folders that were made on the date 'closest date'
        dates_calibrated = []
        for directory in directories_calib:
            if closest_date in directory:
                dates_calibrated.append(directory)

        found_calib = False

        while not found_calib and len(dates_calibrated) > 0:
            # get the last calibration made on the date 'closest_date' (e.g. returns 2022-11-01-C3 if three calibrations were performed on that day)
            idx_last_calib = get_last_calibration_idx(dates_calibrated)
            last_calibration = dates_calibrated[idx_last_calib]

            # check if there is calibration data for all the wavelenghts necessary
            all_found = True
            for wl in wavelenghts_check:
                path_B0 = os.path.join(calib_directory, last_calibration, wl, wl.split('nm')[0] + '_B0.cod')
                path_Bruit = os.path.join(calib_directory, last_calibration, wl, wl.split('nm')[0] + '_Bruit.cod')
                path_L30 = os.path.join(calib_directory, last_calibration, wl, wl.split('nm')[0] + '_L30.cod')
                path_P0 = os.path.join(calib_directory, last_calibration, wl, wl.split('nm')[0] + '_P0.cod')
                path_P90 = os.path.join(calib_directory, last_calibration, wl, wl.split('nm')[0] + '_P90.cod')
                if os.path.isfile(path_B0) and os.path.isfile(path_Bruit) and os.path.isfile(path_L30) and os.path.isfile(path_P0) and os.path.isfile(path_P90):
                    pass
                else:
                    all_found = False
                    
            # if yes, pass
            if all_found:
                found_calib = True
            # otherwise, go to the previous calibration (e.g. 2022-11-01-C2, 2022-11-01-C1...)
            else:
                del dates_calibrated[idx_last_calib]
        
        # if the calibration folder has been found, exit the while loop
        if found_calib:
            date_calibration = last_calibration
            found = True
        # else, remove this date from the list of possible dates
        else:
            min_idx = sorted(min_idx, reverse=True)
            for idx in min_idx:
                del calib_dates[idx]

    if date_calibration == None:
        raise FileNotFoundError('No calibration was found for the closest 1000 days. Check if calibration folders are present.')
    
    # if no calibration can be found for the same day, raise a warning
    if min_duration_distance > 0:
        incorrect_date(min_duration_distance, Flag = Flag)

    return date_calibration


def get_last_calibration_idx(dates_calibrated: list):
    """
    function returning the folder with the highest calibration index name (e.g. returns '2022-11-02-C3' if the input is 
    ['2022-11-02-C3', '2022-11-02-C2', '2022-11-02-C1'])

    Parameters
    ----------
    dates_calibrated : list
        the list of the folder names

    Returns
    -------
    calib_folder : str
        the folder with the highest calibration index name
    """
    calibration_idxs = []
    for date in dates_calibrated:
        calibration_idxs.append(int(date.split('_')[-1]))
    return np.argmax(calibration_idxs)


def get_date_measurement(path: str, idx = -1):
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


def incorrect_date(closest: int, Flag = False):
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
