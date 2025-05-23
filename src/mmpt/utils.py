import os, shutil
import copy
import json

import numpy as np
import math
from datetime import datetime
import warnings

from sklearn.preprocessing import MinMaxScaler

import cv2

import re

import matplotlib.colors as clr

from mmpt import libmpMuelMat

####################################################################################################################################
####################################################################################################################################
######################################### 1. Get the folders that will be processed ################################################
####################################################################################################################################
####################################################################################################################################

def get_measurements_to_process(parameters: dict, PDDN: bool = False, line_visualization: bool = False):
    """
    Returns the list of folders that need to be processed.
    
    Parameters
    ----------
    parameters : dict
        The parameters used for processing.
    PDDN : bool
        Whether denoising is used.

    Returns
    -------
    to_process : list
        List of folders to process.
    wl : list
        List of available wavelengths.
    """
    directories = parameters["input_dirs"]
    data_folders, _ = get_all_folders(directories)   
    
    if parameters['instrument'] == 'IMP':
        for folder in data_folders:
            move_folder_for_processing(folder)

    # Remove old processing logs if present
    for folder in data_folders:
        log_file = os.path.join(folder, "processing_logbook.txt")
        if os.path.exists(log_file):
            os.remove(log_file)

    to_process, processed, data_folder_nm, all_data_paths, wl = find_processed_folders(data_folders, parameters, PDDN, 
                                                                                       line_visualization)
    folder_data = create_folder_dict(parameters, data_folders, to_process, processed, data_folder_nm, all_data_paths, wl)
    
    if parameters['force_reprocess']:
        to_process = folder_data
    else:
        to_process = [f for f in folder_data if f["to_process"]]

    return to_process, wl  # Ensure uniqueness
    
    
def get_all_folders(directories: list):
    """
    Scans given directories and retrieves all valid data folders.

    Parameters
    ----------
    directories : list
        List of directories to scan.

    Returns
    -------
    data_folders : list
        List of valid folders containing data.
    folder_names : list
        List of measurement folder names.
    """
    data_folders, folder_names = [], []

    for directory in directories:
        for root, _, _ in os.walk(directory, topdown=False):
            if "TRANSMISSION" not in root:
                get_folder_name(root, data_folders, folder_names)

    return list(set(data_folders)), list(set(folder_names))


def get_folder_name(root: str, data_folders: list, folder_names: list):
    """
    Checks if a folder follows the YYYY-MM-DD format and adds it to the folder list,
    unless it contains 'C_X' where X is a number.

    Parameters
    ----------
    root : str
        Path to the folder.
    data_folders : list
        List of folders containing data.
    folder_names : list
        List of measurement folder names.
    """
    match = re.search(r"\d{4}-\d{2}-\d{2}", root)
    if match:
        match_index = root.find(match.group(0)) + len(match.group(0))
        remaining_path = root[match_index:]
        
        # Check if 'C_X' where X is a number exists in the folder name
        if re.search(r"C_\d+", os.path.basename(root)):
            return
        
        # Ensure there's no extra path separator after the date
        if not os.sep in remaining_path:
            data_folders.append(root)
            folder_names.append(os.path.basename(root))


def find_processed_folders(data_folders: list, parameters: dict, PDDN: bool, line_visualization: bool = False):
    """
    Identifies which folders contain raw data and determines if they have been processed.

    Parameters
    ----------
    data_folders : list
        List of folders to check.
    parameters : dict
        Processing parameters.
    PDDN : bool
        Whether denoising is used.

    Returns
    -------
    processed_dict : dict
        Dictionary mapping folder paths to their processed status.
    data_presence : dict
        Dictionary mapping folder paths to their raw data availability.
    wavelengths : list
        List of wavelengths to be processed.
    """
    to_process_dict, processed_dict, data_presence, all_data_paths = {}, {}, {}, {}
    wavelengths = get_wavelengths_processing(parameters["wavelengths"])
    polarimetry_path = {wl: "polarimetry_PDDN" if PDDN else "polarimetry" for wl in wavelengths}
    
    for path in data_folders:
        data, data_paths, processed, to_process = [], [], [], []

        for wl in wavelengths:
            if parameters["instrument"] == 'IMP':
                raw_data_path = os.path.join(path, "raw_data", wl) if os.path.exists(os.path.join(path, "raw_data", wl)) else os.path.join(path, "raw_data")
            else:
                raw_data_path = os.path.join(path, "to_process")
            
            data_pres, data_path, saving_paths = is_there_data(raw_data_path, wl, parameters["instrument"])

            
            if type(data_path) == str:
                data.append(data_pres)
                data_paths.append(data_path)
            else:
                for path_dat in data_path:
                    data.append(data_pres)
                    data_paths.append(path_dat)
            
            pol_path = os.path.join(path, polarimetry_path[wl])
            if parameters["instrument"] == 'IMP':
                process_condition(parameters, pol_path, path, wl, polarimetry_path, processed, to_process, saving_path = None,
                                  line_visualization = line_visualization)
            else:
                for path_dat, saving_path in zip(data_paths, saving_paths):
                    pol_folder_name = path_dat.split('.npy')[0].split(os.sep)[-1]
                    process_condition(parameters, os.path.join(pol_path, pol_folder_name), path_dat, 
                                      wl, polarimetry_path, processed, to_process, saving_path = saving_path,
                                      line_visualization = line_visualization)
                
        to_process_dict[path] = to_process
        processed_dict[path] = processed
        data_presence[path] = data
        all_data_paths[path] = data_paths

    return to_process_dict, processed_dict, data_presence, all_data_paths, wavelengths

def process_condition(parameters: dict, pol_path: str, path: str, wl: str, polarimetry_path: str, processed: list, 
                      to_process: list, saving_path = None, line_visualization: bool = False):
    condition_processed = os.path.exists(pol_path) and is_processed(path, wl, polarimetry_path[wl], parameters["workflow_mode"], 
                                    parameters["save_pdf_figs"], line_visualization = line_visualization) if parameters["instrument"] == 'IMP' else is_processed(pol_path, 
                                    wl, polarimetry_path[wl], parameters["workflow_mode"], parameters["save_pdf_figs"], 
                                    instrument = parameters["instrument"], line_visualization = line_visualization) 
    processed.append(condition_processed)
    if parameters["force_reprocess"]:
        to_process.append(True)
    else:
        to_process.append(not condition_processed)

def get_wavelengths_processing(wavelengths):
    """
    Returns a list of wavelengths to be processed.

    Parameters
    ----------
    wavelengths : str or list
        List of wavelengths or 'all'.

    Returns
    -------
    wavelengths_list : list
        List of wavelengths as strings.
    """
    if wavelengths == "all":
        return load_wavelengths()
    
    assert isinstance(wavelengths, list), "Wavelengths should be a list of int (or string)."
    return [f"{wl}nm" for wl in wavelengths]


def create_folder_dict(parameters, data_folders, to_process, processed, data_folder_nm, all_data_paths, wavelengths):
    """
    Creates a list of dictionaries storing folder processing information.

    Parameters
    ----------
    data_folders : list
        List of data folders.
    processed : dict
        Processed status per folder.
    data_folder_nm : dict
        Data availability per folder.
    wavelengths : list
        List of wavelengths.

    Returns
    -------
    list of dict
        List containing folder details.
    """
    folder_data = []
    for folder in data_folders:
        for idx, is_processed in enumerate(processed[folder]):
            if data_folder_nm[folder][idx]:  # Only include folders with data
                folder_data.append({
                    "folder_name": folder,
                    "to_process": to_process[folder][idx],
                    "processed": is_processed,
                    "wavelength": wavelengths[idx] if parameters['instrument'] == 'IMP' else wavelengths[0],
                    "path_intensite": all_data_paths[folder][idx]
                })
    return folder_data

def is_there_data(path: str, wl: str, instrument: str) -> bool:
    """
    """
    if instrument == 'IMP':
        return is_there_data_IMP(path, wl)
    else:
        return is_there_data_IMPv2(path, wl)

def is_there_data_IMPv2(path: str, wl: str):
    # Regular expression pattern
    pattern = re.compile(r"^630_Image_Number_\d+\.npy$")
    intensity_files = [os.path.join(path, f) for f in os.listdir(path) if pattern.match(f)]
    saving_paths = [os.path.join(path, 'polarimetry', f).replace('/to_process', '').replace('.npy', '') for f in os.listdir(path) if pattern.match(f)]
    if os.path.exists(os.path.join(path, f"A.npy")) and os.path.exists(os.path.join(path, f"W.npy")):
        pass
    else:
        print(" [wrn] Missing calibration files in folder: ", path)
    return len(intensity_files) > 0, intensity_files, saving_paths
    
def is_there_data_IMP(path: str, wl: str):
    """
    Checks if raw data is available in the given folder.

    Returns True if the expected `.cod` file is present.
    """
    if not os.path.isdir(path):
        return False
    expected_file = f"{wl.replace('nm', '')}_Intensite.cod"
    return expected_file in set(os.listdir(path)), os.path.join(path, expected_file), None


def is_processed(path: str, wl: str, polarimetry_fname: str, processing_mode: str = "full", save_pdf_figs: bool = False, 
                 instrument: str = 'IMP', line_visualization: bool = False) -> bool:
    """
    Checks if the required files (e.g., polarimetric plots) were generated.

    Returns True if all required files are present.
    """
    if instrument == 'IMP':
        folder_path = os.path.join(path, polarimetry_fname, wl)
    else:
        folder_path = os.path.join(path, wl)

    if not line_visualization:
        expected_filenames = load_filenames(processing_mode, save_pdf_figs)
    else:
        expected_filenames = load_filenames(processing_mode = 'results', save_pdf_figs = save_pdf_figs)
        folder_path = os.path.join(folder_path, 'results')
    if not os.path.exists(folder_path):
        return False
    
    return all(f in set(os.listdir(folder_path)) for f in expected_filenames)

def move_folder_for_processing(folder: str):
    """
    Move a measurement folder to the temp_processing folder.
    
    Parameters
    ----------
    parameters : dict
        The processing parameters.
    folder : dict
        The folder to be processed.
        
    Returns
    -------
    links_folders : dict
        Mapping of original folder to temp_processing folder.
    to_process_temp : list
        List containing the processed folder in the temp_processing directory.
    """
    # Ensure 'raw_data' directory exists
    raw_data_path = os.path.join(folder, 'raw_data')
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if file != 'raw_data':
                shutil.move(file_path, os.path.join(raw_data_path, file))

    # Ensure 'annotation' directory exists
    annotation_path = os.path.join(folder, 'annotation')
    os.makedirs(annotation_path, exist_ok=True)
    
    # Ensure 'histology' directory exists
    histology_path = os.path.join(folder, 'histology')
    os.makedirs(histology_path, exist_ok=True)


####################################################################################################################################
####################################################################################################################################
################################################### 2. Managing folders ############################################################
####################################################################################################################################
####################################################################################################################################

def reorganize_folders(folder_path: str, instrument: str):
    """
    Reorganize the folders in the measurement directory by creating necessary subdirectories
    for polarimetry data and moving raw data into appropriate folders.

    Parameters
    ----------
    folder_path : str
        The path to the measurement directory.
        
    Returns
    -------
    None
    """
    directories_tbc = ['polarimetry', 'polarimetry_PDDN']
    wavelengths = load_wavelengths(instrument)

    # Create necessary directories for polarimetry and polarimetry_PDDN data
    for directory in directories_tbc:
        create_directory_structure(folder_path, directory, wavelengths, instrument)
    
    # Clean up old computations (remove unnecessary files)
    clean_up_old_files(folder_path['folder_name'], directories_tbc, instrument)


def create_directory_structure(folder_path: str, directory: str, wavelengths: list, instrument: str):
    """
    Creates the necessary directory structure for a given directory (polarimetry or polarimetry_PDDN)
    for each wavelength.

    Parameters
    ----------
    folder_path : str
        The path to the folder where the subdirectories will be created.
    directory : str
        The directory name (polarimetry or polarimetry_PDDN).
    wavelengths : list
        List of wavelengths for which directories need to be created.
        
    Returns
    -------
    None
    """    
    target_path = os.path.join(folder_path['folder_name'], directory)
    os.makedirs(target_path, exist_ok=True)
    
    if instrument == 'IMPv2':
        target_path = os.path.join(target_path, folder_path['path_intensite'].split(os.sep)[-1].replace('.npy', '').replace('PDDN_', ''))
        os.makedirs(target_path, exist_ok=True)
        
    for wl in wavelengths:
        os.makedirs(os.path.join(target_path, wl), exist_ok=True)


def clean_up_old_files(folder_path: str, dirs_to_check: list, instrument: str):
    """
    Cleans up old files in the specified directories by removing files that are not in the allowed list.
    
    Parameters
    ----------
    folder_path : str
        The path to the measurement directory.
    dirs_to_check : list of str
        The directories to check for old files (e.g., polarimetry, polarimetry_PDDN).
        
    Returns
    -------
    None
    """
    if instrument == 'IMPv2':
        print(' [wrn] clean_up_old_files not implemented yet for IMPv2.')
        return None
    
    filenames = load_filenames(save_pdf_figs=True)
    filenames_results = load_filenames(processing_mode='results', save_pdf_figs=True)
    
    for dir in dirs_to_check:
        polarimetry_dir = os.path.join(folder_path, dir)
        
        for wl in os.listdir(polarimetry_dir):
            folder = os.path.join(polarimetry_dir, wl)
            
            if wl.startswith('.DS'):
                os.remove(folder)
                continue
                
            # Process the files within each wavelength folder
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                    
                if file in filenames:
                    continue
                    
                if file == 'results':
                    clean_up_results(folder, file, filenames_results)
                    continue
                    
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)


def clean_up_results(folder: str, results_folder: str, filenames_results: list):
    """
    Cleans up result files that are not in the allowed list.
    
    Parameters
    ----------
    folder : str
        The folder containing the results folder.
    results_folder : str
        The name of the results folder.
    filenames_results : list
        List of allowed result filenames.
        
    Returns
    -------
    None
    """
    results_path = os.path.join(folder, results_folder)
    
    for result_file in os.listdir(results_path):
        result_file_path = os.path.join(results_path, result_file)
        
        if result_file not in filenames_results:
            os.remove(result_file_path)


####################################################################################################################################
####################################################################################################################################
########################################### 3. Manage the calibrations folders #####################################################
####################################################################################################################################
####################################################################################################################################


def get_calibration_dates(parameters: dict):
    """
    Returns the dates of all calibration folders containing calibration data for 550nm and 650nm.

    Parameters
    ----------
    calib_directory : str
        Path to the calibration directory.

    Returns
    -------
    list of datetime
        A list containing the dates of the calibration folders.
    """
    calib_directory = parameters['calib_dir']
    
    calib_directory_dates = os.listdir(calib_directory)
    wavelengths = parameters['wavelengths']
     
    calib_directory_dates_cleaned = [
        c for c in calib_directory_dates
        if any(os.path.exists(os.path.join(calib_directory, c, f"{str(wl)}nm", f"{str(wl)}_W.mat"))
               or os.path.exists(os.path.join(calib_directory, c, f"{str(wl)}nm", f"{str(wl)}_W.cod" if parameters['instrument'] == 'IMP' else f"W.npy"))
               or os.path.exists(os.path.join(calib_directory, c, f"{str(wl)}nm", f"{str(wl)}_B0.cod" if parameters['instrument'] == 'IMP' else f"W.npy"))
               for wl in wavelengths)
    ]

    # Convert folder names to datetime objects
    return [datetime.strptime(c.split('_')[0], '%Y-%m-%d') for c in calib_directory_dates_cleaned]



def get_calibration_directory(parameters:dict, calib_directory_dates_num: list, path: str,
                              wavelength: str, folder_eu_time: dict = {}, Flag=False, idx=-1):
    """
    Gets the calibration directory with the date closest to the folder given as input.

    Parameters
    ----------
    calib_directory_dates_num : list
        List of datetime objects for calibration dates.
    path : str
        The path to the folder being processed.
    calib_directory : str
        Path to the folder containing calibration data.
    wavelength : str
        The wavelength to check.
    folder_eu_time : dict, optional
        Dictionary mapping folder names to EU time. Default is {}.
    Flag : bool, optional
        Flag indicating if warnings should be displayed. Default is False.
    idx : int, optional
        Index used for splitting path. Default is -1.

    Returns
    -------
    str
        Path to the calibration directory with the closest date to the folder.
    """
    calib_directory = parameters['calib_dir']
    
    # Get the date of the measurement folder
    date_measurement = folder_eu_time.get(path.split(os.sep)[-1], get_date_measurement(path, idx))

    # Find the closest date in the calibration directory
    closest_date = find_closest_date(parameters, date_measurement, calib_directory_dates_num, wavelength, calib_directory, Flag)
    
    return os.path.join(calib_directory, closest_date)


def find_closest_date(parameters: dict, date_measurement: datetime, calib_dates_complete: list, wavelength: str, calib_directory: str, Flag=False):
    """
    Finds the directory with the closest date to the measurement date.

    Parameters
    ----------
    date_measurement : datetime
        The date of the measurement.
    calib_dates_complete : list of datetime
        List of calibration dates.
    wavelength : str
        The wavelength to check.
    calib_directory : str
        Path to the calibration directory.
    Flag : bool, optional
        Flag indicating if warnings should be displayed. Default is False.

    Returns
    -------
    str
        The folder name with the closest calibration date.
    """
    calib_dates = copy.deepcopy(calib_dates_complete)
    directories_calib = os.listdir(calib_directory)

    # Calculate time differences between measurement and calibration dates
    durations = [abs((calib - date_measurement).days) for calib in calib_dates]
    
    # Find the index of the closest date
    min_idx_distance = np.argmin(durations)
    closest_date = calib_dates[min_idx_distance].strftime('%Y-%m-%d')

    # Find calibration directories for the closest date
    dates_calibrated = [d for d in directories_calib if closest_date in d]

    # Check for valid calibration data
    while dates_calibrated:
        idx_last_calib = get_last_calibration_idx(dates_calibrated)
        last_calibration = dates_calibrated[idx_last_calib]

        # Check if all required calibration files exist
        if all_calibration_files_exist(parameters, last_calibration, calib_directory, wavelength):
            return last_calibration
        else:
            # Remove invalid calibration and check the previous one
            del dates_calibrated[idx_last_calib]

    raise FileNotFoundError('No valid calibration found for the closest 1000 days.')

def all_calibration_files_exist(parameters: dict, calibration_folder: str, calib_directory: str, wavelength: str) -> bool:
    """Checks if all required calibration files exist for a given calibration folder and wavelength."""
    required_files = [f"{wavelength.split('nm')[0]}_A.cod", f"{wavelength.split('nm')[0]}_W.cod"] if parameters['instrument'] == 'IMP' else [f"A.npy", f"W.npy"]
    if all(os.path.isfile(os.path.join(calib_directory, calibration_folder, wavelength, f)) for f in required_files):
        return True
    else:
        required_files = [f"{wavelength.split('nm')[0]}_B0.cod", f"{wavelength.split('nm')[0]}_Bruit.cod",
                        f"{wavelength.split('nm')[0]}_L30.cod", f"{wavelength.split('nm')[0]}_P0.cod",
                        f"{wavelength.split('nm')[0]}_P90.cod"]
        return all(os.path.isfile(os.path.join(calib_directory, calibration_folder, wavelength, f)) for f in required_files)

def get_last_calibration_idx(dates_calibrated: list) -> int:
    """Returns the index of the last calibration folder based on the highest index."""
    calibration_idxs = [int(date.split('_')[-1]) for date in dates_calibrated]
    return np.argmax(calibration_idxs)


def get_date_measurement(path: str, idx=-1) -> datetime:
    """Returns the date of the measured folder."""
    return datetime.strptime(path.split(os.sep)[idx].split('_')[0], '%Y-%m-%d')


def incorrect_date(closest: int):
    """Raises a warning indicating that no calibration was found for the exact date of the measurement."""
    warnings.warn(f'No calibration found for the exact date; the one used was {closest} day(s) ago.', UserWarning, stacklevel=2)
    
    
####################################################################################################################################
####################################################################################################################################
############################################## 4. Mueller Matrix processing ########################################################
####################################################################################################################################
####################################################################################################################################

def get_intensity_old(path: str, wavelength: int, align_wls=False, PDDN=False):
    """
    Retrieves the intensity data from a specified directory based on the wavelength and alignment settings.

    Parameters
    ----------
    path : str
        The path to the directory containing the raw data.
    wavelength : int
        The wavelength to process.
    align_wls : bool, optional
        Whether to align wavelengths (default: False).
    PDDN : bool, optional
        Whether to apply PDDN processing (default: False).

    Returns
    -------
    tuple
        A tuple containing the intensity data and the polarimetry file name.
    """
    pathCod, polarimetry_fname = get_pathCod(path, wavelength, align_wls, PDDN)

    try:
        I = libmpMuelMat.read_cod_data_X3D(pathCod, isRawFlag=0)
    except Exception:
        I = libmpMuelMat.read_cod_data_X3D(pathCod, isRawFlag=1)

    return I, polarimetry_fname

def get_intensity(parameters: dict, path: str):
    """
    Retrieves the intensity data from a specified directory based on the wavelength and alignment settings.

    Parameters
    ----------
    path : str
        The path to the directory containing the raw data.
    wavelength : int
        The wavelength to process.
    align_wls : bool, optional
        Whether to align wavelengths (default: False).
    PDDN : bool, optional
        Whether to apply PDDN processing (default: False).

    Returns
    -------
    tuple
        A tuple containing the intensity data and the polarimetry file name.
    """
    if parameters['instrument'] == 'IMP':

        try:
            I = libmpMuelMat.read_cod_data_X3D(path, isRawFlag=0)
        except Exception:
            I = libmpMuelMat.read_cod_data_X3D(path, isRawFlag=1)
    
    else:
        I = np.load(path, allow_pickle=True)
        I = I.astype(np.double)

    return I


def get_pathCod(directory: str, wavelength: int, align_wls=False, PDDN=False):
    """
    Constructs the path to the .cod file based on wavelength, alignment, and PDDN settings.

    Parameters
    ----------
    directory : str
        The directory containing the raw data.
    wavelength : int
        The wavelength to process.
    align_wls : bool, optional
        Whether to align wavelengths (default: False).
    PDDN : bool, optional
        Whether to apply PDDN processing (default: False).

    Returns
    -------
    tuple
        A tuple containing the constructed path to the .cod file and the corresponding polarimetry file name.
    """
    # Set the file name based on the wavelength and alignment settings
    if wavelength == '550nm':
        align_wls = False
    filename = f"{wavelength}".replace('nm', '') + "_Intensite_aligned.cod" if align_wls else f"{wavelength}".replace('nm', '') + "_Intensite.cod"
    pathCod = os.path.join(directory, "raw_data", f"{wavelength}", filename)

    # Handle PDDN processing if required
    if PDDN:
        pathCod_pddn = pathCod.replace('_Intensite', '_Intensite_PDDN')
        if os.path.exists(pathCod_pddn):
            pathCod = pathCod_pddn
            polarimetry_fname = 'polarimetry_PDDN'
        elif wavelength in ['550nm', '600nm']:
            raise FileNotFoundError(f'No PDDN file found for folder {directory}, for the wavelength {wavelength}.')
        else:
            polarimetry_fname = 'polarimetry'
    else:
        polarimetry_fname = 'polarimetry'

    return pathCod, polarimetry_fname

def normalize_M11(M11: np.ndarray, instrument: str) -> np.ndarray:
    if os.path.exists(os.path.join(get_data_folder_path(), f'gaussian_fit_{instrument}.npy')):
        gaussian_fit = np.load(os.path.join(get_data_folder_path(), f'gaussian_fit_{instrument}.npy'))
        M11_norm = M11 / gaussian_fit
        M11_norm[M11_norm > 65535] = 65535
        return M11_norm
    else:
        print(' [wrn] No Gaussian fit found for normalization. Returning unnormalized M11.')
        return M11
    
def correct_M11(M11: np.ndarray, instrument: str) -> np.ndarray:
    if instrument == 'IMPv2':
        M11 = np.nan_to_num(M11, nan=0)
        M11[M11 > 255] = 255
        M11[M11 < 0] = 0
        
    return M11

def curate_azimuth(azimuth: np.ndarray, folder_name=None) -> np.ndarray:
    """
    Curates the azimuth array by filling NaN values with the mean of neighboring pixels.

    Parameters
    ----------
    azimuth : np.ndarray
        Azimuth array of shape (388, 516)
    folder : str, optional
        The current processed folder (default: None)
        
    Returns
    -------
    np.ndarray
        The curated azimuth array
    """
    if libmpMuelMat._isNumStable(azimuth):
        return azimuth, np.zeros_like(azimuth)

    counter = 0
    
    unstable_azimuth = np.isnan(azimuth)

    for idx in range(azimuth.shape[0]):
        for idy in range(azimuth.shape[1]):
            if math.isnan(azimuth[idx, idy]):
                # Obtain the neighboring pixels and compute the mean
                azi_neighbors = select_region(azimuth.shape, azimuth, idx, idy)
                
                if np.any(~np.isnan(azi_neighbors)):
                    mean_azi = np.nanmean(azi_neighbors)
                    azimuth[idx, idy] = mean_azi
                    counter += 1
                else:
                    azimuth[idx, idy] = 0

    # Optional: debug check for stability after modification
    if not libmpMuelMat._isNumStable(azimuth) and folder_name:
        print(f"Azimuth not stable in folder: {folder_name}")

    # Optionally, handle cases where too many pixels were modified
    if counter > 1 / 100 * azimuth.size:
        print(f"More than 1% of the pixels were modified in folder: {folder_name}.")

    return azimuth, unstable_azimuth


def select_region(shape: tuple, azimuth: np.ndarray, idx: int, idy: int) -> np.ndarray:
    """
    Select the region around a pixel of interest in the azimuth array for curation.

    Parameters
    ----------
    shape : tuple
        The shape of the array
    azimuth : np.ndarray
        Azimuth array of shape (388, 516)
    idx, idy : int
        The index values of the pixel of interest
        
    Returns
    -------
    np.ndarray
        The neighboring pixels of the azimuth array
    """
    # Define the neighborhood size (a 3x3 window)
    min_x = max(0, idx - 1)
    max_x = min(shape[0], idx + 2)
    min_y = max(0, idy - 1)
    max_y = min(shape[1], idy + 2)

    return azimuth[min_x:max_x, min_y:max_y]

def process_mm(I, remove_reflection: bool, A, W, lu_chipman_backend, lu_chipman_model):
    """Processes the Mueller matrix, removing reflections if necessary."""
    times_processing = {}
    
    I = np.nan_to_num(I, nan=0.0).astype(np.float64)
    
    if remove_reflection:
        start = time.time()
        I_ref, dilated_mask = libmpMuelMat.removeReflections3D(I)
        times_processing['remove_reflection'] = time.time() - start
        
        processed, times = libmpMuelMat.process_MM_pipeline(A, I_ref, W, dilated_mask, lu_chipman_backend = lu_chipman_backend)
    else:
        start = time.time()
        dilated_mask = libmpMuelMat.maskReflections3D(I)
        times_processing['remove_reflection'] = time.time() - start
        
        processed, times = libmpMuelMat.process_MM_pipeline(A, I, W, I, lu_chipman_backend = lu_chipman_backend)
        
    for key, value in times.items():
        times_processing[key] = value
        
    start = time.time()
    if lu_chipman_backend == 'prediction':
        MM = predict_lu_chipman(processed[0], lu_chipman_model)
        MM['nM'] = processed[0]
        MM['M11'] = processed[1]
        processed = MM
        times_processing['lu_chipman'] = time.time() - start
    
    return processed, dilated_mask, times_processing
        
        
def load_calibration_data(parameters: dict, measurement: dict, 
                          calibration_directory_wl: str, wavelength: str):
    if parameters['instrument'] == 'IMP':
        return load_calibration_data_IMP(calibration_directory_wl, wavelength)
    else:
        return load_calibration_data_IMPv2(measurement)
    
def load_calibration_data_IMP(calibration_directory_wl: str, wavelength: str):
    """Loads the calibration data (A and W) from the corresponding files."""
    files = os.listdir(calibration_directory_wl)
    wavelength_number = wavelength.replace('nm', '')

    calib_files = [f"{wavelength_number}_A.cod", f"{wavelength_number}_W.cod"] 
    
    if all(f in files for f in calib_files):
        A = libmpMuelMat.read_cod_data_X3D(os.path.join(calibration_directory_wl, f"{wavelength_number}_A.cod"), isRawFlag=0)
        W = libmpMuelMat.read_cod_data_X3D(os.path.join(calibration_directory_wl, f"{wavelength_number}_W.cod"), isRawFlag=0)
    else:
        A, W, _, _, _ = libmpMuelMat.calib_System_AW(calibration_directory_wl, wlen=int(wavelength_number))
    return A, W

def load_calibration_data_IMPv2(measurement: dict):
    path_folder = os.path.join(measurement['folder_name'], 'to_process')
    A = np.load(os.path.join(path_folder, 'A.npy'))
    W = np.load(os.path.join(path_folder, 'W.npy'))
    return A, W
    
def get_angle_correction(path: str) -> int:
    """Gets the angle correction from a file if available."""
    primary_path = os.path.join(path, 'rotation_MM.txt')
    secondary_path = os.path.join(path, 'annotation', 'rotation_MM.txt')

    for file_path in [primary_path, secondary_path]:
        if os.path.exists(file_path):
            try:
                with open(file_path) as f:
                    return int(f.readline().strip())
            except (FileNotFoundError, ValueError):
                return 0  # Return 0 if file not found or contains invalid integer
    return 0  # Return 0 if neither file exists

def save_file_as_npz(variable: dict, path: str):   
    """
    save_file_as_npz allows to store a Mueller Matrix as a numpy zipped file

    Parameters
    ----------
    variable : dict
        the MM to save (should contain ['Intensity', 'M11', 'Msk', 'totD', 'linR', 'azimuth', 'totP'] as keys)
    path : str
        the path in which the MM should be saved
    """
    np.savez(path, **variable)
    
def save_file_as_npy(variable: dict, path: str):
    """
    save_file_as_npy allows to store a Mueller Matrix as a numpy file

    Parameters
    ----------
    variable : dict
        the MM to save (should contain ['Intensity', 'M11', 'Msk', 'totD', 'linR', 'azimuth', 'totP'] as keys)
    path : str
        the path in which the MM should be saved
    """
    np.save(path, variable)
                  

def load_npz_file(path: str) -> dict:
    """
    load_npz_file loads a Mueller Matrix stored as a numpy zipped file.

    Parameters
    ----------
    path : str
        The path of the MM file to load.

    Returns
    -------
    dict
        A dictionary containing the loaded MM data.
    """
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data}
    
####################################################################################################################################
####################################################################################################################################
################################################## 5. Visualization lines ##########################################################
####################################################################################################################################
####################################################################################################################################


def get_cmap(parameter: str, instrument: str = 'IMP', mm_processing: str = 'torch'):
    """
    master function to get the cmap for plot of a polarimetric parameter

    Parameter
    -------
    parameter: str
        the name of the parameter (i.e. 'azimuth', 'depolarization'...)

    Returns
    -------
    cmap : colormap
        the cmap for the azimuth plot
    norm : colormap
        the nromalized cmap for the azimuth plot
    """
    # load the parameters used to generate the colormap
    parameters_plot = load_plot_parameters(instrument, mm_processing)[parameter]
    colors = parameters_plot['colors']
    n_bins = parameters_plot['n_bins']
    cmap_name = parameters_plot['cmap_name']
    vmin, vmax = parameters_plot['cbar']

    # create and normalize the colormap
    cmap = clr.LinearSegmentedColormap.from_list(cmap_name, colors, n_bins)
    norm = clr.Normalize(vmin, vmax)
    
    return cmap, norm


####################################################################################################################################
####################################################################################################################################
################################################ X. Preprocessing torch MM #########################################################
####################################################################################################################################
####################################################################################################################################

import time
import torch

def preprocess_intensities(parameters, mm_model, times, sample = None, predict = False, amat = None, wmat = None, frame = None, 
                           lu_chipman_backend = 'prediction', lu_chipman_model = None, path_intensite = None):
    start_total = time.time()

    processing_mode = 'prediction' if frame is None else 'processing'
    
    if predict:
        if path_intensite is None or '.cod' in path_intensite:
            intensity = libmpMuelMat.read_cod_data_X3D(os.path.join(sample, 'raw_data/550_Intensite.cod'), isRawFlag=1)
            bruit = libmpMuelMat.read_cod_data_X3D(os.path.join(sample, 'raw_data/550_Bruit.cod'), isRawFlag=1)
            frame = intensity - bruit
        else:
            frame = np.load(path_intensite, allow_pickle=True)[::2,::2]
    else:
        assert frame is not None, "frame should be provided if predict is False"
    
    if processing_mode == 'prediction':
        times['load_cod'].append(time.time() - start_total)
    
    start = time.time()        
    if predict:
        if path_intensite is None or '.cod' in path_intensite:
            path_calib_folder = get_calib_folder(parameters, sample)
            wl = 550
            amat = libmpMuelMat.read_cod_data_X3D(os.path.join(path_calib_folder, str(wl) + 'nm', str(wl) + '_A.cod'))
            wmat = libmpMuelMat.read_cod_data_X3D(os.path.join(path_calib_folder, str(wl) + 'nm', str(wl) + '_W.cod'))
        else:
            amat = np.load(os.path.join((os.sep).join(path_intensite.split(os.sep)[:-1]), 'A.npy'))[::2,::2]
            wmat = np.load(os.path.join((os.sep).join(path_intensite.split(os.sep)[:-1]), 'W.npy'))[::2,::2]
    else:
        assert amat is not None and wmat is not None, "amat and wmat should be provided if predict is False"
        
    frame = np.concatenate([frame, amat, wmat], axis=-1)
    if processing_mode == 'prediction':
        times['load_calib'].append(time.time() - start)
    
    start = time.time()
    tensor = torch.from_numpy(frame.astype(np.float32))
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    if processing_mode == 'prediction':
        times['switch_cuda'].append(time.time() - start)
    
    start = time.time()
    if processing_mode == 'prediction':
        try:
            input = mm_model(tensor, path = os.path.join(sample, 'polarimetry/550nm/nM.npy'))
        except:
            input = mm_model(tensor)
        input = input[:, :-1].to('cuda')
        times['get_tensor'].append(time.time() - start)
        
        times['total'].append(time.time() - start_total)
        
        return input, times
    
    else:
        time_load_data_GPU = time.time() - start_total
        start_processing = time.time()
        input, times = mm_model(tensor, processMM = True, lu_chipman_backend = lu_chipman_backend)
        
        start_lu_chipman = time.time()
        if lu_chipman_backend == 'prediction':
            MM = predict_lu_chipman(input[0], lu_chipman_model)
            MM['nM'] = input[0]
            MM['M11'] = input[1]
            input = MM
            times['lu_chipman'] = time.time() - start_lu_chipman
            
        times['total'] = time.time() - start_processing
        
        return input, time_load_data_GPU, times

import torch
import torch.nn as nn
import torch.nn.functional as F
    
class LuChipmanPred(nn.Module):
    def __init__(self):
        super(LuChipmanPred, self).__init__()
        self.fc1 = nn.Linear(16, 128)  # Corrected input dim
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)
            
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Softmax can be added outside if needed
        return x


def predict_lu_chipman(nM, lu_chipman_model):
    
    MM = {}        
    input = nM.reshape(nM.shape[0] * nM.shape[1], 16)
    input = torch.tensor(input, dtype=torch.float32).to('cuda')
    # predictions = lu_chipman_model.predict(input, batch_size = input.shape[0])
    predictions = lu_chipman_model(input).cpu().detach().numpy()

    tissue_dims = (nM.shape[0],  nM.shape[1])
        
    MM['totD'] = predictions[:, 0].reshape(tissue_dims)
    MM['totP'] = predictions[:, 1].reshape(tissue_dims)
    MM['linR'] = np.degrees(np.arctan2(predictions[:, 3], predictions[:, 2]).reshape(tissue_dims))
    MM['azimuth'] = np.degrees(np.arctan2(predictions[:, 5], predictions[:, 4]).reshape(tissue_dims)) % 180
    
    return MM


def get_calib_folder(parameters, sample):
    if os.path.exists(os.path.join(sample, 'calib_folder.txt')):
        with open(os.path.join(sample, 'calib_folder.txt'), 'r') as f:
            calib_folder = f.read().strip()
        return os.path.join(parameters['calib_dir'], calib_folder.replace('./calib/', ''))
    else:
        raise NotImplementedError("The calibration folder is not available in the sample.")

from numpy.lib.stride_tricks import as_strided

def bin_pixels(array, resize_factor=4):
    """
    Fast pixel binning using NumPy's stride tricks.
    Assumes the input array has shape (H, W, C) where H and W are divisible by resize_factor.
    """
    H, W, C = array.shape
    H_new, W_new = H // resize_factor, W // resize_factor

    # Create a strided view
    strided = as_strided(
        array,
        shape=(H_new, resize_factor, W_new, resize_factor, C),
        strides=(
            array.strides[0] * resize_factor, array.strides[0], 
            array.strides[1] * resize_factor, array.strides[1], 
            array.strides[2]
        ),
        writeable=False  # Prevent modification of original data
    )

    # Compute the mean efficiently
    return strided.mean(axis=(1, 3))

def load_model(mm_model, cfg):

    n_channels = mm_model.ochs if cfg.data_subfolder.__contains__('raw') else len(cfg.feature_keys)
    if cfg.model == 'unet':
        from mmpt.addons.polarpred.segment_models.unet import UNet
        model = UNet(n_channels=n_channels, n_classes=cfg.class_num+cfg.bg_opt, shallow=cfg.shallow)
    else:
        raise Exception('Model %s not recognized' % cfg.model)

    model = model.to(memory_format=torch.channels_last)
    model.to(device=cfg.device)
    model.eval()
    
    if cfg.MM:
        path_models = os.path.join(getPredModelsPath(), 'MM')
    else:
        path_models = os.path.join(getPredModelsPath(), 'LuChipman')
        
    model_path = os.path.join(path_models, os.listdir(path_models)[0]) 
    
    state_dict = torch.load(os.path.join(getPolarPredPath(), model_path), map_location=cfg.device)
    model.load_state_dict(state_dict) if cfg.model != 'resnet' else model.model.load_state_dict(state_dict)    
    return model, model_path

from mmpt.addons.polarpred.multi_loss import reduce_htgm

def predict(model, input):
    preds = model(input)
    preds = reduce_htgm(preds)
    return preds

####################################################################################################################################
####################################################################################################################################
################################################ X. File loader functions ##########################################################
####################################################################################################################################
####################################################################################################################################


def load_MM(path: str):
    """
    load the mueller matric given into path

    Parameters
    ----------
    path : str
        the path to the mueller matrix to be loaded
        
    Returns
    -------
    orientation : dict
        the MM data
    """
    mat = np.load(path)
    return mat


def load_wavelengths(instrument: str = 'IMP'):
    """
    load and returns the wavelengths usable by the IMP

    Returns
    -------
    wavelenghts : list
        the wavelengths usable by the IMP
    """
    wavelenghts = {'IMP': ['450nm', '500nm', '550nm', '600nm', '650nm', '700nm'], 'IMPv2': ['630nm']}
    return wavelenghts[instrument]

def load_filenames(processing_mode = 'full', save_pdf_figs = False):
    """
    load and returns the name of the files that will be generated during the processing

    Returns
    -------
    filenames : list
        the list of the files generated during the processing
    """
    path_filenames = os.path.join(get_data_folder_path(), 'filenames.json')
    
    with open(path_filenames) as f:
        fnames = json.load(f)
    fnames = fnames[processing_mode]
    
    for name in fnames:
        if '.pdf' in name and not save_pdf_figs:
            fnames.remove(name)
    return fnames

def load_plot_parameters(instrument: str = 'IMP',  mm_processing: str = 'torch'):
    """
    load and returns the parameters for the polarimetric parameter plots

    Returns
    -------
    plot_parameters : dict
        the parameters to plot the parameters maps
    """
    with open(os.path.join(get_data_folder_path(), 'parameters_plot.json')) as json_file:
        data = json.load(json_file)
    return data[mm_processing][instrument]


def load_parameters_visualization():
    """
    load and returns the wavelengths usable by the IMP

    Returns
    -------
    data : dict
        the parameters used for the parameters of the visualization
    """
    with open(os.path.join(get_data_folder_path(), 'parameters_visualizations.json')) as json_file:
        data = json.load(json_file)
    return data

def load_combined_plot_name(viz: bool = False):
    """
    load and returns the name of the combined plot file

    Parameters
    -------
    viz: bool
        indicates if we are working with the bar visualization

    Returns
    -------
    filenames : list
        the list of the files generated during the processing
    """
    with open(os.path.join(get_data_folder_path(), 'combined_figure.json')) as f:
        data = json.load(f)
    if viz:
        return data['with_viz']
    else:
        return data['without_viz']

def load_parameter_names(processing_mode):
    """
    load and returns the name of the files that will be used to create the combined plot

    Parameters
    -------
    viz: bool
        indicates if we are working with the bar visualization
    
    Returns
    -------
    filenames : list
        the list of the files that will be used to create the combined plot
    """
    with open(os.path.join(get_data_folder_path(), 'parameter_names.json')) as json_file:
        data = json.load(json_file)
    return data[processing_mode]

def getCongfigPath():
    import mmpt
    module_path = os.path.dirname(mmpt.__file__)
    return os.path.join(module_path, 'config')

def get_data_folder_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
 
def getSupergluePath():
    import mmpt
    module_path = os.path.dirname(mmpt.__file__)
    return os.path.join(module_path, '..', '..', 'third_party', 'superglue')

def getPredictionPath():
    import mmpt
    module_path = os.path.dirname(mmpt.__file__)
    return os.path.join(module_path, '..', '..', 'third_party', 'polarfeat_mar25')

def getPolarPredPath():
    import mmpt
    module_path = os.path.dirname(mmpt.__file__)
    return os.path.join(module_path, 'addons', 'polarpred')

def getLuChipmanPredPath():
    import mmpt
    module_path = os.path.dirname(mmpt.__file__)
    return os.path.join(module_path, 'addons', 'luchipmanpred')

def getPredModelsPath():
    import mmpt
    module_path = os.path.dirname(mmpt.__file__)
    return os.path.join(module_path, 'addons', 'polarpred', 'ckpts')

def getTestPath():
    import mmpt
    module_path = os.path.dirname(mmpt.__file__)
    return os.path.join(module_path, '..', '..', 'tests')

def test_pddn_models_existence(PDDN_models_path, instrument = 'IMP'):
    """Check if required PDDN models exist."""
    models = {'IMP': ['PDDN_model_550_Fresh_HB.pt', 'PDDN_model_600_Fresh_HB.pt'],
              'IMPv2': ['PDDN_model_630_Fresh_HB.pt']}
    models = models[instrument]
    
    missing_models = []
    for model in models:
        if not os.path.exists(os.path.join(PDDN_models_path, model)):
            missing_models.append(model)
            
    return len(missing_models) == 0, missing_models

def remove_previous_temp_folders(prefix: str, base_path: str):
    """Check for all directories in the specified base_path and remove those with the given prefix."""
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path) and folder.startswith(prefix):
            shutil.rmtree(folder_path)
            print(f"Removed existing folder: {folder_path}")