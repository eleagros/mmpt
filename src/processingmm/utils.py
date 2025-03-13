import os, shutil
import copy
import json

import numpy as np
import math
from datetime import datetime
import warnings

import re

import matplotlib.colors as clr

from processingmm import libmpMuelMat

####################################################################################################################################
####################################################################################################################################
######################################### 1. Get the folders that will be processed ################################################
####################################################################################################################################
####################################################################################################################################

def get_measurements_to_process(parameters: dict, PDDN: bool = False):
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
    directories = parameters["directories"]
    data_folders, _ = get_all_folders(directories)
    
    for folder in data_folders:
        move_folder_for_processing(parameters, folder)
        
    # Remove old processing logs if present
    for folder in data_folders:
        log_file = os.path.join(folder, "processing_logbook.txt")
        if os.path.exists(log_file):
            os.remove(log_file)

    to_process, processed, data_folder_nm, all_data_paths, wl = find_processed_folders(data_folders, parameters, PDDN)
    folder_data = create_folder_dict(data_folders, to_process, processed, data_folder_nm, all_data_paths, wl)
    
    if parameters['run_all']:
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


def find_processed_folders(data_folders: list, parameters: dict, PDDN: bool):
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
            raw_data_path = os.path.join(path, "raw_data", wl) if os.path.exists(os.path.join(path, "raw_data", wl)) else os.path.join(path, "raw_data")
            data_pres, data_path = is_there_data(raw_data_path, wl)
            data.append(data_pres)
            data_paths.append(data_path)

            pol_path = os.path.join(path, polarimetry_path[wl])
            condition_processed = os.path.exists(pol_path) and is_processed(path, wl, polarimetry_path[wl], parameters["processing_mode"], parameters["save_pdf_figs"])
            processed.append(condition_processed)
            if parameters["run_all"]:
                to_process.append(True)
            else:
                to_process.append(not condition_processed)

        to_process_dict[path] = to_process
        processed_dict[path] = processed
        data_presence[path] = data
        all_data_paths[path] = data_paths

    return to_process_dict, processed_dict, data_presence, all_data_paths, wavelengths


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


def create_folder_dict(data_folders, to_process, processed, data_folder_nm, all_data_paths, wavelengths):
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
                    "wavelength": wavelengths[idx],
                    "path_intensite": all_data_paths[folder][idx]
                })
    return folder_data


def is_there_data(path: str, wl: str) -> bool:
    """
    Checks if raw data is available in the given folder.

    Returns True if the expected `.cod` file is present.
    """
    if not os.path.isdir(path):
        return False
    expected_file = f"{wl.replace('nm', '')}_Intensite.cod"
    return expected_file in set(os.listdir(path)), os.path.join(path, expected_file)


def is_processed(path: str, wl: str, polarimetry_fname: str, processing_mode: str = "full", save_pdf_figs: bool = False) -> bool:
    """
    Checks if the required files (e.g., polarimetric plots) were generated.

    Returns True if all required files are present.
    """
    folder_path = os.path.join(path, polarimetry_fname, wl)
    if not os.path.exists(folder_path):
        return False
    expected_filenames = load_filenames(processing_mode, save_pdf_figs)
    return all(f in set(os.listdir(folder_path)) for f in expected_filenames)

def move_folder_for_processing(parameters: dict, folder: str):
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

def reorganize_folders(folder_path: str):
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
    wavelengths = load_wavelengths()

    # Create necessary directories for polarimetry and polarimetry_PDDN data
    for directory in directories_tbc:
        create_directory_structure(folder_path, directory, wavelengths)

    # Clean up old computations (remove unnecessary files)
    clean_up_old_files(folder_path['folder_name'], directories_tbc)


def create_directory_structure(folder_path: str, directory: str, wavelengths: list):
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
    
    for wl in wavelengths:
        os.makedirs(os.path.join(target_path, wl), exist_ok=True)


def clean_up_old_files(folder_path: str, dirs_to_check: list):
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


def get_calibration_dates(calib_directory: str):
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
    calib_directory_dates = os.listdir(calib_directory)
    wavelengths = load_wavelengths()

    calib_directory_dates_cleaned = [
        c for c in calib_directory_dates
        if any(os.path.exists(os.path.join(calib_directory, c, wl, f"{wl.split('nm')[0]}_W.mat"))
               or os.path.exists(os.path.join(calib_directory, c, wl, f"{wl.split('nm')[0]}_W.cod"))
               for wl in wavelengths)
    ]

    # Convert folder names to datetime objects
    return [datetime.strptime(c.split('_')[0], '%Y-%m-%d') for c in calib_directory_dates_cleaned]



def get_calibration_directory(calib_directory_dates_num: list, path: str, calib_directory: str,
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
    # Get the date of the measurement folder
    date_measurement = folder_eu_time.get(path.split(os.sep)[-1], get_date_measurement(path, idx))

    # Find the closest date in the calibration directory
    closest_date = find_closest_date(date_measurement, calib_directory_dates_num, wavelength, calib_directory, Flag)
    
    return os.path.join(calib_directory, closest_date)


def find_closest_date(date_measurement: datetime, calib_dates_complete: list, wavelength: str, calib_directory: str, Flag=False):
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
        if all_calibration_files_exist(last_calibration, calib_directory, wavelength):
            return last_calibration
        else:
            # Remove invalid calibration and check the previous one
            del dates_calibrated[idx_last_calib]

    raise FileNotFoundError('No valid calibration found for the closest 1000 days.')

def all_calibration_files_exist(calibration_folder: str, calib_directory: str, wavelength: str) -> bool:
    """Checks if all required calibration files exist for a given calibration folder and wavelength."""
    required_files = [f"{wavelength.split('nm')[0]}_A.cod", f"{wavelength.split('nm')[0]}_W.cod"]
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

def get_intensity(path: str):
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
    try:
        I = libmpMuelMat.read_cod_data_X3D(path, isRawFlag=0)
    except Exception:
        I = libmpMuelMat.read_cod_data_X3D(path, isRawFlag=1)

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


def curate_azimuth(azimuth: np.ndarray, folder=None) -> np.ndarray:
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
                mean_azi = np.nanmean(azi_neighbors)
                
                if not math.isnan(mean_azi):
                    azimuth[idx, idy] = mean_azi
                    counter += 1
                else:
                    azimuth[idx, idy] = 0

    # Optional: debug check for stability after modification
    if not libmpMuelMat._isNumStable(azimuth) and folder:
        print(f"Azimuth not stable in folder: {folder}")

    # Optionally, handle cases where too many pixels were modified
    if counter > 1 / 100 * azimuth.size:
        print(f"More than 1% of the pixels were modified in folder: {folder}.")

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

def process_mm(I, remove_reflection: bool, A, W):
    """Processes the Mueller matrix, removing reflections if necessary."""
    I_ref, dilated_mask = libmpMuelMat.removeReflections3D(I)

    if remove_reflection:
        processed = libmpMuelMat.process_MM_pipeline(A, I_ref, W, dilated_mask)
    else:
        processed = libmpMuelMat.process_MM_pipeline(A, I, W, I)

    return processed, dilated_mask

def load_calibration_data(calibration_directory_wl: str, wavelength: str):
    """Loads the calibration data (A and W) from the corresponding files."""
    files = os.listdir(calibration_directory_wl)
    wavelength_number = wavelength.replace('nm', '')
    if f"{wavelength_number}_A.cod" in files and f"{wavelength_number}_W.cod" in files:
        A = libmpMuelMat.read_cod_data_X3D(os.path.join(calibration_directory_wl, f"{wavelength_number}_A.cod"), isRawFlag=0)
        W = libmpMuelMat.read_cod_data_X3D(os.path.join(calibration_directory_wl, f"{wavelength_number}_W.cod"), isRawFlag=0)
    else:
        A, W = libmpMuelMat.calib_System_AW(calibration_directory_wl, wlen=int(wavelength_number))
    return A, W

def get_angle_correction(path: str) -> int:
    """Gets the angle correction from a file if available."""
    try:
        with open(os.path.join(path, 'rotation_MM.txt')) as f:
            return int(f.readline().strip())
    except FileNotFoundError:
        return 0


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


def get_cmap(parameter: str):
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
    parameters_plot = load_plot_parameters()[parameter]
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


def load_wavelengths():
    """
    load and returns the wavelengths usable by the IMP

    Returns
    -------
    wavelenghts : list
        the wavelengths usable by the IMP
    """
    return ['450nm', '500nm', '550nm', '600nm', '650nm', '700nm']

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

def load_plot_parameters():
    """
    load and returns the parameters for the polarimetric parameter plots

    Returns
    -------
    plot_parameters : dict
        the parameters to plot the parameters maps
    """
    with open(os.path.join(get_data_folder_path(), 'parameters_plot.json')) as json_file:
        data = json.load(json_file)
    return data


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

def get_data_folder_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
 
def getSupergluePath():
    import processingmm
    module_path = os.path.dirname(processingmm.__file__)
    return os.path.join(module_path, '..', '..', 'third_party', 'superglue')

def getPredictionPath():
    import processingmm
    module_path = os.path.dirname(processingmm.__file__)
    return os.path.join(module_path, '..', '..', 'third_party', 'polarfeat_mar25')

def test_pddn_models_existence(PDDN_models_path):
    """Check if required PDDN models exist."""
    models = ['PDDN_model_550_Fresh_HB.pt', 'PDDN_model_600_Fresh_HB.pt']
        
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