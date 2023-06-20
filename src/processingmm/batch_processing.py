import re
import os
import pandas as pd
import shutil 
import traceback
from tqdm import tqdm
import traceback

from processingmm.helpers import load_filenames, add_path, chunks, load_wavelengths, is_there_data, is_processed
from processingmm import reorganize_folders, multi_img_processing, MM_processing, plot_polarimetry, visualization_lines
from processingmm import libmpMuelMat


def find_all_folders(directories: list):
    """
    walk through all of the directories present in the folders "directories" given as an input. finds the folder with the 202x-xx-xx name format 
    and return the list of folders.

    Parameters
    ----------
    directories : list of str
        a list containing the directories to scan

    Returns
    -------
    data_folder : list
        the list with all the folders containing data
    folder_names : list
        the list with the name of all the measurements
    """
    data_folder = []
    folder_names = []

    for directory in directories:
        for root, dirs, files in os.walk(directory, topdown=False):
            # remove the transmission measurements
            if 'TRANSMISSION' in root:
                pass
            else:
                find_folder_name(root, data_folder, folder_names)
    return list(set(data_folder)), list(set(folder_names))


def find_folder_name(root: str, data_folder: list, folder_names: list):
    """
    check if a folder has the correct format: 202x-xx-xx. if yes, add it to the list of folders

    Parameters
    ----------
    root : str
        complete path to the folder
    data_folder : list
        the list with all the folders containing data
    folder_names : list
        the list with the name of all the measurements
    """
    try:
        # check if the folder name format is 202x-xx-xx
        assert len(re.findall("[\d]{4}-[\d]{2}-[\d]{2}", root)) == 1
        x = re.search("[\d]{4}-[\d]{2}-[\d]{2}", root).group(0)
        splitted = root.split(x)
                    
        # if yes, append it to the lists containing the folder names
        data_folder.append(os.path.join(splitted[0],  x + splitted[1].split('\\')[0]))
        folder_names.append(x + splitted[1].split('\\')[0])
                    
    except Exception as e:
        pass


def find_processed_folders(data_folder: list):
    """
    iterates over each of the folder to find the ones containing 550nm/650nm raw data and determine the ones that have been processed

    Parameters
    ----------
    transmission : 
        boolean indicating whether or not we are processing transmission data
    
    Returns
    -------
    processed_nm : dict of list
        a dictionnary containing lists indicating if the files (i.e. polarimetric plots, etc...) were generated for each wavelength
    data_folder_nm : dict of list
        a dictionnary containing lists indicating if the data files are present for each wavelength
    wavelenghts : list
        the list of the wavelengths usable with the IMP
    """
    wavelenghts = load_wavelengths()

    processed_nm = {}
    data_folder_nm = {}

    # iterate over each folder containing data
    for path in data_folder:
        data = []
        processed = []
        for wl in wavelenghts:

            # check if raw data is available for the different measurements
            if os.path.exists(os.path.join(path, 'raw_data')):
                data.append(is_there_data(os.path.join(path, 'raw_data', wl)))
            else:
                data.append(is_there_data(os.path.join(path, wl)))
            if os.path.exists(os.path.join(path, 'polarimetry')):
                processed.append(is_processed(path, wl))
            else:
                processed.append(False)
        
        # add the information to lists
        processed_nm[path] = processed
        data_folder_nm[path] = data
        
    return processed_nm, data_folder_nm, wavelenghts


def is_processed(path: str, wl: str):
    """
    check if the files (i.e. polarimetric plots, etc...) were generated for the specified wavelenght

    Parameters
    ----------
    path : str
        the path to the folder to check
    wl : str
        the wavelenghts for which to check

    Returns
    ----------
    all_found : bool
        indicates if all the files were present
    """
    filenames = load_filenames()

    # get the filenames
    all_file_names = os.listdir(os.path.join(path, 'polarimetry', wl))
    all_found = True
    for filename in filenames:
        found_file = False
        if filename == '_realsize.png':
            for file in all_file_names:
                if file.endswith(filename):
                    found_file = True
        else:
            for file in all_file_names:
                if file == filename:
                    found_file = True
        if not found_file:
            all_found = False
    return all_found


def create_folders_df(data_folder: list, processed: dict, data_folder_nm: dict, wavelenghts: list):
    """
    create a dataframe referencing the path to the folder and if it has been processed

    Parameters
    ----------
    data_folder : list
        the list with all the folders containing data
    processed : dict of list
        a dictionnary containing lists indicating if the files (i.e. polarimetric plots, etc...) were generated for each wavelength
    data_folder_nm : dict of list
        a dictionnary containing lists indicating if the data files are present for each wavelength
    wavelenghts : list
        the list of the wavelengths usable with the IMP
        
    Returns
    -------
    df : pd.dataframe
        a dataframe referencing the path to the folder and if it has been processed
    """
    df_list = []
    for folder in data_folder:
        for idx, boolean in enumerate(processed[folder]):
            df_list.append([folder, boolean, data_folder_nm[folder][idx], wavelenghts[idx]])
    df = pd.DataFrame(df_list, columns = ['folder name', 'processed', 'data presence', 'wavelength'])
    df = df[df['data presence']]
    df = df.reset_index(drop=True)
    return df


def process_MM(measurement_directory: str, calib_directory: str):
    """
    master function allowing to reogranize the folders, compute the MMs, generate the plots and the visualizations for one directory

    Parameters
    ----------
    measurement_directory : str
        the path to the directory in which the measurement are located
    calib_directory : str
        the path to the directory in which the calibration data is located
        
    Returns
    -------
    calibration_directories : list
        the list of the calibration folders used to process the data (used to save it for tracability purposes)
    parameters_set : str
        the name of the parameters_set used for the line visualization
    """
    # obtain the directories that we should reorganize the data for
    all_directories = os.listdir(measurement_directory)

    # 1. move the raw data folders into the raw data directory
    directories_tbc = ['polarimetry', 'histology', 'annotation']
    wavelenghts = ['400nm', '450nm', '500nm', '550nm', '600nm', '650nm', '700nm']
    for directory in all_directories:
        reorganize_folders.move_raw_data_folders(directory, measurement_directory)

    # 2. create the new directories that will contain histology, pictures, polarimetric data and 50x50 measurements
    for directory in all_directories:
        reorganize_folders.create_directories(measurement_directory, directory, directories_tbc, wavelenghts)
        
    # 3. empty polarimetry and 50x50 directories
    for directory in all_directories:
        reorganize_folders.remove_old_computation(measurement_directory, directory)
        
    # 4. move what was already computed into the new folders
    for directory in all_directories:
        reorganize_folders.move_50x50_images(measurement_directory, directory)

    run_all = True
    to_compute = multi_img_processing.get_dir_to_compute(measurement_directory, run_all = run_all)
    calib_directory_dates_num = multi_img_processing.get_calibration_dates(calib_directory)

    # compute the MMs
    MuellerMatrices, calibration_directories = MM_processing.compute_analysis_python(measurement_directory, 
                                        calib_directory_dates_num, calib_directory, to_compute, run_all = run_all, batch_processing = True, Flag = False)
    MuellerMatrices_raw = MuellerMatrices

    # and generate the different plots
    for folder, _ in MuellerMatrices.items():
        plot_polarimetry.parameters_histograms(MuellerMatrices_raw, folder)

    for folder, _ in MuellerMatrices.items():
        plot_polarimetry.generate_plots(MuellerMatrices, folder)
        X_montage = plot_polarimetry.show_MM(MuellerMatrices[folder]['nM'], folder)
        plot_polarimetry.MM_histogram(MuellerMatrices, folder)
        plot_polarimetry.save_batch(folder)

    # finally, generate the visuazation with the lines
    measurements_directory_viz = measurement_directory
    parameters_set = 'CUSA'
    _ = visualization_lines.visualization_auto(measurements_directory_viz, parameters_set, run_all = False, batch_processing = True)
    for folder, _ in MuellerMatrices.items():
        plot_polarimetry.save_batch(folder, viz = True)

    return calibration_directories, parameters_set


def batch_process(directories: list, calib_directory: str):
    """
    master function allowing to apply the mueller matrix processing pipeline to all the measurement folders located in one or multiple directories

    Parameters
    ----------
    directories : list
        the list of the directories in which the measurement folders are located
    calib_directory : str
        the path to the calibration directory
    """
    # get all the names of the measurement folders
    data_folder, _ = find_all_folders(directories)
    for folder in data_folder:
        try:
            os.remove(os.path.join(folder, 'processing_logbook.txt'))
        except:
            pass

    # return two list booleans and a dict linking folders and the indication of if the folders have been processed
    processed, data_folder_nm, wavelenghts = find_processed_folders(data_folder)
    df = create_folders_df(data_folder, processed, data_folder_nm, wavelenghts)
    
    # get the files that needs to be processed
    if len(df) == 0:
        to_process = []
    else:
        to_process = df[~df['processed']]

    if len(to_process) == 0:
        to_process = []
    else:
        to_process = list(set(list(to_process.reset_index(level=0).apply(add_path, axis = 1))))

    try:
        os.mkdir('./temp_processing')
    except FileExistsError:
        pass

    # get the different chunks that are used to split the processing of the data
    all_chunks = chunks(to_process, 2)
    all_chunks = list(all_chunks)

    for chunk in tqdm(all_chunks):
    
        to_process_temp = []
        
        try:
            shutil.rmtree('./temp_processing')
        except FileNotFoundError:
            pass
        except:
            traceback.print_exc()
        try:
            os.mkdir('./temp_processing')
        except FileExistsError:
            pass

        # move the measurement folders to the temp_processing folder
        links_folders = {}
        for folder in chunk:
            links_folders[os.path.join('./temp_processing', folder.split('\\')[-1])] = folder
            shutil.move(folder, os.path.join('./temp_processing', folder.split('\\')[-1]))
            to_process_temp.append(os.path.join('./temp_processing', folder.split('\\')[-1]))
        
        # process the mueller matrix and generate the visualizations
        measurements_directory = './temp_processing'
        calibration_directories, parameters_set = process_MM(measurements_directory, calib_directory)
        
        for folder in to_process_temp:
            logbook_MM_processing = open(os.path.join(folder, 'MMProcessing.txt'), 'w')
            logbook_MM_processing.write('Processed: true\n')
            logbook_MM_processing.write(calibration_directories[folder.split('\\')[-1]] + '\n')
            logbook_MM_processing.write(parameters_set + '\n')
            logbook_MM_processing.write(libmpMuelMat.__version__)
            logbook_MM_processing.close()

        # put back the folders in the original folder
        links_folders = {v: k for k, v in links_folders.items()}
        for folder, temp_folder in links_folders.items():
            try:
                shutil.rmtree(folder)
            except FileNotFoundError:
                pass
            shutil.move(temp_folder, folder)
    
        try:
            shutil.rmtree('./temp_processing')
        except FileNotFoundError:
            pass
        except:
            traceback.print_exc()