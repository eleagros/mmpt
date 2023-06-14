import re
import os
import pandas as pd
import shutil 
import traceback
from tqdm import tqdm

from processingmm.helpers import load_filenames, add_path, chunks, copytree
from processingmm import reorganize_folders, multi_img_processing, MM_processing, plot_polarimetry, visualization_lines
from processingmm import libmpMuelMat


def find_all_folders(directories):
    """
    walk through all of the directories present in the folders "directories" given as an input. finds
    the folder with the 202x-xx-xx name format and return the list of folders. separates the transmission
    and reflection measurements in two different lists - processed separatly

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
    data_folder_transmission : list
        the list with all the folders containing data for transmission
    folder_names_transmission : list
        the list with the name of all the measurements for transmission
    """
    data_folder = []
    folder_names = []

    for directory in directories:
        for root, dirs, files in os.walk(directory, topdown=False):
            if 'TRANSMISSION' in root:
                pass
            else:
                find_folder_name(root, data_folder, folder_names)
    return list(set(data_folder)), list(set(folder_names))

def find_folder_name(root, data_folder, folder_names):
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
        data_folder.append(splitted[0] + x + splitted[1].split('\\')[0])
        folder_names.append(x + splitted[1].split('\\')[0])
                    
    except Exception as e:
        pass


def find_processed_folders(data_folder):
    """
    iterates over each of the folder to find the ones containing 550nm/650nm raw data and determine the ones that have been processed

    Parameters
    ----------
    data_folder : list
        the list with all the folders containing data
    transmission : 
        boolean indicating whether or not we are processing transmission data
    
    Returns
    -------
    processed_ : list
        a list of booleans indicating wether the folder has been processed
    data_folder_nm : dict
        a dictionnary indicating wether the data for 550nm, 650nm or both was obtained for each folder
    """
    wavelenghts = ['450nm', '500nm', '550nm', '600nm', '650nm', '700nm']

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

def is_there_data(path):
    """
    check if raw data is available for the path given as an input

    Parameters
    ----------
    path : str
        the path to the folder containing the raw data
    
    Returns
    -------
    data_exist : bool
        boolean indicating the presence of one or more .cod files
    """
    data_exist = False
    try:
        data_exist = len(os.listdir(path)) == 2
    except FileNotFoundError:
        data_exist = False
    return data_exist


def is_processed(path, wl):
    """
    check if the folders for which raw data is available have been processed (i.e. contains more than 'treshold' files)

    Parameters
    ----------
    path : str
        the path to the folder to check
    nm550_data : bool
        boolean indicating whether or not data has been obtained for 550nm
    nm650_data : bool
        boolean indicating whether or not data has been obtained for 650nm
    processed : list
        a list of booleans indicating wether the folder has been processed
    prefix : str
        the folder in which the polarimetry data is located ('/polarimetry' by default)
    treshold : int
        the number of files to be found so that the folder is considered as processed (20 by default)
    """
    filenames = load_filenames()
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


def create_folders_df(data_folder, processed, data_folder_nm, wavelenghts):
    """
    create a dataframe referencing the path to the folder and if it has been processed

    Parameters
    ----------
    data_folder : list
        the list with all the folders containing data
    processed_ : list 
        a list of booleans indicating wether the folder has been processed
    save_files : bool
        a boolean indicating wether or not to save the df as excel file 
    transmission : bool
        boolean indicating whether or not we are processing transmission data
        
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


def process_MM(measurement_directory, calib_directory):
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

    MuellerMatrices, calibration_directories = MM_processing.compute_analysis_python(measurement_directory, 
                                        calib_directory_dates_num, calib_directory, to_compute, run_all = run_all, batch_processing = True, Flag = False)

    MuellerMatrices_raw = MuellerMatrices

    for folder, _ in MuellerMatrices.items():
        plot_polarimetry.parameters_histograms(MuellerMatrices_raw, folder)

    for folder, _ in MuellerMatrices.items():
        plot_polarimetry.generate_plots(MuellerMatrices, folder)
        X_montage = plot_polarimetry.show_MM(MuellerMatrices[folder]['nM'], folder)
        plot_polarimetry.MM_histogram(MuellerMatrices, folder)
        plot_polarimetry.save_batch(folder)

    parameters_visualizations = {
        'PD': {'depolarization': 0.50, 'linear_retardance': 4, 'greyscale': 1.50, 'std_parameter': 50,
            'n': 12, 'widths': 1.5, 'scale': 25},
        'CUSA': {'depolarization': 0.96, 'linear_retardance': 12, 'greyscale': 1.50, 'std_parameter': 50,
            'n': 10, 'widths': 1.6, 'scale': 15},
        'fixed_brain': {'depolarization': 0.88, 'linear_retardance': 15, 'greyscale': 1.08, 'std_parameter': 50,
            'n': 10, 'widths': 1.8, 'scale': 25},
        'fixed_brain_EANS': {'depolarization': 0.90, 'linear_retardance': 3, 'greyscale': 1.80, 'std_parameter': 50,
            'n': 12, 'widths': 1.8, 'scale': 25},
        'fixed_brain': {'depolarization': 0.90, 'linear_retardance': 3, 'greyscale': 1.50, 'std_parameter': 50,
            'n': 12, 'widths': 1.8, 'scale': 25}
    }

    measurements_directory_viz = measurement_directory
    parameters_set = 'CUSA'
    param = visualization_lines.visualization_auto(measurements_directory_viz, parameters_visualizations, 
                                                parameters_set, run_all = True, batch_processing = True)

    print('here')

    for folder, _ in MuellerMatrices.items():
        visualization_lines.save_batch(folder)

    return calibration_directories, parameters_set


def batch_process(directories, calib_directory):
    data_folder, folder_names = find_all_folders(directories)
    
    # return two list booleans and a dict linking folders and the indication of if the folders have been processed
    processed, data_folder_nm, wavelenghts = find_processed_folders(data_folder)
    df = create_folders_df(data_folder, processed, data_folder_nm, wavelenghts)
    
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

        links_folders = {}
        for folder in chunk:
            links_folders[os.path.join('./temp_processing', folder.split('\\')[-1])] = folder
            shutil.move(folder, os.path.join('./temp_processing', folder.split('\\')[-1]))
            to_process_temp.append(os.path.join('./temp_processing', folder.split('\\')[-1]))
        
        measurements_directory = './temp_processing'
        calibration_directories, parameters_set = process_MM(measurements_directory, calib_directory)
        
        for folder in to_process_temp:
            logbook_MM_processing = open(os.path.join(folder, 'MMProcessing.txt'), 'w')
            logbook_MM_processing.write('Processed: true\n')
            logbook_MM_processing.write(calibration_directories[folder.split('\\')[-1]] + '\n')
            logbook_MM_processing.write(parameters_set + '\n')
            logbook_MM_processing.write(libmpMuelMat.__version__)
            logbook_MM_processing.close()

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