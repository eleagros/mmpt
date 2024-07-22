import re
import os
import pandas as pd
import shutil 
import traceback
from tqdm import tqdm
import traceback
from packaging.version import Version

from processingmm.helpers import load_filenames, add_path, chunks, load_wavelengths, is_there_data, is_processed
from processingmm import reorganize_folders, multi_img_processing, MM_processing, plot_polarimetry, visualization_lines
from processingmm import libmpMuelMat
import processingmm
import time

def batch_process_master(directories, calib_directory, run_all = False, parameter_set = 'TheoniPics', PDDN = 'no',
                         wavelengths = 'all', processing_mode = 'full', remove_reflection = True, folder_eu_time = {},
                         time_mode = True):
    
    assert PDDN in ['no', 'pddn', 'both'], ("PDDN_mode should be one of the following: ['no', 'pddn', 'both'].")

    if PDDN == 'no':
        print('processing without PDDN...')
        times, time_complete = batch_process(directories, calib_directory, run_all = run_all, parameter_set = parameter_set, PDDN = False,
                      wavelengths = wavelengths, processing_mode = processing_mode, remove_reflection = remove_reflection,
                      folder_eu_time = folder_eu_time, time_mode = time_mode)
        print('processing without PDDN done.')
        
    else:
        assert Version(processingmm.__version__) >= Version('1.1'), ("Please update the processingmm package to version 1.1 or higher to use PDDN.")
        if PDDN == 'pddn':
            print('processing with PDDN...')
            times, time_complete = batch_process(directories, calib_directory, run_all = run_all, parameter_set = parameter_set, PDDN = True,
                        wavelengths = wavelengths, processing_mode = processing_mode, remove_reflection = remove_reflection,
                        folder_eu_time = folder_eu_time, time_mode = time_mode)

            print('processing with PDDN done.')
        else:
            print('1. processing without PDDN...')
            times, time_complete = batch_process(directories, calib_directory, run_all = run_all, parameter_set = parameter_set, PDDN = False,
                        wavelengths = wavelengths, processing_mode = processing_mode, remove_reflection = remove_reflection,
                        folder_eu_time = folder_eu_time, time_mode = time_mode)

            print('processing without PDDN done.')
            print()
            print('2. processing with PDDN.')
            batch_process(directories, calib_directory, run_all = run_all, parameter_set = parameter_set, PDDN = True,
                        wavelengths = wavelengths, processing_mode = processing_mode, remove_reflection = remove_reflection,
                        folder_eu_time = folder_eu_time, time_mode = time_mode)

            print('processing with PDDN done.')
            
    return times, time_complete


def batch_process(directories: list, calib_directory: str, folder_eu_time: dict = {}, run_all: bool = False, 
                  parameter_set: str = None, PDDN = False, remove_reflection = True, wavelengths = 'all', processing_mode = 'full',
                  time_mode = False):
    """
    master function allowing to apply the mueller matrix processing pipeline to all the measurement folders located in one or multiple directories

    Parameters
    ----------
    directories : list
        the list of the directories in which the measurement folders are located
    calib_directory : str
        the path to the calibration directory
    folder_eu_time : dict
        used to remap the times to the correct ones (for the measurements made in Chicago)
    run_all : bool
        wether the processed folders should be reprocessed (default is False)
    parameter_set : str
        the name of the set of the parameters to be used for the visualization
    max_nb
    """    
    
    # start recording time for the whole processing
    start = time.time()
    
    # get all the names of the measurement folders
    df, wl = get_df_processing(directories, PDDN = PDDN, wavelengths = wavelengths, processing_mode = processing_mode)
    to_process = get_to_process(df, run_all = run_all)

    try:
        os.mkdir('./temp_processing')
    except FileExistsError:
        pass

    # get the different chunks that are used to split the processing of the data
    if time_mode:
        all_chunks = chunks(to_process, 2)
    else:
        all_chunks = chunks(to_process, 1)
        
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
        
        links_folders, to_process_temp = move_the_folders_pre_processing(chunk, to_process_temp)
        
        # process the mueller matrix and generate the visualizations
        measurements_directory = './temp_processing'
        calibration_directories, parameters_set, times = process_MM(measurements_directory, calib_directory, 
                                                folder_eu_time = folder_eu_time, run_all = run_all, 
                                                parameter_set = parameter_set, PDDN = PDDN, 
                                                remove_reflection = remove_reflection, wavelengths = wl,
                                                processing_mode = processing_mode, time_mode = time_mode)
    
        
        for folder in to_process_temp:
            try:
                logbook_MM_processing = open(os.path.join(folder, 'MMProcessing.txt'), 'w')
                logbook_MM_processing.write('Processed: true\n')
                logbook_MM_processing.write(calibration_directories[folder.split('\\')[-1]] + '\n')
                logbook_MM_processing.write(parameters_set + '\n')
                logbook_MM_processing.write(libmpMuelMat.__version__ + '\n')
                import processingmm
                logbook_MM_processing.write(processingmm.__version__)
                logbook_MM_processing.close()
            except:
                logbook_MM_processing.close()
                traceback.print_exc()

        # put back the folders in the original folder
        links_folders = {v: k for k, v in links_folders.items()}
        for folder, temp_folder in links_folders.items():
            if PDDN:
                to_remove = ['polarimetry']
            else:
                to_remove = ['polarimetry_PDDN']
                    
            for fold in os.listdir(temp_folder):
                
                if fold == to_remove[0] or fold == 'annotation' or fold == 'histology':
                    pass
                elif fold == 'MMProcessing.txt':
                    shutil.copy(os.path.join(temp_folder, fold), os.path.join(folder, fold))
                else:
                    try:
                        shutil.rmtree(os.path.join(folder, fold))
                    except FileNotFoundError:
                        pass
                    except:
                        traceback.print_exc()
                    
                    shutil.move(os.path.join(temp_folder, fold), os.path.join(folder, fold))
            

        try:
            shutil.rmtree('./temp_processing')
        except FileNotFoundError:
            pass
        except:
            traceback.print_exc()

    end = time.time()
    time_complete = end - start
    
    try:
        return times, time_complete
    except UnboundLocalError:
        return None, time_complete
    
    
def move_the_folders_pre_processing(chunk: list, to_process_temp):
    
    # move the measurement folders to the temp_processing folder
    links_folders = {}
    for folder in chunk:
        links_folders[os.path.join('./temp_processing', folder.split('\\')[-1])] = folder
        os.mkdir(os.path.join('./temp_processing', folder.split('\\')[-1]))
        for file in os.listdir(os.path.join(folder, 'raw_data')):
            src = os.path.join(folder, 'raw_data', file)
            dst = os.path.join('./temp_processing', folder.split('\\')[-1], file)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy(src, dst)
        to_process_temp.append(os.path.join('./temp_processing', folder.split('\\')[-1]))
        
    return links_folders, to_process_temp
            
            
def get_df_processing(directories: list, PDDN = False, wavelengths = 'all', processing_mode = 'full'):
    data_folder, _ = get_all_folders(directories)
    
    # remove the processing_logbook (old version)
    for folder in data_folder:
        try:
            os.remove(os.path.join(folder, 'processing_logbook.txt'))
        except:
            pass
        
    # return two list booleans and a dict linking folders and the indication of if the folders have been processed
    processed, data_folder_nm, wl = find_processed_folders(data_folder, PDDN = PDDN, 
                                                    wavelengths = wavelengths, processing_mode = processing_mode)
    df = create_folders_df(data_folder, processed, data_folder_nm,
                           wavelengths = wl)
    return df, wl

def get_all_folders(directories: list, win7: bool = False):
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

        for root, _, _ in os.walk(directory, topdown=False):

            # remove the transmission measurements
            if 'TRANSMISSION' in root:
                pass
            else:
                get_folder_name(root, data_folder, folder_names)
    
    if win7:
        return f7(data_folder), f7(folder_names)
    else:
        return list(set(data_folder)), list(set(folder_names))
    
    
def get_folder_name(root: str, data_folder: list, folder_names: list):
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
    
    
def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def find_processed_folders(data_folder: list, PDDN = False, wavelengths = 'all', processing_mode = 'full'):
    """
    iterates over each of the folder to find the ones containing 550nm/650nm raw data and determine the ones that have been processed

    Parameters
    ----------
    data_folder
    PDDN : bool
        indicates wether denosing is used (default is False)
    wavelengths : str or list
        indicates which wavelenghts should be processed (default is 'all')
    processing_mode : str
        indicates the processing mode ('full', 'no_visualization' or 'fast', default is 'full')
    
    Returns
    -------
    processed_nm : dict of list
        a dictionnary containing lists indicating if the files (i.e. polarimetric plots, etc...) were generated for each wavelength
    data_folder_nm : dict of list
        a dictionnary containing lists indicating if the data files are present for each wavelength
    wavelenghts : list
        the list of the wavelengths usable with the IMP
    
    """
    
    processed_nm, data_folder_nm = {}, {}
    wavelengths_compare = get_wavelenghts_processing(wavelengths)
    
    # iterate over each folder containing data
    for path in data_folder:
        data, processed = [], []

        for wl in wavelengths_compare:
    
            path_PDDN_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), r'PDDN_model\PDDN_model_' + str(wl).split('nm')[0] + '_Fresh_HB.pt')

            path_polarimetry = 'polarimetry_PDDN'  if PDDN and os.path.exists(path_PDDN_model) else 'polarimetry'
            
            # check if raw data is available for the different measurements
            path_wl = os.path.join(path, 'raw_data', wl) if os.path.exists(os.path.join(path, 'raw_data')) else os.path.join(path, wl)
            data.append(is_there_data(path_wl))

            if os.path.exists(os.path.join(path, path_polarimetry)):
                try:
                    os.mkdir(os.path.join(os.path.join(path, path_polarimetry, wl)))
                except FileExistsError:
                    pass
                processed.append(is_processed(path, wl, path_polarimetry, processing_mode = processing_mode))
            else:
                processed.append(False)
        
        # add the information to lists
        processed_nm[path] = processed
        data_folder_nm[path] = data
    return processed_nm, data_folder_nm, wavelengths_compare


def get_wavelenghts_processing(wavelengths):
    if wavelengths == 'all':
        wavelengths_compare = load_wavelengths()
    else:
        assert type(wavelengths) == list, ("Wavelengths should be a list of int (or string).")
        wavelengths_compare = []
        for wl in wavelengths:
            if type(wl) == int:
                wavelengths_compare.append(str(wl) + 'nm')
            else:
                if 'nm' not in wl:
                    wl = wl + 'nm'
                wavelengths_compare.append(wl)
    return wavelengths_compare

def is_processed(path: str, wl: str, path_polarimetry: str, processing_mode = 'full'):
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
    filenames = load_filenames(processing_mode)
    
    # get the filenames
    all_file_names = os.listdir(os.path.join(path, path_polarimetry, wl))
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


def create_folders_df(data_folder: list, processed: dict, data_folder_nm: dict, wavelengths: list):
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
            df_list.append([folder, boolean, data_folder_nm[folder][idx], wavelengths[idx]])
    df = pd.DataFrame(df_list, columns = ['folder name', 'processed', 'data presence', 'wavelength'])
    df = df[df['data presence']]
    df = df.reset_index(drop=True)
    return df


def process_MM(measurement_directory: str, calib_directory: str, folder_eu_time: dict = {}, run_all: bool = False, 
               parameter_set: str = None, PDDN = False, remove_reflection: bool = False, wavelengths = [],
               processing_mode = 'full', time_mode = False):
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
    directories_tbc = ['polarimetry', 'polarimetry_PDDN', 'histology', 'annotation']
    # 
    for directory in all_directories:
        reorganize_folders.move_raw_data_folders(directory, measurement_directory)

    wavelenghts_create_folders = ['400nm', '450nm', '500nm', '550nm', '600nm', '650nm', '700nm']
    # 2. create the new directories that will contain histology, pictures, polarimetric data and 50x50 measurements
    for directory in all_directories:
        reorganize_folders.create_directories(measurement_directory, directory, directories_tbc, wavelenghts_create_folders)
        
    # 3. empty polarimetry and 50x50 directories
    for directory in all_directories:
        reorganize_folders.remove_old_computation(measurement_directory, directory)
        
    # 4. move what was already computed into the new folders
    for directory in all_directories:
        reorganize_folders.move_50x50_images(measurement_directory, directory)


    to_compute = multi_img_processing.remove_already_computed_folders(measurement_directory, 
                                                                      run_all = run_all, 
                                                                      PDDN = PDDN,
                                                                      wavelengths = wavelengths,
                                                                      Flag = False)

    calib_directory_dates_num = multi_img_processing.get_calibration_dates(calib_directory)
    
    # compute the MMs
    MuellerMatrices, calibration_directories, times = MM_processing.compute_analysis_python(measurement_directory, 
                                        calib_directory_dates_num, calib_directory, to_compute, 
                                        remove_reflection = remove_reflection, folder_eu_time = folder_eu_time, 
                                        run_all = run_all, batch_processing = True, Flag = False, PDDN = PDDN,
                                        wavelengths = wavelengths, processing_mode = processing_mode, time_mode = time_mode)
    
    MuellerMatrices_raw = MuellerMatrices
    
    start = time.time()
    
    # and generate the different plots
    for folder, _ in MuellerMatrices.items():
        plot_polarimetry.parameters_histograms(MuellerMatrices_raw, folder)
    
    for folder, _ in MuellerMatrices.items():
        plot_polarimetry.generate_plots(MuellerMatrices, folder)
        _ = plot_polarimetry.show_MM(MuellerMatrices[folder]['nM'], folder)
        plot_polarimetry.MM_histogram(MuellerMatrices, folder)
        plot_polarimetry.save_batch(folder)
        
    end = time.time()
    time_plotting = end - start
    times['plotting'] = time_plotting

    if processing_mode == 'full':
        
        # finally, generate the visuazation with the lines
        measurements_directory_viz = measurement_directory
        if parameter_set == None:
            parameter_set = 'CUSA'

        _ = visualization_lines.visualization_auto(measurements_directory_viz, parameter_set, run_all = run_all, 
                                                batch_processing = False, PDDN = PDDN, wavelengths = wavelengths)

        for folder, _ in MuellerMatrices.items():
            plot_polarimetry.save_batch(folder, viz = True)
    
    return calibration_directories, parameter_set, times
    
def get_to_process(df: pd.DataFrame, run_all: bool = False, inverse: bool = False):
    # get the files that needs to be processed
    if len(df) == 0:
        to_process = []
    else:
        if run_all:
            to_process = df
        else:
            if inverse:
                to_process = df[df['processed']]
            else:
                to_process = df[~df['processed']]

    if len(to_process) == 0:
        to_process = []
    else:
        to_process = list(set(list(to_process.reset_index(level=0).apply(add_path, axis = 1))))

    return to_process