import os, shutil
import json

import numpy as np
from scipy import ndimage

import re

import matplotlib.colors as clr

import pandas as pd


####################################################################################################################################
####################################################################################################################################
################################### 1. Get the folders that will be processed ######################################################
####################################################################################################################################
####################################################################################################################################

def get_to_process(parameters: dict, PDDN: bool = False, inverse: bool = False):
    """
    get the folders that needs to be processed
    
    Parameters:
    ----------
    parameters : dict
        the parameters used for the processing
    PDDN : bool
        indicates wether denosing is used
    inverse : bool
        indicates if the folders that have been processed should be returned
        
    Returns:
    -------
    wl : list
        the list of the wavelengths for which data is available
    to_process : list
        the list of the folders that needs to be processed
    
    Raises:
    -------
    None
    """
    df, wl = get_df_processing(parameters['directories'], PDDN = PDDN, wavelengths = parameters['wavelengths'], processing_mode = parameters['processing_mode'],
                               save_pdf_figs = parameters['save_pdf_figs'])
    
    # get the files that needs to be processed
    if len(df) == 0:
        to_process = []
    else:
        if parameters['run_all']:
            to_process = df
        else:
            if inverse:
                to_process = df[df['processed']]
            else:
                to_process = df[~df['processed']]

    if len(to_process) == 0:
        to_process = []
    else:
        to_process = list(set(list(to_process.reset_index(level=0).apply(lambda row: row['folder name'], axis=1))))
    return wl, to_process, df
    
    
def get_df_processing(directories: list, PDDN = False, wavelengths = 'all', processing_mode = 'full', save_pdf_figs = False):
    """
    get the dataframe containing the information about the folders and if they have been processed

    Parameters:
    ----------
    directories : list
        the list of directories to scan
    PDDN : bool
        indicates wether denosing is used (default is False)
    wavelengths : str or list
        indicates which wavelenghts should be processed (default is 'all')
    processing_mode : str
        indicates the processing mode ('full', 'no_visualization' or 'fast', default is 'full')

    Returns:
    -------
    df : pd.dataframe
        a dataframe referencing the path to the folder and if it has been processed
    wl : list
        the list of the wavelengths for which data is available
        
    Raises:
    -------
    None
    """
    data_folder, _ = get_all_folders(directories)

    # remove the processing_logbook (old version)
    for folder in data_folder:
        try:
            os.remove(os.path.join(folder, 'processing_logbook.txt'))
        except:
            pass
        
    # return two list booleans and a dict linking folders and the indication of if the folders have been processed
    processed, data_folder_nm, wl = find_processed_folders(data_folder, PDDN = PDDN, 
                                                    wavelengths = wavelengths, processing_mode = processing_mode, 
                                                    save_pdf_figs = save_pdf_figs)

    df = create_folders_df(data_folder, processed, data_folder_nm,
                           wavelengths = wl)
    return df, wl


def get_all_folders(directories: list, win7: bool = False):
    """
    walk through all of the directories present in the folders "directories" given as an input. finds the folder with the 
    202x-xx-xx name format and return the list of folders.

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
        assert len(re.findall(r"[\d]{4}-[\d]{2}-[\d]{2}", root)) == 1
        x = re.search(r"[\d]{4}-[\d]{2}-[\d]{2}", root).group(0)
        splitted = root.split(x)
        
        # if yes, append it to the lists containing the folder names
        data_folder.append(os.path.join(splitted[0],  x + splitted[1].split('/')[0]))
        folder_names.append(x + splitted[1].split('/')[0])
                    
    except Exception as e:
        pass
    
def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def find_processed_folders(data_folder: list, PDDN = False, wavelengths = 'all', processing_mode = 'full', save_pdf_figs = False):
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
    
            path_PDDN_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), r'PDDN_model/PDDN_model_' + str(wl).split('nm')[0] + '_Fresh_HB.pt')

            path_polarimetry = 'polarimetry_PDDN' if PDDN and os.path.exists(path_PDDN_model) else 'polarimetry'
            
            # check if raw data is available for the different measurements
            path_wl = os.path.join(path, 'raw_data', wl) if os.path.exists(os.path.join(path, 'raw_data')) else os.path.join(path, wl)
            data.append(is_there_data(path_wl))

            if os.path.exists(os.path.join(path, path_polarimetry)):
                try:
                    os.mkdir(os.path.join(os.path.join(path, path_polarimetry, wl)))
                except FileExistsError:
                    pass
                          
                processed.append(is_processed(path, wl, PDDN, processing_mode = processing_mode, save_pdf_figs = save_pdf_figs))
            else:
                processed.append(False)
        
        # add the information to lists
        processed_nm[path] = processed
        data_folder_nm[path] = data
    return processed_nm, data_folder_nm, wavelengths_compare


def is_there_data(path: str):
    """
    check if raw data is available for the path given as an input

    Parameters
    ----------
    path : str
        the path to the folder containing the raw data
    
    Returns
    -------
    data_exist : bool
        boolean indicating the presence of two .cod files
    """
    data_exist = False
    all_files = os.listdir(path)
    files = []
    for f in all_files:
        if 'PDDN' in f or 'aligned' in f:
            pass
        else:
            files.append(f)
    
    try:
        data_exist = len(files) == 2
    except FileNotFoundError:
        data_exist = False

    return data_exist


def is_processed(path: str, wl: str, PDDN: bool = False, processing_mode = 'full', save_pdf_figs = False):
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
    all_found : is_processed
        indicates if all the files were present
    """
    filenames = load_filenames(processing_mode = processing_mode, save_pdf_figs = save_pdf_figs)

    path_PDDN_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), r'PDDN_model/PDDN_model_' + str(wl.split('nm')[0]) + '_Fresh_HB.pt')
    
    if PDDN and os.path.exists(path_PDDN_model):
        polarimetry_fname = 'polarimetry_PDDN'
    else:
        polarimetry_fname = 'polarimetry'
        
    # get the filenames
    all_file_names = os.listdir(os.path.join(path, polarimetry_fname, wl))
    
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



####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################



def move_the_folders_pre_processing(parameters, chunk: list):
    """
    move the measurement folders to the temp_processing folder
    
    Parameters
    ----------
    parameters : dict
        the processing parameters in a dictionary format
    chunk : list
        the list of the folders to be processed
        
    Returns
    -------
    links_folders : dict
        the dictionary containing the links between the original folders and the temp_processing folders
    to_process_temp : list
        the list of the folders to be processed in the temp_processing folder
    
    Raises
    ------
    None
    """
    to_process_temp = []
    
    # move the measurement folders to the temp_processing folder
    links_folders = {}
    for folder in chunk:
        
        links_folders[os.path.join(parameters['temp_folder'], folder.split('/')[-1])] = folder
        os.mkdir(os.path.join(parameters['temp_folder'], folder.split('/')[-1]))

        if 'raw_data' in os.listdir(folder):
            pass
        else:
            os.mkdir(os.path.join(folder, 'raw_data'))
            for file in os.listdir(folder):
                if file != 'raw_data':
                    src = os.path.join(folder, file)
                    dst = os.path.join(folder, 'raw_data', file)
                    shutil.move(src, dst)
            
        os.makedirs(os.path.join(folder, 'annotation'), exist_ok = True)
            
        if os.path.exists(os.path.join(folder, 'annotation', 'rotation_MM.txt')):
            src = os.path.join(folder, 'annotation', 'rotation_MM.txt')
            dst = os.path.join(folder, 'raw_data', 'rotation_MM.txt')
            shutil.copy(src, dst)
        
        for file in os.listdir(os.path.join(folder, 'raw_data')):
            src = os.path.join(folder, 'raw_data', file)
            dst = os.path.join(parameters['temp_folder'], folder.split('/')[-1], file)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy(src, dst)

        to_process_temp.append(os.path.join(parameters['temp_folder'], folder.split('/')[-1]))
        
    return links_folders, to_process_temp


####################################################################################################################################
####################################################################################################################################
############################################## 2. Mueller Matrix processing ########################################################
####################################################################################################################################
####################################################################################################################################

def save_file_as_npz(variable: dict, path: str, processing_mode = 'full'):   
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
               
def rotate_maps_90_deg(map_resize: np.ndarray, azimuth = False):
    """
    rotate_maps allows to rotate an array by 90 degree

    Parameters
    ----------
    map_resize : array
        the array that will be rotated
    idx_azimuth : boolean
        indicates if we are working with azimuth data (hence correction would be needed, default: False)
    
    Returns
    -------
    resized_rotated : array
        the rotated array
    """
    rotated = np.rot90(map_resize)[0:map_resize.shape[0], :]
    
    resized_rotated = np.zeros(map_resize.shape)
    for idx, x in enumerate(resized_rotated):
        for idy, y in enumerate(x):
            if idy < (map_resize.shape[1] - rotated.shape[0]) / 2 or idy > map_resize.shape[1] - (map_resize.shape[1] - rotated.shape[0]) / 2:
                pass
            else:
                try:
                    if azimuth:
                        resized_rotated[idx, idy] = ((rotated[idx, int(idy - ((map_resize.shape[1] - rotated.shape[0]) / 2 - 1))] + 90) % 180)
                    else:
                        resized_rotated[idx, idy] = rotated[idx, int(idy - ((map_resize.shape[1] - rotated.shape[0]) / 2 - 1))]
                except:
                    pass

    return resized_rotated               
                      
def rotate_parameter(parameter, angle_correction, MM_new = None):
    if MM_new is None:
        value = parameter
    else:
        value = MM_new[parameter]
        
    if angle_correction == 90:
        value_rotated = rotate_maps_90_deg(value)
    elif angle_correction == 180:
        value_rotated = value[::-1,::-1]
    else:
        if not MM_new is None:
            if parameter == 'Msk':
                rotated = ndimage.rotate(value.astype(float), angle = angle_correction, reshape = False)
                value_rotated = rotated > 0.5
        else:
            rotated = ndimage.rotate(value, angle = angle_correction, reshape = False)
            value_rotated = rotated
            
    return value_rotated
      
####################################################################################################################################
####################################################################################################################################
################################################## 3. Visualization lines ##########################################################
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
    vmin = parameters_plot['cbar_min']
    vmax = parameters_plot['cbar_max']

    # create and normalize the colormap
    cmap = clr.LinearSegmentedColormap.from_list(cmap_name, colors, n_bins)
    norm = clr.Normalize(vmin, vmax)
    
    return cmap, norm

####################################################################################################################################
####################################################################################################################################
################################################ X. File loader functions ##########################################################
####################################################################################################################################
####################################################################################################################################


def load_wavelengths():
    """
    load and returns the wavelengths usable by the IMP

    Returns
    -------
    wavelenghts : list
        the wavelengths usable by the IMP
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'wavelengths.txt')) as f:
        lines = f.readlines()
    f.close()
    for idx, l in enumerate(lines):
        lines[idx] = l.replace('\n', '') + 'nm'
    return lines

def get_wavelength(fname: str):
    """
    returns the wavelength given a specific folder name

    Parameters
    ----------
    fname : str
        the folder name for which we want the wavelength for

    Returns
    -------
    wavelength : int
        the wavelength corresponding to the folder given as an input
    """
    wavelength = int(str.split(fname, '/')[-1].split('nm')[0])
    return wavelength

def load_filenames(processing_mode = 'full', save_pdf_figs = False):
    """
    load and returns the name of the files that will be generated during the processing

    Returns
    -------
    filenames : list
        the list of the files generated during the processing
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_filenames = os.path.join(dir_path, 'data', 'filenames.json')
    
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
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'parameters_plot.json')) as json_file:
        data = json.load(json_file)
    return data

def chunks(lst: list, n: int):
    """
    chunks returns chunks of data composed of n element

    Parameters
    ----------
    generator : generator
        a generator allowing to get the chunks
        
    Returns
    -------
    lst : list
        the data to split in different chunks
    n : int
        the number of elements to be put in each chunk
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

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

def load_parameters_visualization():
    """
    load and returns the wavelengths usable by the IMP

    Returns
    -------
    data : dict
        the parameters used for the parameters of the visualization
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'parameters_visualizations.json')) as json_file:
        data = json.load(json_file)
    return data

def load_filenames_results(save_pdf_figs):
    """
    load and returns the name of the files that will be generated during the processing

    Returns
    -------
    filenames : list
        the list of the files generated during the processing
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'filenames_results.txt')) as f:
        lines = f.readlines()
    f.close()
    
    for idx, l in enumerate(lines):
        lines[idx] = l.replace('\n', '')
    
    for line in lines:
        if '.pdf' in line and not save_pdf_figs:
            lines.remove(line)
            
    return lines

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
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if viz:
        fname = 'name_combined_viz.txt'
    else:
        fname = 'name_combined.txt'

    with open(os.path.join(dir_path, 'data', fname)) as f:
        lines = f.readlines()[0]
    f.close()
    return lines

def load_filenames_combined_plot(viz: bool = False):
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
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if viz:
        fname = 'figures_names_combined_viz.txt'
    else:
        fname = 'figures_names_combined.txt'

    with open(os.path.join(dir_path, 'data', fname)) as f:
        lines = f.readlines()
    f.close()
    for idx, l in enumerate(lines):
        lines[idx] = l.replace('\n', '')
    return lines


def load_parameter_maps():
    """
    load and returns the parameters for the histogram plots

    Returns
    -------
    parameters_map : dict
        the parameters to plot the parameters histograms
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'parameters_map.json')) as json_file:
        data = json.load(json_file)
    return data


def load_parameter_names():
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
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fname = 'parameter_names.txt'

    with open(os.path.join(dir_path, 'data', fname)) as f:
        lines = f.readlines()
    f.close()
    for idx, l in enumerate(lines):
        lines[idx] = l.replace('\n', '')
    return lines