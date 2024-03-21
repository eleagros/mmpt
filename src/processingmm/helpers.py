import numpy as np
import os
import matplotlib.colors as clr
import json
import ast


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
    wavelength = int(str.split(fname, '\\')[-1].split('nm')[0])
    return wavelength


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


def save_list_as_txt(lst: list, path: str):
    with open(path, 'w') as fp:
        for item in lst:
            fp.write("%s\n" % item)

def save_dict_as_json(dictionnary: dict, path: str):
    with open(path, "w") as file:
        json.dump(dictionnary, file)


def load_filenames_results():
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
    return lines

def load_filenames_raw_data():
    """
    load and returns the name of the files that are generated during the acquisition of the data

    Returns
    -------
    filenames : list
        the list of the files generated during the processing
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'filenames_raw_data.txt')) as f:
        lines = f.readlines()
    f.close()
    for idx, l in enumerate(lines):
        lines[idx] = l.replace('\n', '')
    return lines


def load_filenames_50x50():
    """
    load and returns the name of the files that will be generated during the processing

    Returns
    -------
    filenames : list
        the list of the files generated during the processing
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'filenames_50x50.txt')) as f:
        lines = f.readlines()
    f.close()
    for idx, l in enumerate(lines):
        lines[idx] = l.replace('\n', '')
    return lines


def load_filenames():
    """
    load and returns the name of the files that will be generated during the processing

    Returns
    -------
    filenames : list
        the list of the files generated during the processing
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'filenames.txt')) as f:
        lines = f.readlines()
    f.close()
    for idx, l in enumerate(lines):
        lines[idx] = l.replace('\n', '')
    return lines


def load_path_win7():
    """
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'path_dict_path.txt')) as f:
        lines = f.readlines()
    f.close()
    fname = lines[0]
    with open(fname) as json_file:
        data = json.load(json_file)
    return data

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


def add_path(row):
    """
    add_path

    Parameters
    ----------
    row : pandas.dataframe row
        
    Returns
    -------
    folder_name : str
        the path of the folder in a certain row in a dataframe
    """
    return row['folder name']


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
    np.savez_compressed(path, 
                        Intensity = variable['Intensity'],
                        M11 = variable['M11'],
                        Msk = variable['Msk'],
                        totD = variable['totD'],
                        linR = variable['linR'],
                        azimuth = variable['azimuth'],
                        totP = variable['totP'],
                        azimuth_std = variable['azimuth_std'])
    

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
    try:
        # data_exist = len(os.listdir(path)) == 2 or len(os.listdir(path)) == 3
        data_exist = len(os.listdir(path)) == 2
    except FileNotFoundError:
        data_exist = False
    return data_exist


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