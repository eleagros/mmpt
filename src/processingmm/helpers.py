import numpy as np
import pickle
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


def load_filenames():
    """
    load and returns the name of the files that will be generated during the processing

    Returns
    -------
    filenames : list
        the list of the files generated during the processing
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, '../../data', 'filenames.pickle'), 'rb') as handle:
        filenames = pickle.load(handle)
    return filenames


def load_parameter_maps():
    """
    load and returns the parameters for the histogram plots

    Returns
    -------
    parameters_map : dict
        the parameters to plot the parameters histograms
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, '../../data', 'parameters_map.pickle'), 'rb') as handle:
        parameters_map = pickle.load(handle)
    return parameters_map


def load_plot_parameters():
    """
    load and returns the parameters for the polarimetric parameter plots

    Returns
    -------
    plot_parameters : dict
        the parameters to plot the parameters maps
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, '../../data', 'parameters_plot.pickle'), 'rb') as handle:
        return pickle.load(handle)
    

def load_parameters_visualization():
    """
    load and returns the wavelengths usable by the IMP

    Returns
    -------
    data : dict
        the parameters used for the parameters of the visualization
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, '../../data', 'parameters_visualizations.json')) as json_file:
        data = json.load(json_file)
    return ast.literal_eval(data)


def load_wavelengths():
    """
    load and returns the wavelengths usable by the IMP

    Returns
    -------
    wavelenghts : list
        the wavelengths usable by the IMP
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, '../../data', 'wavelengths.txt')) as f:
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