import numpy as np
import pickle
import os
import matplotlib.colors as clr

def get_wavelength(fname):
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
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, '../../data', 'filenames.pickle'), 'rb') as handle:
        filenames = pickle.load(handle)
    return filenames

def load_parameter_maps():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, '../../data', 'parameters_map.pickle'), 'rb') as handle:
        parameters_map = pickle.load(handle)
    return parameters_map


def load_plot_parameters():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, '../../data', 'parameters_plot.pickle'), 'rb') as handle:
        return pickle.load(handle)

def get_cmap(parameter):
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
    parameters_plot = load_plot_parameters()[parameter]
    colors = parameters_plot['colors']
    n_bins = parameters_plot['n_bins']
    cmap_name = parameters_plot['cmap_name']
    vmin = parameters_plot['cbar_min']
    vmax = parameters_plot['cbar_max']

    cmap = clr.LinearSegmentedColormap.from_list(cmap_name, colors, n_bins)
    norm = clr.Normalize(vmin, vmax)
    
    return cmap, norm

def load_MM(path):
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
    return row['folder name']

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]