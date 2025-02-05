import os

import time

import numpy as np
from numpy import pi

from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as clr
import pickle
import copy

import torch
import torch.nn.functional as F

from processingmm.utils import rotate_parameter

from numba import jit

def get_and_plots_stds(measurements: list, sq_size: int = 4, azimuth: np.array = None, MM_computation: bool = False, angle_correction = 0, save_pdf_figs = False,
                       processing_mode = 'full'):
    """
    get_and_plots_stds is the master function to create the plots to visualize the azimuth noise

    Parameters
    ----------
    measurements : list
        the list of the path to the measurements folders

    Returns
    -------
    sq_size : int
        the size of the square to compute the azimuth std (default is 4)
    """
    azimuth_stds = {}

    # iterate over the measurements
    for folder in measurements:
        
        if azimuth is None:
            # load the azimuth
            path_polarimetry = os.path.join(folder, 'polarimetry', '550nm')
            azimuth = np.load(os.path.join(path_polarimetry, 'MM.npz'))['azimuth']
        else:
            pass
        
        start_azi_std_processing = time.time()
        
        # implementation using jit - improvement of a factor 7
        # result = apply_circular_std(azimuth * 2 * pi / 180)
        result = apply_circular_std(azimuth * 2 * pi / 180)
        azimuth_std = result * 180 / (2 * pi)
                
        if angle_correction != 0:
            azimuth_std = rotate_parameter(azimuth_std, angle_correction)
            
        end = time.time()
        
        # remove the std > 40 for visualization purposes
        azimuth_stds[folder] = copy.deepcopy(azimuth_std)
        azimuth_std[azimuth_std > 40] = 40

        # load the GM/WM mask and plot the azimuth noise
        if os.path.exists(os.path.join(folder, 'histology', 'labels_augmented_GM_WM_masked.png')):
            healthy = False
            mask = np.asarray(Image.open(os.path.join(folder, 'histology', 'labels_augmented_GM_WM_masked.png')))
        elif os.path.exists(os.path.join(folder, 'annotation', 'merged.png')):
            healthy = True
            mask = np.asarray(Image.open(os.path.join(folder, 'annotation', 'merged.png')))
        else:
            healthy = False
            mask = None
            
        if processing_mode in {'default', 'full'}:
            plot_azimuth_noise(azimuth_std, folder, mask = mask, healthy = healthy, plot = MM_computation, save_pdf_figs = save_pdf_figs) 
        
        plt.close()
        
    return azimuth_stds, end - start_azi_std_processing

"""@jit(nopython=True)
def circular_mean(angles):
    sin_sum = 0.0
    cos_sum = 0.0
    for angle in angles:
        sin_sum += np.sin(angle)
        cos_sum += np.cos(angle)
    return np.arctan2(sin_sum, cos_sum)

@jit(nopython=True)
def circular_standard_deviation(angles):


    mean_angle = circular_mean(angles)
    sin_sum = 0.0
    cos_sum = 0.0
    for angle in angles:
        sin_sum += np.sin(angle - mean_angle)
        cos_sum += np.cos(angle - mean_angle)
    R = np.sqrt(sin_sum**2 + cos_sum**2) / len(angles)
    circ_std_dev = np.sqrt(-2 * np.log(R))
    return circ_std_dev

@jit(nopython=True)
def apply_circular_std_ori(arr):


    rows, cols = arr.shape
    result = np.zeros_like(arr)
    for i in range(rows):
        for j in range(cols):

            window = []
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        window.append(arr[ni, nj])
            window = np.array(window)
            result[i, j] = circular_standard_deviation(window)
    return result"""


def circular_standard_deviation(angles):
    """
    Calculate the circular standard deviation for a set of angles in radians.
    """
    mean_angle = torch.atan2(torch.mean(torch.sin(angles), dim=-1), torch.mean(torch.cos(angles), dim=-1))
    sin_sum = torch.sum(torch.sin(angles - mean_angle.unsqueeze(-1)), dim=-1)
    cos_sum = torch.sum(torch.cos(angles - mean_angle.unsqueeze(-1)), dim=-1)
    R = torch.sqrt(sin_sum**2 + cos_sum**2) / angles.shape[-1]
    circ_std_dev = torch.sqrt(-2 * torch.log(R))
    return circ_std_dev

def apply_circular_std(arr):
    """
    Apply the circular standard deviation function over a 2D array using convolution.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arr = torch.tensor(arr, dtype=torch.float32).to(device)
    kernel_size = 5
    padding = kernel_size // 2

    # Pad the array and unfold to get sliding windows
    padded_arr = F.pad(arr.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='constant', value=0)
    unfolded = F.unfold(padded_arr, kernel_size=(kernel_size, kernel_size))
    unfolded = unfolded.view(kernel_size * kernel_size, -1).transpose(0, 1)

    # Apply circular_standard_deviation to each window
    result = circular_standard_deviation(unfolded)

    return result.view(arr.shape).cpu().numpy()


def plot_azimuth_noise(azimuth_std: np.array, folder: str, mask: np.array, healthy: bool = False, plot: bool = False, save_pdf_figs = False):
    """
    plot_azimuth_noise is the function to plot the azimuth noise

    Parameters
    ----------
    azimuth_std : np.array
        the azimuth circular standard deviation matrix
    folder : str
        the path to the measurement folder
    mask : np.array
        the mask of the GM / WM
    healthy : bool
        whether the measurement is healthy tissue or not

    Returns
    -------
    """
    # get the colormaps for the legend
    n_bins = 200
    colors = [[0, 0, 0.5], [0, 0, 1], 
              [0, 0.5, 1], [0, 1, 1], 
              [0.5, 1, 0.5], [1, 1, 0], 
              [1, 0.5, 0], [1, 0, 0], 
              [0.5, 0, 0]]
    # create and normalize the colormap
    cmap_colorbar = clr.LinearSegmentedColormap.from_list('azimuth_std_colorbar', colors, n_bins)
    norm_colorbar = clr.Normalize(0, 40)
    
    if mask is None:
        pass
    else:
        # get the borders between WM and GM
        edges = get_edges(mask, healthy = healthy)
    
        # change the values of the azimuth std to visualize the borders and remove the background
        for idx, x in enumerate(mask):
            for idy, y in enumerate(x):
                if healthy:
                    if y == 0:
                        azimuth_std[idx, idy] = -5
                else:
                    if sum(y) == 0:
                        azimuth_std[idx, idy] = -5
        
        for idx, x in enumerate(edges):
            for idy, y in enumerate(x):
                if y != 0:
                    azimuth_std[idx, idy] = 45
    
    # plot the azimuth std
    fig, ax = plt.subplots(figsize = (15, 11))
    ax.imshow(azimuth_std, norm=norm_colorbar, cmap=cmap_colorbar)
    # format the color bar
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm_colorbar, cmap=cmap_colorbar), pad = 0.02, 
                        ticks=np.arange(0, 41, 10), fraction=0.06, ax=ax)
    cbar.ax.set_yticklabels(["{:.0f}".format(a) for a in np.arange(0, 41, 10)], 
                        fontsize=40, weight='bold')

    # add a title and save the plot
    title = 'Azimuth local variability'
    plt.title(title, fontsize=35, fontweight="bold", pad=14)
    plt.xticks([])
    plt.yticks([])
        
    plt.savefig(os.path.join(folder, 'Azimuth_local_variability.png'), dpi = 100)
    if save_pdf_figs:
        plt.savefig(os.path.join(folder, 'Azimuth_local_variability.pdf'), dpi = 80)
        
    path_save_img = os.path.join(folder, 'Azimuth_local_variability_img.png')
    plt.imsave(path_save_img, azimuth_std, cmap = cmap_colorbar, vmin = 0, vmax = 40)

    
    
def get_edges(mask: np.array, healthy: bool = False):
    """
    get_edges finds the border between the WM and the GM

    Parameters
    ----------
    mask : np.array
        the mask of the GM / WM
    healthy : bool
        whether the measurement is healthy tissue or not

    Returns
    -------
    img_dilation : np.array
        the dilated edges
    """
    if healthy:
        edges = cv2.Canny(mask, threshold1=30, threshold2=100)
    else:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=30, threshold2=100)

    # Taking a matrix of size 5 as the kernel and dilate the border
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(edges, kernel, iterations=1)

    return img_dilation
