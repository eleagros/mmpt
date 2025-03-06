import os

import time

import numpy as np
from numpy import pi

from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as clr
import copy

import torch
import torch.nn.functional as F

def get_and_plots_stds(folder: str, sq_size: int, wavelength: str, azimuth: np.array = None, save_pdf_figs = False, processing_mode = 'full'):
    """
    get_and_plots_stds is the master function to create the plots to visualize the azimuth noise

    Parameters
    ----------
    measurements : list
        the list of the path to the measurements folders

    Returns
    -------
    azimuth_std : np.array
        the azimuth circular standard deviation matrix
    time : float
        the time to process the azimuth noise
    """
    # iterate over the measurements        
    if azimuth is None:
        # load the azimuth
        path_polarimetry = os.path.join(folder, 'polarimetry', '550nm')
        azimuth = np.load(os.path.join(path_polarimetry, 'MM.npz'))['azimuth']
    else:
        pass
        
    start_azi_std_processing = time.time()
        
    # implementation using jit - improvement of a factor 7
    # result = apply_circular_std(azimuth * 2 * pi / 180)
    result = apply_circular_std(azimuth * 2 * pi / 180, sq_size)
    azimuth_std = result * 180 / (2 * pi)
                
    # if angle_correction != 0:
    #    azimuth_std = rotate_parameter(azimuth_std, angle_correction)
            
    end = time.time()
        
    return azimuth_std, end - start_azi_std_processing

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

def apply_circular_std(arr, kernel_size: int):
    """
    Apply the circular standard deviation function over a 2D array using convolution.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arr = torch.tensor(arr, dtype=torch.float32).to(device)
    padding = kernel_size // 2

    # Pad the array and unfold to get sliding windows
    padded_arr = F.pad(arr.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='constant', value=0)
    unfolded = F.unfold(padded_arr, kernel_size=(kernel_size, kernel_size))
    unfolded = unfolded.view(kernel_size * kernel_size, -1).transpose(0, 1)

    # Apply circular_standard_deviation to each window
    result = circular_standard_deviation(unfolded)
    
    # Reshape the result to match the input array shape
    result = result.view(arr.shape).cpu().numpy()

    # Set the borders (edges) of the result to NaN
    result[:padding, :] = np.nan
    result[-padding:, :] = np.nan
    result[:, :padding] = np.nan
    result[:, -padding:] = np.nan

    return result