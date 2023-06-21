import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import math
import cv2

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cmocean

from skimage.restoration import unwrap_phase
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import binary_dilation

from processingmm.helpers import get_cmap, get_wavelength, load_MM, load_parameters_visualization, load_filenames_results, load_wavelengths, is_there_data, load_combined_plot_name, load_filenames_combined_plot
from processingmm.multi_img_processing import remove_already_computed_directories
from processingmm.libmpMuelMat import _isNumStable


def visualization_auto(measurements_directory: str, parameters_set: str, batch_processing = False, to_compute = None, run_all = True):
    """
    master function calling line_visualization_routine for each of the folders in the measurements_directory
    
    Parameters
    ----------
    measurements_directory : str
        the path to the directory containing the measurements
    parameters_set : str
        the name of the set of parameters that should be used (i.e. 'CUSA')
    batch_processing : boolean
        indicates if we are batch processing the data (and if we should print progress bars, default = False)
    to_compute : list
        the list of folders to compute (if one wants to process only specific ones, default = None)
    run_all : boolean
        defines wether all the folders should be processed (default = True)
    """
    if to_compute:
        pass
    else:
        if run_all:
            to_compute = os.listdir(measurements_directory)
        else:
            # get the folders to compute if not given as an input
            to_compute = remove_computed_folders_viz(measurements_directory, run_all = run_all)

    if not batch_processing:
        for c in tqdm(to_compute):
            path = os.path.join(measurements_directory, c)
            perform_visualisation(path, parameters_set, run_all = run_all)
    else:
        for c in to_compute:
            path = measurements_directory + c
            path = os.path.join(measurements_directory, c)
            perform_visualisation(path, parameters_set, run_all = run_all)


def remove_computed_folders_viz(measurements_directory, run_all: bool = False):
    """
    removes the folders for which the visualization was already obtained
    
    Parameters
    ----------
    measurements_directory : str
        the path to the directory containing the measurements

    Returns
    -------
    to_compute : list
        the list of the folders that need to be computed
    """
    filename_results = load_filenames_results()
    folders = os.listdir(measurements_directory)
    to_compute = []

    for c in folders:
        visualized = True

        path = os.path.join(measurements_directory, c)
        wavelengths = load_wavelengths()
        check_wl = []
        for wl in wavelengths:
            if is_there_data(os.path.join(path, 'raw_data', wl)):
                check_wl.append(wl)

        for wl in check_wl:
            path_wl = os.path.join(path, 'polarimetry', wl)
            if os.path.isdir(os.path.join(path_wl, 'results')) and not run_all:
                for file in filename_results:
                    if file in os.listdir(os.path.join(path_wl, 'results')) :
                        pass
                    else:
                        visualized = False
            else:
                visualized = False

        if visualized:
            pass
        else:
            to_compute.append(c)

    return to_compute


def perform_visualisation(path: str, parameters_set: str, run_all: bool = False):
    """
    master function running the visualization script for one folder
    
    Parameters
    ----------
    path : str
        the path to the folder on which the routine should be run
    parameters_set : str
        the name of the set of parameters that should be used (i.e. 'CUSA')
    """
    mask = get_mask(path)
    # remove_already_computed_directories with sanity = True puts all the wavelengths to be processed
    directories = remove_already_computed_directories(path, sanity = True, run_all = run_all)

    for d_ in directories:
        d = d_.replace('raw_data', 'polarimetry')
        # get wavelength and call line_visualization_routine
        wavelength = get_wavelength(d)
        line_visualization(os.path.join(d, 'MM.npz'), mask, parameters_set)


def get_mask(path: str):
    """
    get_mask returns a manually created mask if has been created
    
    Parameters
    ----------
    path : str
        the path to the folder of interest

    Returns
    -------
    mask : 
        the manually created mask, if existing
    """
    if 'mask-viz.tif' in os.listdir(os.path.join(path, 'annotation')):
        mask = np.array(Image.open(os.path.join(path, 'annotation', 'mask-viz.tif')))
        mask = mask != 0
    elif 'merged.jpeg' in os.listdir(os.path.join(path, 'annotation')):
        mask = np.array(Image.open(os.path.join(path, 'annotation', 'merged.jpeg')))
        mask = mask != 128
        plt.imshow(mask)
    else:
        mask = None
    return mask


def line_visualization(path: str, mask: np.ndarray, parameters_set : str, title: str = 'Azimuth of the optical axis (°)'):
    """
    apply the visualization routine to a folder (i.e. load MM, plot M11, generate the masks, get the arrows and plot them)

    Parameters
    ----------
    path : str
        the path of the folder of interest
    mask : np.ndarray
        the manual mask, if it has been created
    parameters_set : str
        the name of the set of parameters that should be used (i.e. 'CUSA')
    title : str
        the title of the graph
    """
    parameters_visualizations = load_parameters_visualization()

    # 1. deplarization and retardance are higher than the tresholds
    # remove the points for which depolarization < depolarization_parameter and linear_retardance < linear_retardance_parameter
    depolarization_parameter = parameters_visualizations[parameters_set]['depolarization']
    linear_retardance_parameter = parameters_visualizations[parameters_set]['linear_retardance']

    # 2. remove pixels that are too dark compared to white matter (intensity < 1/parameter * mean(image))
    grey_scale_parameter = parameters_visualizations[parameters_set]['greyscale']

    # 3. remove the "pixels" for which std > std_parameter
    # std_parameter = parameters_visualizations[parameters_set]['std_parameter']

    # 4. bigger n = more points incorporated in each "pixel"
    n = parameters_visualizations[parameters_set]['n']

    # 5. change the width of the rectangles - typically around 1.5
    widths = parameters_visualizations[parameters_set]['widths']

    # 6. change the scale of the arrows - typically around 20
    scale = parameters_visualizations[parameters_set]['scale']

    # load the MM and creates, if necessary, the results folder
    orientation = load_MM(path)

    path_results = '/'.join(path.split('\\')[:-1]) + '/'
    if os.path.exists(os.path.join(path_results, 'results')):
        if os.path.isdir(os.path.join(path_results, 'results')):
            pass
        else:
            raise NameError('Should not happen.')
    else:
        os.mkdir(os.path.join(path_results, 'results'))
    path_results = os.path.join(path_results, 'results')

    # load the first element of the Mueller Matrix for each pixel - the M11 element scales the input intensity to the output 
    # intensity so this element can be interpreted as the simple transmittance
    M11 = orientation['M11']
    # lower_M11 = np.mean(M11) - 2*np.std(M11)
    # upper_M11 = np.mean(M11) + 2*np.std(M11)

    # get the depolarization and linear retardance value for each pixel of the original image
    depolarization = orientation['totP']
    retardance = orientation['linR']

    assert _isNumStable(orientation['azimuth'])
    # get the azimuth and convert angles to radian
    azimuth = np.pi*orientation['azimuth']/360
    
    # phase unwrap or unwrap is a process often used to reconstruct a signal's original phase
    azimuth_unwrapped = unwrap_phase(4*azimuth)/2
    
    new_mask = np.logical_not(np.logical_or(np.logical_and(depolarization<depolarization_parameter, 
                                gaussian_filter(orientation['linR'],4)<linear_retardance_parameter), 
                                binary_dilation(M11<(1/grey_scale_parameter*np.mean(M11)), iterations = 3)))
    new_mask_M11 = np.logical_not(binary_dilation(M11<(1/grey_scale_parameter*np.mean(M11)), iterations = 3))
        
    plot_azimuth(retardance, M11, azimuth_unwrapped, n, widths, scale, title, path_results,
                 new_mask_M11, mask, normalized = False)
    plot_azimuth(retardance, M11, azimuth_unwrapped, n, widths, scale, title, path_results,
                 new_mask, mask, normalized = True)
    

def plot_azimuth(retardance: np.ndarray, M11: np.ndarray, azimuth_unwrapped: np.ndarray, n: int, widths: int, length: int, title: str, 
                 path_results: str, new_mask: np.ndarray, mask: np.ndarray, normalized = False):
    """
    apply the visualization routine to a folder (i.e. load MM, plot M11, generate the masks, get the arrows and plot them)

    Parameters
    ----------
    retardance : np.ndarray
        the polarimetric map of the retardance
    M11 : np.ndarray
        the grayscale intensity image
    azimuth_unwrapped : np.ndarray
        the map of the unwrapped azimuth
    n : int
        the number of oints incorporated in each "pixel"
    widths : int
        change the width of the rectangles
    length : int
        change the scale of the arrows
    title : str
        the title of the plot
    path_results : str
        the path in which to save the plots generated
    new_mask : np.ndarray
        the mask used to decided where to plot and where to mask
    mask : np.ndarray
        the manual mask, if it has been created
    """
    image_size = M11.shape
    cmap_azi, norm = get_cmap(parameter = 'azimuth')

    if type(mask) == np.ndarray:
        new_mask = np.logical_and(mask, new_mask)
    else:
        pass
        
    if normalized:
        # plot the mask image
        fig = plt.figure(figsize = (15, 10))
        plt.imshow(new_mask, interpolation='none')
        plt.xticks([])
        plt.yticks([])

        plt.savefig(os.path.join(path_results, 'mask.pdf'))
        plt.savefig(os.path.join(path_results, 'mask.png'))
        plt.close(fig)
    
    X, Y = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))
        
    orientation_cos = np.cos(azimuth_unwrapped)
    orientation_sin = np.sin(azimuth_unwrapped)
    
    u = (retardance*orientation_cos)
    v = (retardance*orientation_sin)
    u_plot, v_plot = get_plot_arrows(u, v, new_mask, n)
    u_normalized, v_normalized = get_u_v(u_plot, v_plot, normalized = normalized)
    
    # plot the lines on the figure
    fig, ax = plt.subplots(figsize = (15, 10))
    ax.imshow(M11, cmap = 'gray')
    
    all_norms = []
    for idy, _ in enumerate(X[::n,::n]):
        for idx, _ in enumerate(Y[::n,::n][0]):
            u = u_normalized[idy, idx]
            v = v_normalized[idy, idx]
            all_norms.append(np.linalg.norm([u,v]))
    
    for idy, _ in enumerate(X[::n,::n]):
        for idx, _ in enumerate(Y[::n,::n][0]):
            idx_scale = idx * n
            idy_scale = idy * n
            u = u_normalized[idy, idx]
            v = v_normalized[idy, idx]
            
            norm_vector = np.linalg.norm([u,v])
            
            angle = np.arctan(u/v)*180/math.pi
            if angle < 0 :
                angle += 180
            if angle > 180:
                angle -= 180

            angle_color = np.arctan(v/u)*180/math.pi
            if angle_color < 0 :
                angle_color += 180
            if angle_color > 180:
                angle_color -= 180

            if not math.isnan(u):
                color = cmap_azi(norm(angle_color))
                if normalized:
                    ax.add_patch(Rectangle((idx_scale, idy_scale), widths, 
                                           length, 
                                           angle = angle, facecolor = color, edgecolor = color))
                else:
                    ax.add_patch(Rectangle((idx_scale, idy_scale), 
                                           widths * function_sig_fixed(norm_vector), 
                                           length * function_sig_fixed(norm_vector), 
                                           angle = angle, facecolor = color, edgecolor = color))
                    
    plt.xticks([])
    plt.yticks([])
    plt.title(title, fontsize=40, fontweight="bold", pad=20)
    
    cbar_max = 180
    cbar_min = 0
    cbar_step = 30
    formatter = '{:.0f}'
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_azi), pad = 0.01, 
                        ticks=np.arange(cbar_min, cbar_max + cbar_step, cbar_step), fraction=0.06)
    cbar.ax.set_yticklabels([formatter.format(a) for a in np.arange(cbar_min, cbar_max+cbar_step, cbar_step)], 
                            fontsize=30, weight='bold')
        
    if normalized:
        plt.savefig(os.path.join(path_results, 'CUSA_fig.pdf'))
        plt.savefig(os.path.join(path_results, 'CUSA_fig.png'))
    else:
        plt.savefig(os.path.join(path_results, 'CUSA_fig_weighted.pdf'))
        plt.savefig(os.path.join(path_results, 'CUSA_fig_weighted.png'))
    
    plt.axis('off')
    plt.title('')
    cbar.remove()
    if normalized:
        fig.savefig(os.path.join(path_results, 'CUSA_fig_image.png'), bbox_inches='tight',transparent=True, pad_inches=0)
    else:
        fig.savefig(os.path.join(path_results, 'CUSA_fig_image_weighted.png'), bbox_inches='tight',transparent=True, pad_inches=0)
    plt.close(fig)


def func(x: float, a: float, b: float, c: float, d: float, e: float):
    """
    the sigmoid function

    Parameters
    ----------
    x, a, b, c, d, e: float
        the sigmoid parameters
        
    Returns
    -------
    fun : 
        the sigmoid function
    """
    return a + (b - c) / (1 + pow(x/d, e))

def function_sig_fixed(x: float):
    """
    the sigmoid function for fixed tissue

    Parameters
    ----------
    x : float
        the value to evaluate the function for
        
    Returns
    -------
    f(x) : float
        the value of the sigmoid function at the point x
    """
    popt = [1.00176023e+00, 7.50232796e+03, 7.50330844e+03, 2.95781650e+00,
       3.88876080e+00]
    return func(x, *popt)


def get_u_v(u_plot: np.ndarray, v_plot: np.ndarray, normalized: bool = True):
    """
    obtain the arrows that will be used for plotting

    Parameters
    ----------
    u_plot : numpy.ndarray
        the x component of the azimuth data to be plotted
    v_plot : numpy.ndarray
        the y component of the azimuth data to be plotted
        
    Returns
    -------
    u_normalized : numpy.ndarray
        the normalized x component of the azimuth data to be plotted
    v_normalized : numpy.ndarray
        the normalized y component of the azimuth data to be plotted
    """
    u_normalized = []
    v_normalized = []

    for idx_x, x in enumerate(u_plot):

        u_plot_line = []
        v_plot_line = []

        for idx_y, y in enumerate(x):

            # normalize the lines
            if normalized:
                norm = np.sqrt(u_plot[idx_x][idx_y]*u_plot[idx_x][idx_y] + v_plot[idx_x][idx_y]*v_plot[idx_x][idx_y])
                u_plot_line.append(u_plot[idx_x][idx_y]/norm)
                v_plot_line.append(v_plot[idx_x][idx_y]/norm)
            else:
                u_plot_line.append(u_plot[idx_x][idx_y])
                v_plot_line.append(v_plot[idx_x][idx_y])

        u_normalized.append(u_plot_line)
        v_normalized.append(v_plot_line)

    u_normalized = np.array(u_normalized)
    v_normalized = np.array(v_normalized)
    
    return u_normalized, v_normalized


def get_plot_arrows(u: np.ndarray, v: np.ndarray, new_mask: np.ndarray, n: int):
    """
    obtain the arrows that will be used for plotting

    Parameters
    ----------
    u : numpy.ndarray
        unwrapped x component of azimuth
    v : numpy.ndarray
        unwrapped y component of azimuth
    new_mask : numpy.ndarray
        mask to be used for selecting pixels of interest
    n : int
        bigger n = more points incorporated in each "pixel"
        
    Returns
    -------
    u_plot : numpy.ndarray
        the x component of the azimuth data to be plotted
    v_plot : numpy.ndarray
        the y component of the azimuth data to be plotted
    """
    u_plot = []
    v_plot = []

    u_std = []
    v_std = []

    # iterate over each row...
    for idx_x, x in enumerate(u):

        u_plot_line = []
        v_plot_line = []

        # and each column...
        for idx_y, y in enumerate(x):
            if idx_x%n == 0 and idx_y%n == 0:
                min_idx_x = max(0, idx_x - n)
                max_idx_x = min(len(u), idx_x + n)

                min_idx_y = max(0, idx_y - n)
                max_idx_y = min(len(v[0]), idx_y + n)

                # get the data for each sub-area
                mask_sq = new_mask[min_idx_x:max_idx_x, min_idx_y:max_idx_y]
                if sum(sum(mask_sq))/(len(mask_sq)*len(mask_sq[0])) > 0.30:
                    u_std.append(np.std(u[min_idx_x:max_idx_x, min_idx_y:max_idx_y]))
                    v_std.append(np.std(v[min_idx_x:max_idx_x, min_idx_y:max_idx_y]))

                    u_plot_line.append(np.mean(u[min_idx_x:max_idx_x, min_idx_y:max_idx_y]))
                    v_plot_line.append(np.mean(v[min_idx_x:max_idx_x, min_idx_y:max_idx_y]))
                else:
                    u_plot_line.append(math.nan)
                    v_plot_line.append(math.nan)

        if u_plot_line:
            u_plot.append(u_plot_line)
            v_plot.append(v_plot_line)

    u_plot = np.array(u_plot)
    v_plot = np.array(v_plot)

    return u_plot, v_plot


def save_batch(folder: str):
    """
    combine the 4 images (linear retardance, depolarization, azimuth and intensity in a single file)

    Parameters
    ----------
    folder : str
        the folder in which to find the images
    figures : list of str
        the names of the figures (i.e. 'Depolarization.png', 'Linear Retardance.png'...)
    names : str
        the name of the new file to be created
    """
    names = load_combined_plot_name(viz = True)
    figures = load_filenames_combined_plot(viz = True)

    # load the four images
    img_3 = cv2.imread(folder + '/' + figures[0])[40:960, 160:1450]
    img_2 = cv2.imread(folder + '/' + figures[1])[40:960, 160:1450]
    img_1 = cv2.imread(folder + '/' + figures[2])[40:960, 160:1450]
    img_4 = cv2.imread(folder + '/' + figures[3])[40:960, 160:1450]

    h1, w1 = img_1.shape[:2]
    h2, w2 = img_2.shape[:2]
    h3, w3 = img_3.shape[:2]
    h4, w4 = img_4.shape[:2]

    # combined the four images in a single image
    output = np.zeros((max(h1 + h3, h2 + h4), w1+w2,3), dtype=np.uint8)
    output[:,:] = (255,255,255)
    output[:h1, :w1, :3] = img_1
    output[:h2, w1:w1+w2, :3] = img_2
    output[h1:h1+h3, :w3, :3] = img_3
    output[h2:h2+h4, w2:w2+w4, :3] = img_4

    # save the new image
    cv2.imwrite(folder + '/' + names, output)