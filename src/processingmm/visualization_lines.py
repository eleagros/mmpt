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

from processingmm.helpers import get_cmap, get_wavelength, load_MM
from processingmm.multi_img_processing import remove_already_computed_directories


def visualization_auto(measurements_directory, parameters_visualizations, parameters_set, batch_processing = False,
                       to_compute = None, run_all = True):
    """
    master function calling line_visualization_routine for each of the folders in the measurements_directory
    
    Parameters
    ----------
    measurements_directory : str
        the path to the directory containing the measurements
    depolarization_parameter : double
        remove the points for which depolarization < depolarization_parameter and 
        linear_retardance < linear_retardance_parameter
    linear_retardance_parameter : double
        remove the points for which depolarization < depolarization_parameter and 
        linear_retardance < linear_retardance_parameter
    grey_scale_parameter : double
        remove pixels that are too dark compared to white matter (intensity < 1/parameter * mean(image))
    n : int
        bigger n = more points incorporated in each "pixel"
    scale : int
        change the scale of the arrows
    std_parameter : int
        remove the "pixels" for which azimuth std > std_parameter
    to_compute : list
        (optional) list of folders to compute visualization for
    """
    if to_compute:
        pass
    else:
        if run_all: 
            treshold = 1000
        else:
            treshold = 8
        # get the folders to compute if not given as an input
        to_compute = remove_computed_folders_viz(measurements_directory, treshold)


    if not batch_processing:
        for c in tqdm(to_compute):
            path = os.path.join(measurements_directory, c)
            perform_visualisation(path, parameters_visualizations, parameters_set)
    else:
        for c in to_compute:
            path = measurements_directory + c
            path = os.path.join(measurements_directory, c)
            perform_visualisation(path, parameters_visualizations, parameters_set)
                

def get_mask(path):
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

def perform_visualisation(path, parameters_visualizations, parameters_set):

        # 1. deplarization and retardance are higher than the tresholds
    # remove the points for which depolarization < depolarization_parameter and linear_retardance < linear_retardance_parameter
    depolarization_parameter = parameters_visualizations[parameters_set]['depolarization']
    linear_retardance_parameter = parameters_visualizations[parameters_set]['linear_retardance']

    # 2. remove pixels that are too dark compared to white matter (intensity < 1/parameter * mean(image))
    grey_scale_parameter = parameters_visualizations[parameters_set]['greyscale']

    # 3. remove the "pixels" for which std > std_parameter
    std_parameter = parameters_visualizations[parameters_set]['std_parameter']

    # 4. bigger n = more points incorporated in each "pixel"
    n = parameters_visualizations[parameters_set]['n']

    # 5. change the width of the rectangles - typically around 1.5
    widths = parameters_visualizations[parameters_set]['widths']

    # 6. change the scale of the arrows - typically around 20
    scale = parameters_visualizations[parameters_set]['scale']
    
    cmap, norm = get_cmap(parameter = 'azimuth')

    mask = get_mask(path)
    # remove_already_computed_directories with sanity = True puts all the wavelengths to be processed
    directories = remove_already_computed_directories(path, sanity = True)

    for d_ in directories:
        d = d_.replace('raw_data', 'polarimetry')
        # get wavelength and call line_visualization_routine
        wavelength = get_wavelength(d)
        line_visualization(os.path.join(d, 'MM.npz'), mask, grey_scale_parameter, linear_retardance_parameter, depolarization_parameter, n, scale, std_parameter,
                           widths, cmap, norm)


def line_visualization(path, mask, grey_scale_parameter, linear_retardance_parameter, depolarization_parameter, 
                               n, length, std_parameter, widths, cmap_azi, norm, title = 'Azimuth of the optical axis (°)'):
    """
    apply the visualization routine to a folder (i.e. load MM, plot M11, generate the masks, get the arrows and plot them)

    Parameters
    ----------
    path : str
        the path of the folder of interest
    grey_scale_parameter : double
        remove pixels that are too dark compared to white matter (intensity < 1/parameter * mean(image))
    linear_retardance_parameter : double
        remove the points for which depolarization < depolarization_parameter and 
        linear_retardance < linear_retardance_parameter
    depolarization_parameter : double
        remove the points for which depolarization < depolarization_parameter and 
        linear_retardance < linear_retardance_parameter
    n : int
        bigger n = more points incorporated in each "pixel"
    scale : int
        change the scale of the arrows
    std_parameter : int
        remove the "pixels" for which azimuth std > std_parameter
    """
    
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
    image_size = M11.shape
    lower_M11 = np.mean(M11) - 2*np.std(M11)
    upper_M11 = np.mean(M11) + 2*np.std(M11)

    # get the depolarization and linear retardance value for each pixel of the original image
    depolarization = orientation['totP']
    retardance = orientation['linR']

    # get the azimuth and convert angles to radian
    # azimuth = np.pi*orientation['Image_orientation_linear_retardance_full']/360
    azimuth = np.pi*orientation['azimuth']/360
    
    # phase unwrap or unwrap is a process often used to reconstruct a signal's original phase
    azimuth_unwrapped = unwrap_phase(4*azimuth)/2
    cycle_limits = (np.floor(np.min(azimuth_unwrapped/(np.pi))), np.ceil(np.max(azimuth_unwrapped/(np.pi))))
    num_cycles = cycle_limits[1]-cycle_limits[0]
    
    cut_points = np.linspace(0,1.0, int(num_cycles+1))
    intervals = zip(cut_points[:-1],cut_points[1:])

    # create the colormap for each interval of interest
    cmaps = [list(zip(np.linspace(x,y,128),cmocean.cm.phase(np.linspace(0,1.,128)))) for x,y in zip(cut_points[:-1],cut_points[1:])]
    new_cmap_values = [x for y in cmaps for x in y[:-1]]
    new_cmap_values.append(cmaps[-1][-1])
    interval_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('interval_cmap',new_cmap_values)
    
    
    new_mask = np.logical_not(np.logical_or(np.logical_and(depolarization<depolarization_parameter, 
                                gaussian_filter(orientation['linR'],4)<linear_retardance_parameter), 
                                binary_dilation(M11<(1/grey_scale_parameter*np.mean(M11)), iterations = 3)))
    new_mask_M11 = np.logical_not(binary_dilation(M11<(1/grey_scale_parameter*np.mean(M11)), iterations = 3))
        
    plot_azimuth(retardance, M11, azimuth_unwrapped, image_size, cmap_azi, norm, n, widths, length, title, path_results,
                 new_mask_M11, mask, normalized = False)
    plot_azimuth(retardance, M11, azimuth_unwrapped, image_size, cmap_azi, norm, n, widths, length, title, path_results,
                 new_mask, mask, normalized = True)
    

def plot_azimuth(retardance, M11, azimuth_unwrapped, image_size, cmap_azi, norm, n, widths, length, title, path_results,
                 new_mask, mask, normalized = False):

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
    # if normalized:
        # pass
    # else: 
        # new_mask = np.ones(retardance.shape)
        
    orientation_cos = np.cos(azimuth_unwrapped)
    orientation_sin = np.sin(azimuth_unwrapped)
    orientation_cos_depol_masked = np.ma.array(orientation_cos, mask = new_mask)
    orientation_sin_depol_masked = np.ma.array(orientation_sin, mask = new_mask)
    
    u = (retardance*orientation_cos)
    v = (retardance*orientation_sin)
    u_plot, v_plot = get_plot_arrows(u, v, new_mask, n)
    u_normalized, v_normalized = get_u_v(u_plot, v_plot, normalized = normalized)
    
    # plot the lines on the figure
    fig, ax = plt.subplots(figsize = (15, 10))
    ax.imshow(M11, cmap = 'gray')
    
    all_norms = []
    for idy, y in enumerate(X[::n,::n]):
        for idx, x in enumerate(Y[::n,::n][0]):
            u = u_normalized[idy, idx]
            v = v_normalized[idy, idx]
            all_norms.append(np.linalg.norm([u,v]))
    max_norm = np.median(all_norms)
    
    for idy, y in enumerate(X[::n,::n]):
        for idx, x in enumerate(Y[::n,::n][0]):
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


def func(x, a, b, c, d, e):
    return a + (b - c) / (1 + pow(x/d, e))

def function_sig_fixed(x):
    popt = [1.00176023e+00, 7.50232796e+03, 7.50330844e+03, 2.95781650e+00,
       3.88876080e+00]
    return func(x, *popt)


def get_u_v(u_plot, v_plot, normalized = True):
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


def get_plot_arrows(u, v, new_mask, n):
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
    std_parameter : int
        remove the "pixels" for which azimuth std > std_parameter
        
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


def remove_computed_folders_viz(measurements_directory, treshold = 8):
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
    folders = os.listdir(measurements_directory)
    folers_to_compute = []

    for c in folders:
        visualized = True

        path = os.path.join(measurements_directory, c)
        directories = remove_already_computed_directories(path, sanity = True)
        
        for d_ in directories:
            d = d_.replace('raw_data', 'polarimetry')
            wavelength = get_wavelength(d)
            if os.path.exists(os.path.join(d, 'results')):
                if os.path.isdir(os.path.join(d, 'results')):
                    
                    # check if the visualization was obtained or not
                    if len(os.listdir(os.path.join(d, 'results'))) >= treshold:
                        pass
                    else:
                        visualized = False
                else:
                    visualized = False
            else:
                visualized = False

        if visualized:
            pass
        else:
            folers_to_compute.append(c)
            
    to_compute = folers_to_compute[::-1]
    return to_compute


def save_batch(folder):
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
    names = 'combined_img_line.png'
    figures = ['Depolarization.png', 'results/CUSA_fig.png', 'Intensity.png', 'Linear Retardance.png']

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