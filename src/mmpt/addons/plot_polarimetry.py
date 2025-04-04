import numpy as np
import os
import cv2

from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator
from matplotlib import cm, rcParams
rcParams['backend'] = 'Agg'

from mmpt.utils import get_cmap, load_plot_parameters, load_combined_plot_name, load_npz_file


def visualize_MM(path_save: str, MM: dict = None, MM_path: str = None, processing_mode: str = "default", save_pdf_figs: str = False,
                 instrument: str = "IMP", mm_processing: str = "torch"):
    """
    Visualizes and generates plots based on the provided parameters and MM (measurement) data.

    Parameters:
    ----------
    parameters: dict
        the processing parameters in a dictionary format
    MM: dict
        the Mueller matrix data
    path_save : str
        the path to the folder where the plots should be saved

    Returns:
    -------
    None

    Description:
    This function processes and visualizes data based on the specified 'processing_mode'. 
    Depending on the mode, it generates various plots, including histograms and MM details, 
    and saves them as PDF files if the 'save_pdf_figs' flag is set to True. In 'full' mode, 
    additional histograms and a batch save process are performed.
    """
    if MM is None:
        MM = load_npz_file(MM_path)
    
    os.makedirs(path_save, exist_ok=True)
    
    if processing_mode == 'full':
        parameters_histograms(MM, path_save, save_pdf_figs = save_pdf_figs, instrument = instrument, mm_processing = mm_processing)
        show_MM(MM['nM'], path_save, save_pdf_figs = save_pdf_figs, instrument = instrument, mm_processing = mm_processing)
        MM_histogram(MM, path_save, save_pdf_figs = save_pdf_figs)
        
    if processing_mode != 'no_viz':
        generate_plots(MM, path_save, save_pdf_figs = save_pdf_figs, instrument = instrument, mm_processing = mm_processing)
        if processing_mode == 'full':
            if instrument == 'IMP':
                save_batch(path_save)
            else:
                print(' [wrn] Batch save not implemented for this instrument.')
            
            
###################################################################################################################
###################################################################################################################
#################################### 1. Plot histograms of the parameters #########################################
###################################################################################################################
###################################################################################################################

def parameters_histograms(MM: dict, path_save: str, max_ = False, save_pdf_figs = False, instrument: str = "IMP", mm_processing: str = "torch"):
    """
    Generate histograms for the four parameters extracted from Mueller Matrices.

    Parameters
    ----------
    MM : dict
        Dictionary containing the computed Mueller Matrices parameters.
    path_save : str
        Path where the histograms should be saved.
    max_ : bool, optional
        If True, includes the maximum value in the histogram text (default: False).
    save_pdf_figs : bool, optional
        If True, also saves the figure as a PDF (default: False).
    """
    parameters = ['azimuth', 'totP', 'totD', 'linR']
    parameters_map = {key: load_plot_parameters(instrument, mm_processing)[key] for key in parameters}
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 11))

    for i, (key, param) in enumerate(parameters_map.items()):
        if key not in MM:
            continue  # Skip missing keys
        
        row, col = divmod(i, 2)
        ax = axes[row, col]

        range_hist = param['cbar_hist']

        data = MM[key]
        y, x = np.histogram(data, bins=500, density=False, range=range_hist)
        x_plot = (x[:-1] + x[1:]) / 2

        # Compute statistics
        max_val = x[np.argmax(y)]
        mean = np.nanmean(data)
        std = np.nanstd(data)
        y = y / np.max(y)  # Normalize

        # Plot histogram
        ax.plot(x_plot, y, color='black', linewidth=3)
        ax.set_ylim(0, 1.5)
        ax.locator_params(axis='y', nbins=4)
        ax.locator_params(axis='x', nbins=5)

        # Set text on the plot
        text = f'$\\mu$ = {mean:.3f}\n$\\sigma$ = {std:.3f}'
        if max_:
            text += f'\nmax = {max_val:.3f}'

        ax.text(0.75, 0.85, text,
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                fontsize=25, fontweight='bold')

        # Set font sizes for axes
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(22)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(22)
            tick.label1.set_fontweight('bold')

        ax.set_title(param['title'], fontdict={'fontsize': 30, 'fontweight': 'bold'})
        ax.set_ylabel('Normalized pixel number', fontdict={'fontsize': 25, 'fontweight': 'bold'})

    plt.tight_layout()

    # Save figures
    path_png = os.path.join(path_save, 'parameters_histogram.png')
    path_pdf = os.path.join(path_save, 'parameters_histogram.pdf')

    fig.savefig(path_png, dpi=80)
    if save_pdf_figs:
        fig.savefig(path_pdf, dpi=100)

    plt.close()


###################################################################################################################
###################################################################################################################
################################ 2. Plot maps of the polarimetric parameters ######################################
###################################################################################################################
###################################################################################################################

def generate_plots(MM: dict, folder: str, save_pdf_figs=False, instrument: str = "IMP", mm_processing: str = "torch"):
    """
    Generates and saves plots for polarimetric parameter maps.

    Parameters
    ----------
    MuellerMatrices : dict
        The computed Mueller Matrix with parameters as keys.
    folder : str
        The name of the currently processed folder.
    save_pdf_figs : bool, optional
        Whether to save figures as PDFs (default: False).
    """
    parameters_map = load_plot_parameters()
    parameters_map.pop('MM')

    for key, param in parameters_map.items():

        param_name = param['title'].replace(' (\u00B0)', '').lower()
        path_save = os.path.join(folder, f"{param_name}.png".capitalize())
        
        cmap, norm = get_cmap(key, instrument, mm_processing=mm_processing)
        plot_polarimetric_parameter(MM[key], cmap, norm, key, param['title'], path_save, save_pdf_figs, instrument, mm_processing)


def plot_polarimetric_parameter(X2D: np.ndarray, cmap, norm, parameter: str, title:str, path_save: str,
                                save_pdf_figs: bool = False, instrument: str = "IMP", mm_processing: str = "torch",
                                ax: Optional[plt.Axes] = None):
    """
    Displays and saves an individual 2D polarimetric parameter map.

    Parameters
    ----------
    X2D : np.ndarray
        The 2D image data.
    cmap : colormap
        The colormap to be used for plotting.
    norm : colormap norm
        The normalization associated with the colormap.
    parameter : str
        The name of the parameter (e.g., 'azimuth', 'depolarization').
    path_save : str
        The path to save the final plot.
    folder : str
        The name of the currently processed folder.
    save_pdf_figs : bool, optional
        Whether to save figures as PDFs (default: False).
    """
    if parameter != 'azimuth':
        X2D = np.nan_to_num(X2D)

    # Load plot parameters
    plot_parameters = load_plot_parameters(instrument, mm_processing)[parameter]
    [cbar_min, cbar_max], cbar_step = plot_parameters['cbar'], plot_parameters['cbar_step']
    formatter = plot_parameters['formatter']

    if parameter == 'M11' and instrument == 'IMP':
        X2D /= 10000  # Normalize intensity

    np.clip(X2D, cbar_min, cbar_max, out=X2D)

    if path_save:
        # Save raw image without colorbar
        plt.imsave(path_save.replace('.png', '_img.png'), X2D, cmap=cmap, vmin=cbar_min, vmax=cbar_max)

    # Create figure with colorbar
    # Create figure and ax if not provided
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 10))
        own_fig = True
    else:
        fig = ax.get_figure()
        
    cax = ax.imshow(X2D, cmap=cmap, norm=norm)

    if own_fig:
        cbar = fig.colorbar(cax, ax=ax, pad=0.02, fraction=0.06)
        ticks = np.arange(cbar_min, cbar_max + 0.01, cbar_step)
        cbar.set_ticks(ticks)  # Ensure fixed tick locations
        cbar.ax.yaxis.set_major_locator(FixedLocator(ticks))  # Fix locator
        cbar.ax.set_yticklabels([formatter.format(a) for a in ticks], fontsize=40, weight='bold')

        if parameter == 'M11':
            ax.text(500, -10, "x10\u2074", fontsize=40, fontweight="bold")
        
    # Set title
    ax.set_title(title, fontsize=35, fontweight="bold", pad=14)
    ax.set_xticks([])
    ax.set_yticks([])

    if own_fig and path_save:
        fig.savefig(path_save, dpi=80)
        if save_pdf_figs:
            fig.savefig(path_save.replace('.png', '.pdf'), dpi=100)
        plt.close(fig)

###################################################################################################################
###################################################################################################################
######################################### 3. Plot the mueller matrix ##############################################
###################################################################################################################
###################################################################################################################


def show_MM(X3D, folder, save_pdf_figs=False, instrument: str = "IMP", mm_processing: str = "torch"):
    """
    Display the 16 components (e.g., of Mueller Matrix) in a 4x4 montage.

    Parameters
    ----------
    X3D : np.ndarray
        3D stack of 2D components with shape (dim[0], dim[1], 16)
    folder : str
        Path to the folder where the figure should be saved
    save_pdf_figs : bool, optional
        Whether to save the figure as a PDF in addition to PNG (default is False)
    """
    shp3 = np.shape(X3D)
    if len(shp3) != 3 or shp3[-1] != 16:
        raise ValueError(f"X3D must have shape (height, width, 16). Found: {shp3}")

    # Create montage matrix
    X_montage = np.block([
        [rescale_MM(X3D[:, :, 0]), 5*X3D[:, :, 1], 5*X3D[:, :, 2], 5*X3D[:, :, 3]],
        [5*X3D[:, :, 4], rescale_MM(X3D[:, :, 5]), 5*X3D[:, :, 6], 5*X3D[:, :, 7]],
        [5*X3D[:, :, 8], 5*X3D[:, :, 9], rescale_MM(X3D[:, :, 10]), 5*X3D[:, :, 11]],
        [5*X3D[:, :, 12], 5*X3D[:, :, 13], 5*X3D[:, :, 14], rescale_MM(X3D[:, :, 15])]
    ])
    
    cmap, norm = get_cmap(parameter='MM', instrument = instrument, mm_processing = mm_processing)
    cbar_min, cbar_max, cbar_step = -1, 1, 0.5
    X_montage = np.clip(X_montage, cbar_min, cbar_max)
    
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.imshow(X_montage, cmap=cmap)

    # Constants
    cell_width, cell_height = 516, 388  # Base dimensions
    last_col_adjust, last_row_adjust = 8, 2  # Adjustments for last column/row

    # Line widths
    lw, lw_color = 10, 4
    color_mapping = {True: '#1F51FF', False: '#e0ae00'}  # Diagonal elements (blue), others (yellow)

    # Draw white outer borders
    for idx in range(16):
        row, col = divmod(idx, 4)
        x, y = col * cell_width, row * cell_height
        width, height = cell_width - 2, cell_height - 2  # Default size

        # Adjust for last column and last row
        if col == 3:
            width -= last_col_adjust
        if row == 3:
            height -= last_row_adjust

        ax.add_patch(Rectangle((x, y), width, height, edgecolor='white', facecolor='none', lw=lw))

    # Draw inner colored borders
    for idx in range(16):
        row, col = divmod(idx, 4)
        x, y = col * cell_width + lw, row * cell_height + lw
        width, height = cell_width - 2 * lw, cell_height - 2 * lw  # Default inner size

        # Adjust for last column and last row
        if col == 3:
            width -= last_col_adjust
        if row == 3:
            height -= last_row_adjust

        edge_color = color_mapping[row == col]  # Blue if diagonal, otherwise yellow

        ax.add_patch(Rectangle((x, y), width, height, edgecolor=edge_color, facecolor='none', lw=lw_color))

    # Function to format and add colorbars
    def add_colorbar(position, tick_formatter, color, pad, scale=1):
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), pad=pad, location=position, shrink=0.65,
                            ticks=np.arange(cbar_min, cbar_max + cbar_step, cbar_step), ax=ax)
        cbar.ax.set_yticklabels([tick_formatter.format(a / scale) for a in np.arange(cbar_min, cbar_max + cbar_step, cbar_step)],
                                fontsize=30, weight='bold')
        cbar.ax.tick_params(size=0)
        cbar.ax.add_patch(Rectangle((4.9, 4.9), 0.2, 0.2, fill=None, alpha=1))
        cbar.ax.get_children()[7].set_color(color)
        cbar.ax.get_children()[7].set_linewidth(5)

    # Add two formatted colorbars
    add_colorbar(position='right', tick_formatter='{:.1f}', color='#1F51FF', pad=0.025)
    add_colorbar(position='left', tick_formatter='{:.2f}', color='#e0ae00', pad=0.04, scale=5)

    # Set axis properties
    ax.spines.values().__iter__().__next__().set_color('#FFFFFF')  # Set all spines to white
    plt.xticks([])
    plt.yticks([])
    plt.title('Mueller Matrix', fontsize=40, fontweight="bold", pad=20)

    ax.spines['top'].set_color('#FFFFFF')
    ax.spines['right'].set_color('#FFFFFF')
    ax.spines['bottom'].set_color('#FFFFFF')
    ax.spines['left'].set_color('#FFFFFF')
    
    
    # Save figure
    plt.savefig(os.path.join(folder, 'MM.png'), dpi=100)
    if save_pdf_figs:
        plt.savefig(os.path.join(folder, 'MM.pdf'), dpi=100)
    plt.close()

    return X_montage
    

def rescale_MM(X2D: np.ndarray):
    """
    Rescale MM component for consistent visualization. """
    return X2D if np.max(X2D) == np.min(X2D) == 1 else X2D / np.max(np.abs(X2D))


def MM_histogram(MM: dict, folder: str, save_pdf_figs: bool = False):
    """
    Create histograms for the Mueller matrices components.

    Parameters
    ----------
    MM : dict
        Dictionary containing the computed Mueller Matrices.
    folder : str
        Path where the histograms should be saved.
    save_pdf_figs : bool, optional
        If True, also saves the figure as a PDF (default: False).
    """
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(25, 17), sharex=True, sharey=True)
    plt.locator_params(axis='y', nbins=4)
    plt.locator_params(axis='x', nbins=5)

    for i in range(16):
        row, col = divmod(i, 4)
        ax = axes[row, col]

        # Extract data
        data = MM['nM'][:, :, i].flatten()
        if data.size == 0:
            continue  # Skip empty data

        # Compute histogram
        y, x = np.histogram(data, bins=500, density=False, range=(-1, 1))
        x_plot = (x[:-1] + x[1:]) / 2  # Compute bin centers

        y = y / np.max(y)  # Normalize
        max_ = x[np.argmax(y)]  # Get max value

        # Plot histogram
        ax.plot(x_plot, y, color='black', linewidth=3)

        # Format ticks
        if row == 3:
            ax.xaxis.set_tick_params(labelsize=20, width=2)
        if col == 0:
            ax.yaxis.set_tick_params(labelsize=20, width=2)

        # Compute statistics
        mean = np.mean(data)
        std = np.std(data)

        # Add text with statistics
        text = f"$\\mu$ = {mean:.3f}\n$\\sigma$ = {std:.3f}\nmax = {max_:.3f}"
        ax.text(-1, 0.8, text, fontsize=16, fontweight='bold')

        # Set title for each subplot
        ax.set_title(f'MM[{row},{col}]', fontsize=18, fontweight='bold')

    # Adjust layout and save figure
    fig.suptitle('Mueller Matrix Histogram', fontsize=40, fontweight='bold', y=1)
    plt.tight_layout()

    plt.savefig(os.path.join(folder, 'MM_histogram.png'))
    if save_pdf_figs:
        plt.savefig(os.path.join(folder, 'MM_histogram.pdf'))
    
    plt.close()


###################################################################################################################
###################################################################################################################
######################### 4. Save an image combining all the polarimetric parameters ##############################
###################################################################################################################
###################################################################################################################

def save_batch(folder: str, viz = False):
    """
    combine the 4 images (linear retardance, depolarization, azimuth and intensity in a single file)

    Parameters
    ----------
    folder : str
        the folder in which to find the images
    """
    
    names = load_combined_plot_name(viz)

    # load the four images
    img_3 = cv2.imread(os.path.join(folder, names['files'][0]))[40:960, 160:1450]
    img_2 = cv2.imread(os.path.join(folder, names['files'][1]))[40:960, 160:1450]
    img_1 = cv2.imread(os.path.join(folder, names['files'][2]))[40:960, 160:1450]
    img_4 = cv2.imread(os.path.join(folder, names['files'][3]))[40:960, 160:1450]

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
    cv2.imwrite(os.path.join(folder, names['save_file']), output)
    
###################################################################################################################
###################################################################################################################
############################### 5. Compare different backends for the computation #################################
###################################################################################################################
###################################################################################################################   
    
    
from mmpt import utils

def visualize_comparison(mmProcessor, mm_computation_backends, lu_chipman_backends):
    """Visualize the comparison of the MM computation and lu-chipman prediction."""
    samples, _ = utils.get_measurements_to_process(mmProcessor.get_parameters(), PDDN=False)
        
    import itertools
    combinations = list(itertools.product(mm_computation_backends, lu_chipman_backends))
        
    to_visualize = ['M11', 'totD', 'totP', 'linR', 'azimuth', 'azimuth_local_var']
        
    parameters_map = utils.load_plot_parameters(instrument=mmProcessor.instrument, mm_processing='c')
    parameters_map.pop('MM')

    for sample in samples:
        MMs = {}
        output_folder = os.path.join(sample['folder_name'], 'test_backends', sample['wavelength'])
        output_plots = os.path.join(output_folder, 'comparison')
        os.makedirs(output_plots, exist_ok=True)
        
        for combo in combinations:
            MM_path = os.path.join(output_folder, f'{combo[0]}_{combo[1]}', 'MM.npz')
            MMs[f'{combo[0]}_{combo[1]}'] = np.load(MM_path)
            
        for parameter in to_visualize:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            cmap, norm = utils.get_cmap(parameter, mmProcessor.instrument, mm_processing='torch')

            axes_flat = axes.flat  # or axes.ravel()
                
            for backends, MM in MMs.items():
                ax = next(axes_flat)
                plot_polarimetric_parameter(X2D = MM[parameter], cmap = cmap, norm = norm, parameter = parameter,
                                                                title = None, path_save = None, save_pdf_figs = False, 
                                                                instrument = mmProcessor.instrument, mm_processing = "torch",
                                                                ax = ax)
                ax.set_title(backends)
                    
            fig.tight_layout()
            fig.savefig(os.path.join(output_plots, f"{parameter}_comparison.png"))
            plt.close()
            
            # Prepare subplots
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.ravel()  # flatten for easy indexing

            range_hist = parameters_map[parameter]['cbar']

            for idx, (backends, MM) in enumerate(MMs.items()):
                if idx >= 4:
                    break  # only show 4 plots max

                ax = axes[idx]
                data = MM[parameter].flatten()

                if parameter == 'M11':
                    data = data / 10000
                data = data[~np.isnan(data)]  # remove NaNs

                y, x = np.histogram(data, bins=500, density=False, range=range_hist)
                # Remove the first and last bins
                x_plot = (x[2:-1])  # New bin centers (499 values)
                y = y[1:-1]  # Remove the first and last y-values (499 values)

                # Normalize y values
                y = y / np.max(y) if np.max(y) > 0 else y  # Normalize safely

                # Debugging output
                max_val = x[np.argmax(y)]
                mean = np.nanmean(data)
                std = np.nanstd(data)
                
                # Plot
                ax.plot(x_plot, y, color='black', linewidth=1.5)
                ax.set_title(backends, fontsize=10)
                ax.set_ylim(0, 1.2)
                ax.locator_params(axis='y', nbins=10)
                ax.locator_params(axis='x', nbins=10)
                
                # Set text on the plot
                text = f'$\\mu$ = {mean:.3f}\n$\\sigma$ = {std:.3f}'
                text += f'\nmax = {max_val:.3f}'
                ax.text(0.5, 0.85, text,
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                    fontsize=10, fontweight='bold')
                
                ax.grid(True)

                if idx in [0, 2]:  # Only left plots get Y label
                    ax.set_ylabel("Normalized Count")
                if idx in [2, 3]:  # Only bottom plots get X label
                    ax.set_xlabel(parameter)

            # Hide any unused axes
            for j in range(idx + 1, 4):
                fig.delaxes(axes[j])

            fig.tight_layout()  # leave space for suptitle
            fig.savefig(os.path.join(output_plots, f"{parameter}_histogram_subplots.png"))
            plt.close()
