import numpy as np
import os
import cv2

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm

from processingmm.helpers import load_parameter_maps, get_cmap, load_plot_parameters, load_combined_plot_name, load_filenames_combined_plot


def parameters_histograms(MuellerMatrices: dict, folder: str, max_ = False):
    """
    generate the histogram for the four parameters

    Parameters
    ----------
    MuellerMatrices : dict
        the dictionnary containing the computed Mueller Matrices
    folder : str
        the name of the current processed folder
    max_ : bool
        boolean indicating wether or not the max_ should be printed
    """
    parameters_map = load_parameter_maps()

    try:
        parameters_map.pop('M11')
    except:
        pass
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,11))
    
    for i, (key, param) in zip(range(0,4), parameters_map.items()):
        row = i%2
        col = i//2
        ax = axes[row, col]
        
        # change the range of the histograms
        if param[2]:
            range_hist = (0, 1)
        elif param[1]:
            range_hist = (0, 180)
        elif param[3]:
            range_hist = (0, 0.20)
        else:
            range_hist = (0, 60)
        
        y, x = np.histogram(
            MuellerMatrices[folder][key],
            bins=500,
            density=False,
            range = range_hist)
        
        x_plot = []
        for idx, x_ in enumerate(x):
            try: 
                x_plot.append((x[idx] + x[idx + 1]) / 2)
            except:
                assert len(x_plot) == 500
        
        # get the mean, max and std
        max_ = x[np.argmax(y)]
        mean = np.nanmean(MuellerMatrices[folder][key])
        std = np.nanstd(MuellerMatrices[folder][key])
        
        y = y / np.max(y)
        
        # plot the histogram
        ax.plot(x_plot,y, c = 'black', linewidth=3)
        ax.axis(ymin=0,ymax=1.5)
        ax.locator_params(axis='y', nbins=4)
        ax.locator_params(axis='x', nbins=5)
    
        if max_:
            ax.text(0.75, 0.85, '$\mu$ = {:.3f}\n$\sigma$ = {:.3f}'.format(mean, std, max_), 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, 
                    fontsize=25, fontweight = 'bold')
        else:
            ax.text(0.75, 0.85, '$\mu$ = {:.3f}\n$\sigma$ = {:.3f}'.format(mean, std), 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, 
                    fontsize=25, fontweight = 'bold')
        
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(22)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(22)
            tick.label1.set_fontweight('bold')
            
        ax.set_title(param[0], fontdict = {'fontsize': 30, 'fontweight': 'bold'})
        ax.set_ylabel('Normalized pixel number', fontdict = {'fontsize': 25, 'fontweight': 'bold'})
        
    # save the figures
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'parameters_histogram.png'))
    plt.savefig(os.path.join(folder, 'parameters_histogram.pdf'))
    
    plt.close()


def generate_plots(MuellerMatrices: dict, folder: str):
    """
    master function allowing to create the plots for the polarimetric parameters map, calls the function plot_polarimetric_paramter for 
    each parameter

    Parameters
    ----------
    MuellerMatrices : dict
        the computed Mueller Matrices with folders and parameters as keys
    folder : str
        the name of the current processed folder
    """
    parameters_map = load_parameter_maps()
    
    plots = []
    for key, param in parameters_map.items():
        path_save = os.path.join(folder, param[0].replace(' (°)', '') + '.png')
        path_save_histo = os.path.join(path_save.split('\\polarimetry')[0], 'histology', 
                            param[0].replace(' (°)', '') + '_' + path_save.split('nm\\')[0].split('\\')[-1] + 'nm.png')
        
        if 'Diattenuation' in path_save:
            pass
        else:
            if 'Depolarization' in path_save:
                parameter = 'depolarization'
            elif 'retardance' in path_save:
                parameter = 'retardance'
            elif 'Azimuth' in path_save:
                parameter = 'azimuth'
            elif 'Intensity' in path_save:
                parameter = 'intensity'
            else:
                pass
            cmap, norm = get_cmap(parameter)

            plot_polarimetric_paramter(MuellerMatrices[folder][key], cmap, norm, parameter, path_save, folder)


def plot_polarimetric_paramter(X2D: np.ndarray, cmap, norm, parameter: str, path_save: str, folder: str):
    """
    function to display an individual 2D component (e.g. one component of the Mueller Matrix coeffs, or a Mask)

    Parameters
    ----------
    X2D : 2D Image of shape shp2 = [dim[0],dim[1]]
    cmap : colormap
        the colormap to be used for plotting
    norm : colormap norm
        the norm associated to the colormap
    parameter : str
        the name of the parameter (i.e. 'azimuth', 'depolarization'...)
    path_save : str
        the path to which the final plot should be saved
    folder : str
        the name of the current processed folder
    """
    # load the parameters that will be used for the plot
    plot_parameters = load_plot_parameters()[parameter]
    cbar_min = plot_parameters['cbar_min']
    cbar_max = plot_parameters['cbar_max']
    cbar_step = plot_parameters['cbar_step']
    formatter = plot_parameters['formatter']
    
    if parameter == 'intensity':
        X2D = X2D / 10000

    # rescale the matrices to have the right color bars
    X2D[X2D < cbar_min] = cbar_min
    X2D[X2D > cbar_max] = cbar_max
    if len(X2D[X2D < cbar_min]) != 0:
        pass
    else:
        X2D[-1, -1] = cbar_min
    if len(X2D[X2D > cbar_max]) != 0:
        pass
    else:
        X2D[-1, 0] = cbar_max

    # 1. save the plot as an image only
    path_save_img = path_save.replace('.png', '_img.png')
    plt.imsave(path_save_img, X2D, cmap = cmap, vmin = cbar_min, vmax = cbar_max)

    # 2. save the plot with the colorbar and title
    fig, ax = plt.subplots(figsize = (15,10))
    ax.imshow(X2D, cmap = cmap)
    ax = plt.gca()

    # format the color bar
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), pad = 0.02, 
                        ticks=np.arange(cbar_min, cbar_max + cbar_step, cbar_step), fraction=0.06)
    cbar.ax.set_yticklabels([formatter.format(a) for a in np.arange(cbar_min, cbar_max+cbar_step, cbar_step)], 
                            fontsize=25, weight='bold')
    
    if parameter == 'intensity':
        ax.text(500, -10, "x10⁴", fontsize=25, fontweight="bold")

    title = path_save.split('\\')[-1].split('.')[0]
    plt.title(title, fontsize=35, fontweight="bold", pad=14)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path_save)
    plt.savefig(path_save.replace('.png', '.pdf'))
    plt.close('all')

    # 3. for the intensity, save the realsize image
    if parameter == 'intensity':
        folder = folder.replace('\\', '/')
        path_save = os.path.join('/'.join(folder.split('/')[:-1]), folder.split('/')[-1],
                    folder.split('/')[-3] + '_' + folder.split('/')[-1] + '_realsize.png').replace('\\', '/')
        plt.imsave(path_save, X2D, cmap = cmap, vmin = cbar_min, vmax = cbar_max)
 

def show_MM(X3D, folder):
    """
    function to display the 16 components (e.g. of Mueller Matrix) in a montage form (4x4)

    Parameters
    ----------
    X3D : 3D stack of 2D Components of shape shp3
        the matrix X must have shape equal to [dim[0],dim[1],16]
    folder : str
        the path to the folder in which the figure should be saved
    backend : str
        the format in which to save the images (default 'png')
    """
    shp3 = np.shape(X3D)
    if (np.prod(np.shape(shp3)) != 3):
        raise Exception(
            'Input: "X3D" should have shape of a 3D image, e.g. (idx0, idx1, idx2). The shape value was found: {}'.format(
                shp3))
    if (shp3[-1] != 16):
        raise Exception(
            'Input: "X3D" should have shape 16 components in the last dimension. The number of elements in the last dimension is: {}'.format(
                shp3[-1]))

    # create the rescaled matrix that will be used to plot
    X_montage = np.concatenate((np.concatenate(
        (rescale_MM(X3D[:, :, 0]).squeeze(), 5*X3D[:, :, 1].squeeze(), 5*X3D[:, :, 2].squeeze(), 5*X3D[:, :, 3].squeeze()), axis=1),
        np.concatenate((5*X3D[:, :, 4].squeeze(), rescale_MM(X3D[:, :, 5]).squeeze(), 5*X3D[:, :, 6].squeeze(),
                                                5*X3D[:, :, 7].squeeze()), axis=1),
                                np.concatenate((5*X3D[:, :, 8].squeeze(), 5*X3D[:, :, 9].squeeze(), rescale_MM(X3D[:, :, 10]).squeeze(),
                                                5*X3D[:, :, 11].squeeze()), axis=1),
                                np.concatenate((5*X3D[:, :, 12].squeeze(), 5*X3D[:, :, 13].squeeze(),
                                                5*X3D[:, :, 14].squeeze(), rescale_MM(X3D[:, :, 15]).squeeze()), axis=1)), axis=0)
                               
    # get the colormap to be used to plot the MM
    cmap, norm = get_cmap(parameter = 'MM')
    cbar_max = 1
    cbar_min = -1
    cbar_step = 0.5
    
    # rescale the matrices to have the right color bars
    if len(X_montage[X_montage < cbar_min]) != 0:
            pass
    else:
        X_montage[-1,-1] = cbar_min
        
    if len(X_montage[X_montage > cbar_max]) != 0:
        pass
    else:
        X_montage[-1, 0] = cbar_max
    X_montage[X_montage < cbar_min] = cbar_min
    X_montage[X_montage > cbar_max] = cbar_max
    formatter = '{:.1f}'
    formatter_v2 = '{:.2f}'
    
    fig, ax = plt.subplots(figsize = (20,15))
    ax.imshow(X_montage, cmap = cmap)
    
    lw=15

    for idx in range(0, 16):
        if idx == 0:
            ax.add_patch(Rectangle((5, 5), 511, 385,
                                   edgecolor = 'white',
                                 facecolor = (1,0,0,0),
                                 fill=True,
                                 lw=lw))
        elif idx < 3:
            ax.add_patch(Rectangle((idx % 4 * 516, 6), 514, 383,
                                  edgecolor = 'white',
                                 facecolor = (1,0,0,0),
                                 fill=True,
                                 lw=lw))
        elif idx == 3:
            ax.add_patch(Rectangle((idx % 4 * 516, 6), 508, 383,
                                  edgecolor = 'white',
                                 facecolor = (1,0,0,0),
                                 fill=True,
                                 lw=lw))
        elif idx % 4 == 0 and idx // 4 == 3:
            ax.add_patch(Rectangle((5, idx //4 *388), 511, 382,
                                  edgecolor = 'white',
                                 facecolor = (1,0,0,0),
                                 fill=True,
                                 lw=lw))
        elif idx % 4 == 0:
            ax.add_patch(Rectangle((5, idx //4 *388), 511, 385,
                                  edgecolor = 'white',
                                 facecolor = (1,0,0,0),
                                 fill=True,
                                 lw=lw))
        elif idx // 4 == 3 and idx % 4 == 3:
            ax.add_patch(Rectangle((idx % 4 *516, idx //4 *388), 511, 385,
                                  edgecolor = 'white',
                                 facecolor = (1,0,0,0),
                                 fill=True,
                                 lw=lw))
        elif idx // 4 == 3:
            ax.add_patch(Rectangle((idx % 4 *516, idx //4 *388), 514, 386,
                                  edgecolor = 'white',
                                 facecolor = (1,0,0,0),
                                 fill=True,
                                 lw=lw))
        elif idx % 4 == 3:
            ax.add_patch(Rectangle((idx % 4 *516, idx //4 *388), 508, 386,
                                  edgecolor = 'white',
                                 facecolor = (1,0,0,0),
                                 fill=True,
                                 lw=lw))
        else:
            ax.add_patch(Rectangle((idx % 4 *516, idx //4 *388), 515, 386,
                                  edgecolor = 'white',
                                 facecolor = (1,0,0,0),
                                 fill=True,
                                 lw=lw))
            
            
            
    lw_color=5

    for idx in range(0, 16):
        if idx == 0:
            ax.add_patch(Rectangle((5 + lw, 5 + lw + 1), 511 - 2*lw, 384 - 2*lw,
                                   edgecolor = '#1F51FF',
                                 facecolor = (1,0,0,0),
                                 fill=True,
                                 lw=lw_color))
        elif idx < 3:
            ax.add_patch(Rectangle((idx % 4 * 516 + lw , 6 + lw ), 514 - 2*lw, 383 - 2*lw,
                                  edgecolor = '#e0ae00',
                                 facecolor = (1,0,0,0),
                                 fill=True,
                                 lw=lw_color))
        elif idx == 3:
            ax.add_patch(Rectangle((idx % 4 * 516 + lw, 6 + lw), 508 - 2*lw , 383 - 2*lw,
                                  edgecolor = '#e0ae00',
                                 facecolor = (1,0,0,0),
                                 fill=True,
                                 lw=lw_color))
            
        elif idx % 4 == 0 and idx // 4 == 3:
            ax.add_patch(Rectangle((5 + lw , idx //4 *388 + lw), 511 - 2*lw, 382 - 2*lw,
                                  edgecolor = '#e0ae00',
                                 facecolor = (1,0,0,0),
                                 fill=True,
                                 lw=lw_color))
        elif idx % 4 == 0:
            ax.add_patch(Rectangle((5 + lw, idx //4 *388 + lw), 511 - 2*lw, 385 - 2*lw,
                                  edgecolor = '#e0ae00',
                                 facecolor = (1,0,0,0),
                                 fill=True,
                                 lw=lw_color))
        elif idx // 4 == 3 and idx % 4 == 3:
            ax.add_patch(Rectangle((idx % 4 *516 + lw, idx //4 *388 + lw), 511 - 2*lw, 385 - 2*lw,
                                  edgecolor = '#1F51FF',
                                 facecolor = (1,0,0,0),
                                 fill=True,
                                 lw=lw_color))
        elif idx // 4 == 3:
            ax.add_patch(Rectangle((idx % 4 *516 + lw, idx //4 *388 + lw), 514 - 2*lw, 386  - 2*lw,
                                  edgecolor = '#e0ae00',
                                 facecolor = (1,0,0,0),
                                 fill=True,
                                 lw=lw_color))
        elif idx % 4 == 3:
            ax.add_patch(Rectangle((idx % 4 *516 + lw, idx //4 *388 + lw), 508 - 2*lw, 386 - 2*lw,
                                  edgecolor = '#e0ae00',
                                 facecolor = (1,0,0,0),
                                 fill=True,
                                 lw=lw_color))
        else:
            if idx % 4 == idx // 4:
                ax.add_patch(Rectangle((idx % 4 *516 + lw, idx //4 *388 + lw), 515 - 2*lw, 386 - 2*lw,
                                      edgecolor = '#1F51FF',
                                     facecolor = (1,0,0,0),
                                     fill=True,
                                     lw=lw_color))
            else:
                ax.add_patch(Rectangle((idx % 4 *516 + lw, idx //4 *388 + lw), 515 - 2*lw, 386 - 2*lw,
                                      edgecolor = '#e0ae00',
                                     facecolor = (1,0,0,0),
                                     fill=True,
                                     lw=lw_color))
            
    # format the color bar
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), pad = 0.025, location='right', shrink = 0.65,
                        ticks=np.arange(cbar_min, cbar_max + cbar_step, cbar_step))
    cbar.ax.set_yticklabels([formatter.format(a) for a in np.arange(cbar_min, cbar_max+cbar_step, cbar_step)], 
                            fontsize=30, weight='bold')
    cbar.ax.tick_params(size=0)
    cbar.ax.get_children()[7].set_color('#1F51FF')
    cbar.ax.get_children()[7].set_linewidth(5)
    
    # format the color bar
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), pad = 0.04, location='left', shrink = 0.65,
                        ticks=np.arange(cbar_min, cbar_max + cbar_step, cbar_step))
    cbar.ax.set_yticklabels([formatter_v2.format(a/5) for a in np.arange(cbar_min, cbar_max+cbar_step, cbar_step)], 
                            fontsize=30, weight='bold')
    cbar.ax.tick_params(size=0)
    cbar.ax.get_children()[7].set_color('#e0ae00')
    cbar.ax.get_children()[7].set_linewidth(5)
    
    ax.spines['top'].set_color('#FFFFFF')
    ax.spines['right'].set_color('#FFFFFF')
    ax.spines['bottom'].set_color('#FFFFFF')
    ax.spines['left'].set_color('#FFFFFF')
    
    plt.xticks([])
    plt.yticks([])
    plt.title('Mueller Matrix', fontsize=40, fontweight="bold", pad=20)
    plt.savefig(os.path.join(folder, 'MM.' + 'png'))
    plt.savefig(os.path.join(folder, 'MM.' + 'pdf'))
    plt.close()
    
    return X_montage


def rescale_MM(X2D: np.ndarray):
    """
    resize each of the 16 MM components for plotting it

    Parameters
    ----------
    X2D : np.ndarray
        the MM component

    Returns
    ----------
    newvalues : np.ndarray
        the rescaled MM component
    """
    if np.max(X2D) == np.min(X2D) == 1:
        return X2D
    a = 2 / (np.max(X2D) - np.min(X2D))
    a = 1
    b = 1 - a * np.max(X2D)
    b = 0
    newvalue = a * X2D + b
    return newvalue


def MM_histogram(MuellerMatrices: dict, folder: str):
    """
    function to create the histograms for the Mueller matrices

    Parameters
    ----------
    MuellerMatrices : dict
        the dictionnary containing the computed Mueller Matrices
    folder : str
        the name of the current processed folder
    """
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(25,17), sharex = True, sharey = True)
    plt.locator_params(axis='y', nbins=4)
    plt.locator_params(axis='x', nbins=5)

    for i in range(0,16):
        row = i%4
        col = i//4
        ax = axes[row, col]

        # get the histogram data
        y, x = np.histogram(
        MuellerMatrices[folder]['nM'][:,:,i].flatten(),
        bins=500,
        density=False,
        range = (-1, 1))

        x_plot = []
        for idx, x_ in enumerate(x):
            try: 
                x_plot.append((x[idx] + x[idx + 1]) / 2)
            except:
                assert len(x_plot) == 500

        y = y / np.max(y)
        max_ = x[np.argmax(y)]
        
        # plot the histogram
        ax.plot(x_plot,y, c = 'black', linewidth=3)

        if row == 3:
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(20)
                tick.label1.set_fontweight('bold')
        if col == 0:
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(20)
                tick.label1.set_fontweight('bold')

        # get the mean and std of the component of interest
        mean = np.mean(MuellerMatrices[folder]['nM'][:,:,i].flatten())
        std = np.std(MuellerMatrices[folder]['nM'][:,:,i].flatten())

        if row == 0 and col ==0:
            ax.text(-1, 0.8, '$\mu$ = {:.0f}\n$\sigma$ = {:.0f}\nmax = {:.3f}'.format(mean, std, max_), fontsize=16, fontweight = 'bold')
        else:
            ax.text(-1, 0.8, '$\mu$ = {:.3f}\n$\sigma$ = {:.3f}\nmax = {:.3f}'.format(mean, std, max_), fontsize=16, fontweight = 'bold')

    # actually save the Mueller Matrix
    fig.suptitle('Mueller Matrix histogram', fontsize=40, fontweight = 'bold', y = 1)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'MM_histogram.png'))
    plt.savefig(os.path.join(folder, 'MM_histogram.pdf'))
    plt.close()


def save_batch(folder: str, viz = False):
    """
    combine the 4 images (linear retardance, depolarization, azimuth and intensity in a single file)

    Parameters
    ----------
    folder : str
        the folder in which to find the images
    """
    if viz:
        names = load_combined_plot_name(viz = True)
        figures = load_filenames_combined_plot(viz = True)
    else:
        names = load_combined_plot_name()
        figures = load_filenames_combined_plot()

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