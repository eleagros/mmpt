import os

import json

import traceback
from tqdm import tqdm
import time

import traceback

from packaging.version import Version

from processingmm import utils
from processingmm import libmpMuelMat
from processingmm.addons import denoise_intensities, align_wavelengths, plot_polarimetry, azimuth_local_var, rotate_MM
import processingmm

def get_parameters(directories: list, calib_directory: str, wavelengths: list, parameter_set: str = 'default', PDDN_mode: str = 'both', 
                   PDDN_models_path: str = None, instrument: str = 'IMP', remove_reflection: bool = True, processing_mode: str = 'default',
                   run_all: bool = True, save_pdf_figs: bool = True, align_wls: bool =True, denoise_patch: bool = False) -> dict:
    """
    Returns the processing parameters in a dictionary format, readable by the next functions in the pipeline, for the given inputs.

    Parameters
    ----------
    directories : list
        the list of the directories in which the measurement folders are located
    calib_directory : str
        the path to the calibration directory
    wavelengths : list
        the list of the wavelengths to be processed
    parameter_set : str
        the name of the set of the parameters to be used for the visualization
    run_all : bool
        wether the processed folders should be reprocessed (default is False)
    PDDN_mode : str
        the PDDN mode ('no', 'pddn' or 'both', default is 'both')
    PDDN_models_path : str
        the path to the PDDN models (default is None, will redirect to the default path ./src/processingmm/PDDN_model/)
    processing_mode : str
        the processing mode ('full', 'default', or 'no_viz', default is 'default')
    time_mode : bool
        wether the processing times should be computed or not (default is False)
    save_pdf_figs : bool
        wether the pdf figures should be saved or not (default is True), takes about 10 seconds per folder
        
    Returns
    -------
    processing_parameters : dict
        the processing parameters in a dictionary format for the next functions in the pipeline

    Raises
    ------
    ValueError
        if the folder_measurement parameter is not the path to an existing folder
        if the calib_directory parameter is not the path to an existing folder (if img_kind is 'polarimetry')
        if the wavelenghts parameters contain unvalid wavelengths
        if PDDN_mode is not one of 'no', 'pddn' or 'both'
        if processing_mode is not one of 'full', 'no_visualization' or 'fast'
    """
    # check the input parameters are valid
    for directory in directories:
        if not os.path.isdir(directory):
            raise ValueError('The folder {} does not exist.'.format(directory))

    if not os.path.isdir(calib_directory):
        raise ValueError('The calib_directory parameter {} should be the path to an existing folder.'.format(calib_directory))

    if instrument not in {'IMP', 'IMPv2'}:
        raise ValueError('The list of instruments supported is ["IMP", "IMPv2"].')
    else:
        print(' [info] Processing data from the {} instrument.'.format(instrument))
        
    if instrument == 'IMPv2':
        if wavelengths != [630]:
            print(' [info] Switching wavelength selection from {} to 630nm.'.format(wavelengths))
            wavelengths = [630]
        if align_wls:
            align_wls = False
            print(' [info] Switching align_wls to False.')
        if remove_reflection:
            remove_reflection = False
            print(' [info] Switching remove_reflection to False (not supported).')
        if denoise_patch:
            print(' [info] Denoising is applied by patch, results might not be optimal.')
            
    elif instrument == 'IMP':
        if denoise_patch:
            denoise_patch = False
            print(' [info] Switching denoise_patch to False.')
            
    valid_wavelengths = utils.load_wavelengths(instrument)
    valid_wavelengths_num = []
    for wl in valid_wavelengths:
        valid_wavelengths_num.append(int(wl.split('nm')[0]))

    if wavelengths == 'all':
        wavelengths = valid_wavelengths_num
    else:
        for wavelength in wavelengths:
            if wavelength not in valid_wavelengths_num:
                raise ValueError('The wavelength {} is not a valid wavelength.'.format(wavelength))
        
    if PDDN_mode not in {'no', 'pddn', 'both'}:
        raise ValueError('PDDN_mode must be one of "no", "pddn", or "both".')
    
    if processing_mode not in {'full', 'default', 'no_viz'}:
        raise ValueError('processing_mode must be one of "full", "default", or "no_viz".')
        
    import processingmm
    if PDDN_models_path is None:
        PDDN_models_path = os.path.join(processingmm.__file__.split('__init__')[0], 'PDDN_model')
        
    if PDDN_mode in {'pddn', 'both'}:
        try:
            import torchvision
        except ImportError:
            raise ValueError('Please install the torchvision package to use PDDN.')
        
        models_found, missing_models = utils.test_pddn_models_existence(PDDN_models_path, instrument)
        if not models_found:
            print('Missing models:', missing_models)
            raise ValueError('The PDDN models are missing. Please check the path to the PDDN models or change PDDN_mode to no.')
    
    # create the processing parameters dictionnary
    processing_parameters = {'directories': directories,
                            'calib_directory': calib_directory,
                            'wavelengths': wavelengths,
                            'parameter_set': parameter_set,
                            'PDDN': PDDN_mode,
                            'PDDN_models_path': PDDN_models_path,
                            'run_all': run_all,
                            'processing_mode': processing_mode,
                            'time_mode': True,
                            'save_pdf_figs': save_pdf_figs,
                            'align_wls': align_wls,
                            'remove_reflection': remove_reflection,
                            'instrument': instrument,
                            'denoise_patch': denoise_patch} 
    
    return processing_parameters


def batch_process_master(parameters, folder_eu_time = {}):
    """
    Master function allowing to apply the mueller matrix processing pipeline to all the measurement folders located in one or multiple directories.

    Parameters
    ----------
    parameters : dict
        the processing parameters in a dictionary format
    remove_reflection : bool
        wether the reflection should be removed from the data (default is True)
    folder_eu_time : dict
        used to remap the times to the correct ones (for the measurements made in Chicago)
        
    Returns
    -------
    times : list
        the list of the processing times for each folder
    time_complete : float
        the total processing time
    
    Raises
    ------
    None
    """

    if parameters['PDDN'] in {'no', 'both'}:
        print('processing without PDDN...')
        times, time_complete = batch_process(parameters, PDDN = False, 
                                             folder_eu_time = folder_eu_time)
        print('processing without PDDN done.')
    
    if parameters['PDDN'] in {'pddn', 'both'}:
        assert Version(processingmm.__version__) >= Version('1.1'), ("Please update the processingmm package to version 1.1 or higher to use PDDN.")
                    
        print('Processing with PDDN...')
        times, time_complete = batch_process(parameters, PDDN = True, folder_eu_time = folder_eu_time)
        print('Processing with PDDN done.')
           
    times['total'] = time_complete 
    return times




def batch_process(parameters: dict, PDDN: bool = False, folder_eu_time: dict = {}):
    """
    apply the mueller matrix processing pipeline to all the measurement folders located in one or multiple directories.

    Parameters
    ----------
    parameters: dict
        the processing parameters in a dictionary format
    remove_reflection : bool
        wether the reflection should be removed from the data (default is True)
    PDDN : bool
        wether the PDDN should be used (default is False)
    folder_eu_time : dict
        used to remap the times to the correct ones (for the measurements made in Chicago)
    
    Returns
    -------
    times : list
        the list of the processing times for each folder
    time_complete : float
        the total processing time
    
    Raises
    ------
    None
    """        
    # get all the names of the measurement folders
    to_process, _ = utils.get_measurements_to_process(parameters, PDDN=PDDN)
        
    if PDDN:
        denoise_intensities.denoise_intensities(parameters, to_process)
    else:
        for folder in to_process:
            folder['polarimetry_fname'] = 'polarimetry'
            
    
    if parameters['align_wls']:
        align_wavelengths.align_wavelengths(parameters['directories'], parameters['PDDN'], False, parameters['wavelengths'])
        for folder in to_process:
            if os.path.exists(folder['path_intensite'].replace('.cod', '_aligned.cod')):
                folder['path_intensite'] = folder['path_intensite'].replace('.cod', '_aligned.cod')
            else:
                assert folder['wavelength'] != '600nm', "The wavelength 600nm is not aligned."
    
    start = time.time()
    
    for folder in tqdm(to_process):

        print('Processing:', folder['folder_name'])
    
        # Step 1: Organize the folders    
        # utils.move_folder_for_processing(parameters, folder)
        utils.reorganize_folders(folder, parameters['instrument'])

        calib_directory_dates_num = utils.get_calibration_dates(parameters)
        
        # Step 2: Compute the MM
        MM, calibration_directory, times, path_save = compute_one_MM(parameters, folder, calib_directory_dates_num, 
                                    folder_eu_time = folder_eu_time, Flag = False)
            
        # Step 3: Save the MM
        start_save_npz = time.time()
        os.makedirs(path_save, exist_ok = True)
        utils.save_file_as_npz(MM, os.path.join(path_save, "MM.npz"))
        end_save_npz = time.time()
        times['save_npz'] = end_save_npz - start_save_npz
            
        # Step 4: Visualize the MM
        start_viz = time.time()
        plot_polarimetry.visualize_MM(path_save, MM = MM, processing_mode = parameters.get('processing_mode', 'default'),
                                      save_pdf_figs = parameters.get('save_pdf_figs', False), instrument = parameters['instrument'])
        end_viz = time.time()
        times['viz'] = end_viz - start_viz
        
        
        parameters_reconstruction = {
            'processed': True,
            'calibration_directories': calibration_directory,
            'waveleghts': folder['wavelength'],
            'parameters': parameters,
            'libmpMuelMat': libmpMuelMat.__version__,
            'processingmm': processingmm.__version__
        }

        log_path = os.path.join(folder['folder_name'], 'MMProcessing.txt')
        try:
            with open(log_path, 'w') as logbook:
                json.dump(parameters_reconstruction, logbook, indent=3)
        except Exception as e:
            traceback.print_exc()
            
    end = time.time()
    if len(to_process) == 0:
        time_complete = 0
    else:
        time_complete = (end - start)/len(to_process)
    
    try:
        return times, time_complete
    except UnboundLocalError:
        return {}, time_complete
    


def compute_one_MM(parameters, measurement, calib_directory_dates_num: list, folder_eu_time: dict = {}, 
                   pbar = None, Flag = False):
    """
    compute_one_MM is a function that computes the MM for the folders in c

    Parameters
    ----------
    measurements_directory : str
        the path to the directory containing the measurements
    calib_directory_dates_num : list of datetime
        a list containing the dates of the calibration folders
    calib_directory : str
        the path to the calibration directory
    MuellerMatrices : dict
        the dictionnary containing the computed Mueller Matrices
    treshold : int
        the number of files that should be found in the folder to consider it processed
    c : str
        the folder name
    remove_reflection : boolean
        indicates wether or not the reflections of the image should be removed (default: True)
    pbar : progress bar
        the progress bar that should (or not) be processed (default None)
    Flag : boolean
        boolean indicating if the warnings should be displayed (default: False)
    wavelengths : list
        the list of wavelengths that should be processed 
    
    Returns
    -------
    calibration_directory_closest : str
        the path to the calibration folder that will be used
    """    
    path = measurement['folder_name']
    wavelength = measurement['wavelength']

    processing_mode = parameters.get('processing_mode', 'default')
    remove_reflection = parameters.get('remove_reflection', True)
    
    angle_correction = utils.get_angle_correction(measurement['folder_name'])
    
    # Find the closest calibration directory based on the wavelength
    calibration_directory_closest = utils.get_calibration_directory(parameters, calib_directory_dates_num, path, wavelength, folder_eu_time, Flag)
    calibration_directory_wl = os.path.join(calibration_directory_closest, wavelength)
    
    start_full_processing = time.time()
            
    # Load calibration & intensity data
    A, W = utils.load_calibration_data(parameters, calibration_directory_wl, wavelength)
    # I, polarimetry_fname = utils.get_intensity(path, wavelength, align_wls, PDDN)
    I = utils.get_intensity(parameters, measurement['path_intensite'])
    
    time_data_loading = time.time() - start_full_processing
    
    start_processing = time.time()  
    MM, dilated_mask = utils.process_mm(I, remove_reflection, A, W)
    if dilated_mask is not None:
        MM['dilated_mask'] = dilated_mask
    else:
        MM['dilated_mask'] = None
    time_MM_processing = time.time() - start_processing
    
    start_processing = time.time()  
    # remove the NaNs from the atzimuth measurements
    MM['azimuth'], MM['azimuth_curation'] = utils.curate_azimuth(MM['azimuth'], f"{path}/{measurement['polarimetry_fname']}")
    MM['M11'] = utils.correct_M11(MM['M11'], parameters['instrument'])
    MM['M11_normalized'] = utils.normalize_M11(MM['M11'], parameters['instrument'])
    time_azimuth_curation = time.time() - start_processing
    
    start_processing = time.time()
    azimuth_std = azimuth_local_var.get_azimuth_local_var(azimuth = MM['azimuth'], patch_size = 5)
    MM['azimuth_local_var'] = azimuth_std
    time_azimuth_std_processing = time.time() - start_processing

    parameter_names = utils.load_parameter_names(processing_mode)

    # Remove keys from MM that are not in parameter_names
    MM = {key: value for key, value in MM.items() if key in parameter_names}

    MM = rotate_MM.apply_angle_correction(MM, angle_correction)
        
    time_full_processing = time.time() - start_full_processing
    
    if pbar is None:
        pass
    else:
        pbar.update(1)
    
    times = {'azimuth_curation': time_azimuth_curation, 'data_loading': time_data_loading, 
             'MM_processing': time_MM_processing, 'azimuth_std_processing': time_azimuth_std_processing,
             'full_processing': time_full_processing}
    return MM, calibration_directory_closest, times, os.path.join(path, measurement['polarimetry_fname'], f"{wavelength}")



