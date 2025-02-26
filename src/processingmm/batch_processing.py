import os
import json
import pandas as pd
import shutil 
import traceback
from tqdm import tqdm
import traceback
from packaging.version import Version

from processingmm import utils
from processingmm import MM_processing, plot_polarimetry
from processingmm import libmpMuelMat
import processingmm
import time

from processingmm.addons import visualization_lines
from processingmm.multi_img import multi_img_processing, reorganize_folders
from processingmm.MM_processing import get_intensity
from processingmm import libmpMPIdenoisePDDN

def get_parameters(directories: list, calib_directory: str, wavelengths: list, parameter_set: str = 'TheoniPics', PDDN_mode: str = 'both', 
                   processing_mode: str = 'default', run_all: bool = True, save_pdf_figs: bool = True, align_wls: bool =True) -> dict:
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

    valid_wavelengths = utils.load_wavelengths()
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

    # create the processing parameters dictionnary
    processing_parameters = {'directories': directories,
                            'calib_directory': calib_directory,
                            'wavelengths': wavelengths,
                            'parameter_set': parameter_set,
                            'PDDN': PDDN_mode,
                            'run_all': run_all,
                            'processing_mode': processing_mode,
                            'time_mode': True,
                            'temp_folder': './temp_processing',
                            'save_pdf_figs': save_pdf_figs,
                            'align_wls': align_wls} 
    
    return processing_parameters


def batch_process_master(parameters, remove_reflection = True, folder_eu_time = {}):
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
        times, time_complete = batch_process(parameters, remove_reflection = remove_reflection, PDDN = False, folder_eu_time = folder_eu_time)
        print('processing without PDDN done.')
    
    if parameters['PDDN'] in {'pddn', 'both'}:
        assert Version(processingmm.__version__) >= Version('1.1'), ("Please update the processingmm package to version 1.1 or higher to use PDDN.")
        print('processing with PDDN...')
        times, time_complete = batch_process(parameters, remove_reflection = remove_reflection, PDDN = True, folder_eu_time = folder_eu_time)
        print('processing with PDDN done.')
           
    times['total'] = time_complete 
    return times


def batch_process(parameters: dict, remove_reflection: bool = True, PDDN: bool = False, folder_eu_time: dict = {}):
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
    # start recording time for the whole processing
    start = time.time()
    
    PDDN_models = {}
    if PDDN:
        print('Loading PDDN models...')
        for wavelength in parameters['wavelengths']:
            path_model = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        r'PDDN_model/PDDN_model_' + str(wavelength) + '_Fresh_HB.pt')
            if os.path.isfile(path_model):
                print(path_model)
                PDDN_models[wavelength] = libmpMPIdenoisePDDN.MPI_PDDN(path_model)
        print('Loading PDDN models done.\n')
    else:
        PDDN_models = None
    
    # get all the names of the measurement folders
    to_process, wavelenghts, dfProcessing = utils.get_to_process(parameters, PDDN=PDDN)
    
    if PDDN:
        print('Denoising inference started...')
        for folder in tqdm(to_process):
            for wavelength, model in PDDN_models.items():
                pathDenoisedCod = os.path.join(f"{folder}/raw_data", f"{wavelength}nm", f"{wavelength}_Intensite_PDDN.cod")
                if os.path.exists(pathDenoisedCod):
                    pass
                else:
                    I, _ = get_intensity(f"{folder}/raw_data", str(wavelength), parameters['align_wls'], False)
                    I, _ = model.Denoise(I)
                    libmpMuelMat.write_cod_data_X3D(I, os.path.join(f"{folder}/raw_data", 
                                                                    f"{wavelength}nm", f"{wavelength}_Intensite_PDDN.cod"), VerboseFlag=1)
        print('Denoising inference done.')
    
    if parameters['align_wls']:
        
        print('Aligning wavelengths...')
        from processingmm.addons import align_wavelengths
        directories = parameters['directories']
        PDDN_mode = parameters['PDDN']
        run_all = False
        align_wavelengths.align_wavelenghts(directories, PDDN_mode, run_all)
        print('Aligning wavelengths done.\n')
        

    # get the different chunks that are used to split the processing of the data
    for folder in tqdm(to_process):
        
        print('Processing:', folder)
        
        to_process_temp = []
        shutil.rmtree(parameters['temp_folder'], ignore_errors=True)
        os.makedirs(parameters['temp_folder'], exist_ok=True)
        
        start_move = time.time()
        links_folders, to_process_temp = utils.moveTheFoldersForProcessing(parameters, folder)
        end = time.time()
        
        # process the mueller matrix and generate the visualizations
        calibration_directories, parameters_set, times = process_MM(parameters, folder, PDDN = PDDN, remove_reflection = remove_reflection,
                                                                    wavelengths = wavelenghts, folder_eu_time = folder_eu_time,
                                                                    dfProcessing = dfProcessing)

        times['moving'] = end - start_move
        
        for folder in to_process_temp:
            parameters_reconstruction = {
                'processed': True,
                'calibration_directories': calibration_directories[folder.split('/')[-1]],
                'parameters': parameters,
                'libmpMuelMat': libmpMuelMat.__version__,
                'processingmm': processingmm.__version__
            }
            try:
                logbook_MM_processing = open(os.path.join(folder, 'MMProcessing.txt'), 'w')
                logbook_MM_processing.write(json.dumps(parameters_reconstruction, indent = 3))
                logbook_MM_processing.close()
            except:
                logbook_MM_processing.close()
                traceback.print_exc()

        # put back the folders in the original folder
        links_folders = {v: k for k, v in links_folders.items()}
        for folder, temp_folder in links_folders.items():
            if PDDN:
                to_remove = ['polarimetry', 'polarimetry_PDDN']
            else:
                to_remove = ['polarimetry_PDDN', 'polarimetry']
                    
            for fold in os.listdir(temp_folder):
                                
                if fold in {to_remove[0], 'annotation', 'histology', 'rotation_MM.txt'}:
                    pass
                elif fold == 'MMProcessing.txt':
                    shutil.copy(os.path.join(temp_folder, fold), os.path.join(folder, fold))
                elif fold == to_remove[1] or fold == 'raw_data':
                    shutil.copytree(os.path.join(temp_folder, fold), os.path.join(folder, fold), dirs_exist_ok=True) 
                else:
                    raise ValueError('The folder {} is not a valid folder.'.format(fold))
            
        shutil.rmtree(parameters['temp_folder'], ignore_errors=True)

    end = time.time()
    if len(to_process) == 0:
        time_complete = 0
    else:
        time_complete = (end - start)/len(to_process)
    
    try:
        return times, time_complete
    except UnboundLocalError:
        return {}, time_complete
    
    
        
def process_MM(parameters: dict, folder: str, PDDN: bool, remove_reflection: bool, wavelengths: list, folder_eu_time: dict, 
               dfProcessing: pd.DataFrame):
    """
    master function allowing to reogranize the folders, compute the MMs, generate the plots and the visualizations for one directory

    Parameters
    ----------
    measurement_directory : str
        the path to the directory in which the measurement are located
    calib_directory : str
        the path to the directory in which the calibration data is located
        
    Returns
    -------
    calibration_directories : list
        the list of the calibration folders used to process the data (used to save it for tracability purposes)
    parameters_set : str
        the name of the parameters_set used for the line visualization
    """
    measurement_directory = parameters['temp_folder']
    
    # obtain the directories that we should reorganize the data for
    all_directories = os.listdir(measurement_directory)
    reorganize_folders.reorganizeFolders(measurement_directory, all_directories)
    
    """to_compute = multi_img_processing.remove_already_computed_folders(measurement_directory, 
                                                                      run_all = parameters['run_all'], 
                                                                      PDDN = PDDN,
                                                                      wavelengths = wavelengths,
                                                                      save_pdf_figs = parameters['save_pdf_figs'],
                                                                      Flag = False)"""

    
    to_compute = []
    for fld in dfProcessing[dfProcessing['folder name'] == folder].index:
        if dfProcessing.loc[fld, 'data presence'] and not dfProcessing.loc[fld, 'processed']:
            to_compute.append([dfProcessing.loc[fld, 'folder name'].split('/')[-1],
                            dfProcessing.loc[fld, 'wavelength']])

    calib_directory_dates_num = multi_img_processing.get_calibration_dates(parameters['calib_directory'])

    # compute the MMs
    MuellerMatrices, calibration_directories, times = MM_processing.compute_analysis_python(measurement_directory, 
                                        calib_directory_dates_num, parameters['calib_directory'], to_compute,
                                        remove_reflection = remove_reflection, folder_eu_time = folder_eu_time, 
                                        run_all = parameters['run_all'], batch_processing = True, Flag = False, PDDN = PDDN,
                                        wavelengths = wavelengths, processing_mode = parameters['processing_mode'], time_mode = parameters['time_mode'],
                                        save_pdf_figs = parameters['save_pdf_figs'], align_wls = parameters['align_wls'])
    
    MuellerMatrices_raw = MuellerMatrices

    start = time.time()

    # and generate the different plots
    if parameters['processing_mode'] == 'full':
        for folder, _ in MuellerMatrices.items():
            plot_polarimetry.parameters_histograms(MuellerMatrices_raw, folder ,save_pdf_figs = parameters['save_pdf_figs'])
            _ = plot_polarimetry.show_MM(MuellerMatrices[folder]['nM'], folder, save_pdf_figs = parameters['save_pdf_figs'])
            plot_polarimetry.MM_histogram(MuellerMatrices, folder, save_pdf_figs = parameters['save_pdf_figs'])
        
    if parameters['processing_mode'] != 'no_viz':
        for folder, _ in MuellerMatrices.items():
            plot_polarimetry.generate_plots(MuellerMatrices, folder, save_pdf_figs = parameters['save_pdf_figs'])
            if parameters['processing_mode'] == 'full':
                plot_polarimetry.save_batch(folder)
        
    end = time.time()
    time_plotting = end - start
    times['plotting'] = time_plotting
    
    return calibration_directories, parameters['parameter_set'], times



def batch_visualization(parameters):
    
    start = time.time()
    
    _, _, df = utils.get_to_process(parameters)
    to_process = df[df['data presence']]
    to_process = list(set(list(df.reset_index(level=0).apply(lambda row: row['folder name'], axis=1))))
        
    for wl in parameters['wavelengths']:
        if parameters['PDDN'] in {'no', 'both'}:
            _ = visualization_lines.visualization_auto(to_process, parameters['parameter_set'], run_all = parameters['run_all'], 
                                                        batch_processing = False, PDDN = False, wavelengths = [wl], 
                                                        save_pdf_figs = parameters['save_pdf_figs'])
        if parameters['PDDN'] in {'pddn', 'both'}:
            _ = visualization_lines.visualization_auto(to_process, parameters['parameter_set'], run_all = parameters['run_all'], 
                                                        batch_processing = False, PDDN = True, wavelengths = [wl], 
                                                        save_pdf_figs = parameters['save_pdf_figs'])
    
    end = time.time()
    time_plotting = end - start
    return time_plotting/len(to_process) if len(to_process) > 0 else 0