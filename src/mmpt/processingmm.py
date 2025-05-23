import os

import json

import numpy as np
import traceback
from tqdm import tqdm
import time

import matplotlib.pyplot as plt

import torch

import traceback

from packaging.version import Version
from omegaconf import OmegaConf

from mmpt import utils, libmpMuelMat, libmpMPIdenoisePDDN
from mmpt.addons import align_wavelengths, plot_polarimetry, azimuth_local_var, rotate_MM
from mmpt.addons.polarpred.mm.models import init_mm_model
# align the measurements captured at different wavelengths - could cause issue when using masks obtained
# at 550nm as the images are slightly shifted 

import mmpt

class MuellerMatrixProcessor:
    def __init__(self, data_source, wavelengths = [], input_dirs = '', calib_dir = '', PDDN_mode='both', 
                 PDDN_models_path=None, instrument='IMP', remove_reflection=True, workflow_mode='default',
                 force_reprocess=True, save_pdf_figs=True, align_wls=True, denoise_patch=False, binning_factor = 1,
                 mm_computation_backend='c', lu_chipman_backend = 'processing', prediction_mode = 'MM', folder_eu_time={}):
        self.data_source = data_source
        self.input_dirs = input_dirs
        self.calib_dir = calib_dir
        self.wavelengths = wavelengths
        self.visualization_preset = 'default'
        self.PDDN_mode = PDDN_mode
        self.PDDN_models_path = PDDN_models_path
        self.instrument = instrument
        self.remove_reflection = remove_reflection
        self.workflow_mode = workflow_mode
        self.force_reprocess = force_reprocess
        self.save_pdf_figs = save_pdf_figs
        self.align_wls = align_wls
        self.denoise_patch = denoise_patch
        self.binning_factor = binning_factor
        self.mm_computation_backend = mm_computation_backend
        self.lu_chipman_backend = lu_chipman_backend
        self.prediction_mode = prediction_mode
        self.folder_eu_time = folder_eu_time
        self.test_mode = False
        
        self.validate_inputs()
        self.prepare_parameters()
        
        if self.PDDN_mode in {'pddn', 'both'}:
            self.load_PDDN_models()
            
        if self.mm_computation_backend == 'torch':
            self.load_mm_model()
        else:
            self.mm_model = None

        if self.prediction_mode is None:
            print(' [info] No prediction model selected.')
        else:
            print(' [info] Loading tumor prediction model...')
            self.load_prediction_model()
            print(' [info] Loading tumor prediction model done.')
        
        if self.lu_chipman_backend == 'prediction':
            print(' [info] Loading lu-chipman prediction model...')
            self.load_lu_chipman_prediction_model()
            print(' [info] Loading lu-chipman prediction model done.')
        else:
            self.lu_chipman_model = None
        
        self.get_calib_dates()
        
    def validate_inputs(self):
        """Validates input parameters."""
        if self.data_source not in {'offline', 'online'}:
            raise ValueError('data_source must be "offline" or "online".')
        
        if self.data_source == 'offline':
            for directory in self.input_dirs:
                if not os.path.isdir(directory):
                    raise ValueError(f'The folder {directory} does not exist.')

            if not os.path.isdir(self.calib_dir) and self.instrument == 'IMP':
                raise ValueError(f'The calib_directory parameter {self.calib_dir} should be an existing folder.')
            elif self.instrument == 'IMPv2':
                print(f' [info] Calibration directory {self.calib_dir} will not be used.')

        if self.instrument not in {'IMP', 'IMPv2'}:
            raise ValueError('Supported instruments: ["IMP", "IMPv2"].')

        valid_wavelengths = utils.load_wavelengths(self.instrument)
        valid_wavelengths_num = [int(wl.split('nm')[0]) for wl in valid_wavelengths]
        
        if self.mm_computation_backend not in {'c', 'torch'}:
            raise ValueError('mm_computation_backend must be "c" or "torch".')
        
        if self.lu_chipman_backend not in {'processing', 'prediction'}:
            raise ValueError('lu_chipman_backend must be "processing" or "prediction".')
            
        if self.wavelengths == 'all':
            self.wavelengths = valid_wavelengths_num
        else:
            for wavelength in self.wavelengths:
                if wavelength not in valid_wavelengths_num:
                    raise ValueError(f'The wavelength {wavelength} is not valid.')
        
        if self.PDDN_mode not in {'no', 'pddn', 'both'}:
            raise ValueError('PDDN_mode must be "no", "pddn", or "both".')
        
        if self.workflow_mode not in {'full', 'default', 'no_viz'}:
            raise ValueError('processing_mode must be "full", "default", or "no_viz".')
        
        if self.binning_factor < 1:
            raise ValueError('The binning factor should be greater or equal than 1.')
        
        if self.PDDN_models_path is None:
            self.PDDN_models_path = os.path.join(mmpt.__file__.split('__init__')[0], 'PDDN_model')
        
        if self.instrument == 'IMP':
            self.binning_factor = 1
            print(' [info] The binning factor is set to 1 for IMP.')
            
        if self.PDDN_mode in {'pddn', 'both'}:
            try:
                import torchvision
            except ImportError:
                raise ValueError('Please install the torchvision package to use PDDN.')
            
            models_found, missing_models = utils.test_pddn_models_existence(self.PDDN_models_path, self.instrument)
            if not models_found:
                print('Missing models:', missing_models)
                raise ValueError('The PDDN models are missing. Please check the path or set PDDN_mode to "no".')

            
    def prepare_parameters(self):
        """Prepares processing parameters."""
        if self.instrument == 'IMPv2':
            if self.wavelengths != [630]:
                print(f' [info] Switching wavelength selection to 630nm.')
                self.wavelengths = [630]
            if self.align_wls:
                self.align_wls = False
                print(' [info] Switching align_wls to False.')
            if self.remove_reflection:
                self.remove_reflection = False
                print(' [info] Switching remove_reflection to False.')
        elif self.instrument == 'IMP' and self.denoise_patch:
            self.denoise_patch = False
            print(' [info] Switching denoise_patch to False.')

    def __str__(self):
        """Returns a formatted string representation of the processing parameters."""
        params = self.get_parameters()
        return "\nProcessing parameters:\n" + "\n".join([f"{key}: {value}" for key, value in params.items()])
    
    def get_parameters(self):
        """Returns the processing parameters as a dictionary."""
        return {
            'data_source': self.data_source,
            'instrument': self.instrument,
            
            'input_dirs': self.input_dirs,
            'calib_dir': self.calib_dir,

            
            'wavelengths': self.wavelengths,
            'align_wls': self.align_wls,
            
            'PDDN': self.PDDN_mode,
            'PDDN_models_path': self.PDDN_models_path,
            'denoise_patch': self.denoise_patch,
            
            'visualization_preset': self.visualization_preset,
            'workflow_mode': self.workflow_mode,
            'save_pdf_figs': self.save_pdf_figs,
            
            'force_reprocess': self.force_reprocess,
            'time_mode': True,
            
            'remove_reflection': self.remove_reflection,
            
            'binning_factor': self.binning_factor,
            
            'mm_computation_backend': self.mm_computation_backend,
            'lu_chipman_backend': self.lu_chipman_backend,
            'folder_eu_time': self.folder_eu_time,
        }

    def load_PDDN_models(self):
        """Loads the PDDN models for the given wavelengths."""    
        PDDN_models = {}

        print('\n [info] Loading PDDN models...')
        for wavelength in self.wavelengths:
            path_model = os.path.join(self.PDDN_models_path, 'PDDN_model_' + str(wavelength) + '_Fresh_HB.pt')
            if os.path.isfile(path_model):
                PDDN_models[wavelength] = libmpMPIdenoisePDDN.MPI_PDDN(path_model)
                    
        assert len(PDDN_models) > 0, ("Problem when loading the PDDN models. Do they exist? They should be located in ./src/mmpt/PDDN_model/")
        print(' [info] Loading PDDN models done.')

        self.PDDN_models = PDDN_models

    def load_lu_chipman_prediction_model(self):
        # Load the trained model (update file name as needed).
        """model_path = os.path.join(utils.getLuChipmanPredPath(), 'model', 'full_decomposition', 'model_brain_nn.keras')
        self.lu_chipman_model = models.load_model(model_path)"""
        model_path = os.path.join(utils.getLuChipmanPredPath(), 'model', 'predict_lu_chipman.pth')
        self.lu_chipman_model = torch.load(model_path, weights_only=False).to('cuda')
        
    def load_mm_model(self):
        cfg = OmegaConf.load(os.path.join(utils.getPolarPredPath(), 'configs/train_local.yml'))
        cfg = OmegaConf.merge(cfg, OmegaConf.load(os.path.join(utils.getPolarPredPath(), 'configs/test.yml')))
        cfg.MM = True
        print(' [info] Loading MM model...')
        self.mm_model = init_mm_model(cfg, train_opt=False)
        print(' [info] Loading MM model done.')
        
    def get_calib_dates(self):
        if self.instrument == 'IMP':
            self.calib_directory_dates_num = utils.get_calibration_dates(self.get_parameters())
        else:
            self.calib_directory_dates_num = []
            
    def load_prediction_model(self):
        cfg = OmegaConf.load(os.path.join(utils.getPolarPredPath(), 'configs/train_local.yml'))
        cfg = OmegaConf.merge(cfg, OmegaConf.load(os.path.join(utils.getPolarPredPath(), 'configs/test.yml')))
        
        if self.prediction_mode not in {'MM', 'LuChipman'}:
            raise ValueError('prediction_mode must be "MM" or "LuChipman".')
        
        cfg.MM = self.prediction_mode == 'MM'
        # model selection
        self.mm_model_prediction = init_mm_model(cfg, train_opt=False)
        self.prediction_model, self.model_path = utils.load_model(self.mm_model_prediction, cfg)
            
    
    def batch_process_master(self):
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

        if self.data_source == 'online':
            raise ValueError('The data_source is set to "online". The function to process the data online is online_processing.')
            
        if self.PDDN_mode in {'no', 'both'}:
            print('processing without PDDN...')
            times, time_complete = self.batch_process(PDDN = False)
            print('processing without PDDN done.')
        
        if self.PDDN_mode in {'pddn', 'both'}:
            assert Version(mmpt.__version__) >= Version('1.1'), ("Please update the mmpt package to version 1.1 or higher to use PDDN.")
                        
            print('Processing with PDDN...')
            times, time_complete = self.batch_process(PDDN = True)
            print('Processing with PDDN done.')
        times['total'] = time_complete 
        return times

    def batch_process(self, PDDN: bool = False):
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
        self.to_process, _ = utils.get_measurements_to_process(self.get_parameters(), PDDN=PDDN)
        
        if PDDN:
            print(' [info] Denoising intensities live.')
            for folder in self.to_process:
                folder['polarimetry_fname'] = 'polarimetry_PDDN'
            # denoise_intensities.denoise_intensities(self.get_parameters(), self.PDDN_models, self.to_process)
        else:
            for folder in self.to_process:
                folder['polarimetry_fname'] = 'polarimetry'
        
        if self.align_wls:
            align_wavelengths.align_wavelengths(self.input_dirs, self.PDDN_mode, False, self.wavelengths)
            for folder in self.to_process:
                if os.path.exists(folder['path_intensite'].replace('.cod', '_aligned.cod')):
                    folder['path_intensite'] = folder['path_intensite'].replace('.cod', '_aligned.cod')
                else:
                    assert folder['wavelength'] != '600nm', "The wavelength 600nm is not aligned."        

        start = time.time()
        
        for folder in tqdm(self.to_process):

            print('Processing:', folder['folder_name'])

            try:
                torch.cuda.empty_cache()
            except:
                pass

            # Step 1: Organize the folders    
            # utils.move_folder_for_processing(parameters, folder)
            utils.reorganize_folders(folder, self.instrument)
            
            measurement, time_data_loading = self.load_input_data(folder)
            if PDDN:
                times_denoising = {}
                for wavelength, model in self.PDDN_models.items():
                    if folder['wavelength'].replace('nm', '') == str(wavelength):
                        measurement['I'], denoise_times = model.Denoise(measurement['I'])
                        times_denoising[wavelength] = denoise_times[1]
                        continue
            else:
                times_denoising = {}
                            
            # regorganize the output folder, for the test mode
            if self.test_mode:
                path_save_test_mode = measurement['path_save'].replace(f'{os.sep}{measurement['polarimetry_fname']}{os.sep}', f'{os.sep}test_backends{os.sep}')
                os.makedirs(os.sep.join(path_save_test_mode.split(os.sep)[:-1]), exist_ok = True)
                os.makedirs(path_save_test_mode, exist_ok = True)
                path_save_test_mode = os.path.join(path_save_test_mode, f"{self.mm_computation_backend}_{self.lu_chipman_backend}")
                os.makedirs(path_save_test_mode, exist_ok = True)
                print(' [info] Test mode: saving to', path_save_test_mode)
                measurement['path_save'] = path_save_test_mode
            
            # Step 2: Compute the MM
            # only required inputs (I, A, W)
            measurement['MM'], times_compute_mm = compute_and_curate_mm(measurement['I'], measurement['A'], measurement['W'],
                                  self.mm_computation_backend, self.lu_chipman_backend, self.mm_model, self.lu_chipman_model,
                                  self.remove_reflection, measurement['path_save'], self.instrument, self.workflow_mode,
                                  self.angle_correction, measurement['folder_name'])
                        
            # Step 3: Save the MM
            start_save_npz = time.time()
            os.makedirs(measurement['path_save'], exist_ok = True)
            utils.save_file_as_npz(measurement['MM'], os.path.join(measurement['path_save'], "MM.npz"))
            end_save_npz = time.time()
            time_save_npz = end_save_npz - start_save_npz
                
            # Step 4: Visualize the MM
            start_viz = time.time()
            plot_polarimetry.visualize_MM(measurement['path_save'], MM = measurement['MM'],
                                          processing_mode = self.workflow_mode, save_pdf_figs=self.save_pdf_figs,
                                          instrument=self.instrument, mm_processing=self.mm_computation_backend)
            end_viz = time.time()
            time_viz = end_viz - start_viz
            
            times = {'data_loading': time_data_loading, 'times_denoising': times_denoising, 'processing_and_curation': times_compute_mm,
                     'save_npz_file': time_save_npz, 'visualization': time_viz}
            
            log_path = os.path.join(folder['folder_name'], 'MMProcessing.txt')
            try:
                if os.path.exists(log_path):  # Check if file exists
                    with open(log_path, 'r') as logbook:
                        parameters_reconstruction = json.load(logbook)  # Read JSON content
                        if 'processed' in parameters_reconstruction:
                            parameters_reconstruction = {}
                else:
                    parameters_reconstruction = {}
            except:
                parameters_reconstruction = {}
            
            parameters_reconstruction[folder['wavelength']] = {
                'processed': True,
                'calibration_directories': measurement['calibration_directory'],
                'wavelenght': folder['wavelength'],
                'parameters': self.get_parameters(),
                'libmpMuelMat': libmpMuelMat.__version__,
                'mmpt': mmpt.__version__
            }

            try:
                parameters_reconstruction_save = parameters_reconstruction.copy()
                parameters_reconstruction_save[folder['wavelength']]['parameters'].pop("folder_eu_time", None)
                with open(log_path, 'w') as logbook:
                    json.dump(parameters_reconstruction_save, logbook, indent=3)
            except Exception as e:
                traceback.print_exc()
                
        end = time.time()
        if len(self.to_process) == 0:
            time_complete = 0
        else:
            time_complete = (end - start)/len(self.to_process)
        
        try:
            return times, time_complete
        except UnboundLocalError:
            return {}, time_complete
    
    
    def load_input_data(self, measurement):
        path = measurement['folder_name']
        wavelength = measurement['wavelength']
        if self.instrument == 'IMP':
            measurement['path_save'] = os.path.join(path, measurement['polarimetry_fname'], f"{wavelength}")
        else:
            measurement['path_save'] = os.path.join(path, measurement['polarimetry_fname'], 
                    measurement['path_intensite'].split(os.sep)[-1].replace('.npy', '').replace('PDDN_', ''), f"{wavelength}")
        
        self.angle_correction = utils.get_angle_correction(measurement['folder_name'])
        
        # Find the closest calibration directory based on the wavelength
        if self.instrument == 'IMP':
            calibration_directory_closest = utils.get_calibration_directory(self.get_parameters(), self.calib_directory_dates_num, 
                                                                            path, wavelength, self.folder_eu_time, Flag = False)
            calibration_directory_wl = os.path.join(calibration_directory_closest, wavelength)
        else:
            calibration_directory_closest, calibration_directory_wl = None, None
                
        measurement['calibration_directory'] = calibration_directory_closest
        measurement['calibration_directory_wl'] = calibration_directory_wl
        
        start_data_loading = time.time()
        # Load calibration & intensity data
        A, W = utils.load_calibration_data(self.get_parameters(), measurement, calibration_directory_wl, wavelength)
        measurement['A'] = A
        measurement['W'] = W
        
        # I, polarimetry_fname = utils.get_intensity(path, wavelength, align_wls, PDDN)
        I = utils.get_intensity(self.get_parameters(), measurement['path_intensite'])
        measurement['I'] = I
        time_data_loading = time.time() - start_data_loading

        if self.binning_factor == 1:
            pass
        elif self.binning_factor > 1:
            measurement['I'] = utils.bin_pixels(I, self.binning_factor)
            measurement['A'] = utils.bin_pixels(A, self.binning_factor)
            measurement['W'] = utils.bin_pixels(W, self.binning_factor)
        elif self.binning_factor < 1:
            raise ValueError('The binning factor should be greater or equal than 1.')
        
        return measurement, time_data_loading
    
    def online_processing(self, I, A, W, binning_factor = 4):
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
        if self.data_source == 'offline':
            raise ValueError('The data_source is set to "offline". The function to process the data offline is batch_process_master.')
        
        if self.PDDN_mode in {'no', 'both'}:
            print(' [info] Processing without PDDN')
            PDDN = False
        else:
            print(' [info] Processing with PDDN')
            PDDN = True
        
        start_binning = time.time()
        if binning_factor == 1:
            pass
        elif binning_factor > 1:
            I = utils.bin_pixels(I, binning_factor)
            A = utils.bin_pixels(A, binning_factor)
            W = utils.bin_pixels(W, binning_factor)
        elif binning_factor < 1:
            raise ValueError('The binning factor should be greater or equal than 1.')
        time_binning = time.time() - start_binning
        
        start_denoising = time.time()
        if PDDN:
            I, _ = self.PDDN_models[self.wavelengths[0]].Denoise(I)
        time_denoising = time.time() - start_denoising
        
        MM, time_processing = compute_mm(I, A, W, self.mm_computation_backend, self.lu_chipman_backend, 
                                         self.mm_model, self.lu_chipman_model, self.remove_reflection)
        
        MM['Intensity'] = I
          
        return MM, {'time_processing': time_processing, 'time_binning': time_binning, 'time_denoising': time_denoising}

        
    def prediction(self, MM, I):
        
        times = {}
        start = time.time()
        input = self.mm_model_prediction(None, MM = torch.tensor(MM).cuda(), I = torch.tensor(I).cuda())[:, :-1]
                
        start_predict = time.time()
        preds = utils.predict(self.prediction_model, input)
        times['predict'] = time.time() - start_predict
                
        """start_save = time.time()
        save_results.save_predictions(preds, input, sample, mode = mode, model_path = model_path, path_intensite = path_intensite)
        times['save'].append(time.time() - start_save)"""
        times['total'] = time.time() - start
        
        return preds, input, times
    
    def compare_backends(self, mm_computation_backends, lu_chipman_backends):
        """Compare the backends for the MM computation and lu-chipman prediction."""
        import itertools
        combinations = list(itertools.product(mm_computation_backends, lu_chipman_backends))

        self.test_mode = True
        self.force_reprocess = True
        self.workflow_mode = 'no_viz'
        self.load_mm_model()
        self.load_lu_chipman_prediction_model()
        
        for combo in combinations:
            self.mm_computation_backend = combo[0]
            self.lu_chipman_backend = combo[1]
            print(f"\nmm_computation_backend: {self.mm_computation_backend}, lu_chipman_backend: {self.lu_chipman_backend}")
            self.batch_process_master()
            print(f"Done.")
            
        plot_polarimetry.visualize_comparison(self, mm_computation_backends, lu_chipman_backends)
        
    def get_online_test_data(self):
        """Get the online test data."""
        # example with a test case
        if self.instrument == 'IMP':
            path_data = os.path.join(utils.getTestPath(), 'data_IMP', '2022-11-02_T_HORAO-59-AF_FR_S1_1/raw_data/550nm/550_Intensite.cod')
            I = libmpMuelMat.read_cod_data_X3D(path_data, isRawFlag=1)
            path_calib = os.path.join(utils.getTestPath(), 'calib', '2022-11-08_C_1', '550nm')
            A = libmpMuelMat.read_cod_data_X3D(os.path.join(path_calib, '550_A.cod'), isRawFlag=0)
            W = libmpMuelMat.read_cod_data_X3D(os.path.join(path_calib, '550_W.cod'), isRawFlag=0)
        else:
            path_data = os.path.join(utils.getTestPath(), 'data_IMPv2', '2025-03-24_152306_F_FF_HORAO_0005_003', 'to_process')
            I = np.load(os.path.join(path_data, '630_Image_Number_1.npy'))
            A = np.load(os.path.join(path_data, 'A.npy'))
            W = np.load(os.path.join(path_data, 'W.npy'))
        
        return I, A, W
            
def compute_and_curate_mm(I, A, W, mm_computation_backend = 'c', lu_chipman_backend = 'processing', 
                          mm_model = None, lu_chipman_model = None, remove_reflection = False, path_save = None,
                          instrument = 'IMP', workflow_mode = 'default', angle_correction = 0, folder_name = None):
    MM, time_MM_processing = compute_mm(I, A, W, mm_computation_backend, lu_chipman_backend, mm_model, lu_chipman_model, 
                                        remove_reflection)
    MM, times_curate_mm = curate_mm(MM, path_save, instrument, workflow_mode, angle_correction, folder_name = folder_name)
    return MM, {'mm_processing': time_MM_processing, 'mm_curation': times_curate_mm,}

def compute_mm(I, A, W, mm_computation_backend = 'c', lu_chipman_backend = 'processing', mm_model = None, lu_chipman_model = None,
                   remove_reflection = False):
    """
    compute_one_MM is a function that computes the MM for the folders"""
    if mm_computation_backend == 'torch':
        assert mm_model is not None, "The mm_model should be provided (loaded in MuellerMatrixProcessor) when using the torch backend."
        MM, time_data_loading_GPU, times_MM_computation = utils.preprocess_intensities(parameters = None, mm_model = mm_model,
                                                                                     times = {}, amat = A, wmat = W, frame = I,
                                                                                     lu_chipman_backend = lu_chipman_backend,
                                                                                     lu_chipman_model = lu_chipman_model)
    else:
        start_processing = time.time() 
        MM, dilated_mask, times_MM_computation = utils.process_mm(I, remove_reflection, A, W, lu_chipman_backend, lu_chipman_model)
        if dilated_mask is not None:
            MM['dilated_mask'] = dilated_mask
        else:
            MM['dilated_mask'] = None
        time_data_loading_GPU = 0
        times_MM_computation['total'] = time.time() - start_processing
        
    return MM, {'time_data_loading_GPU': time_data_loading_GPU, 'time_MM_processing': times_MM_computation}

def curate_mm(MM, path_save, instrument, workflow_mode = 'default', angle_correction = 0, folder_name = None):
    """curate_mm is a function that curates the MM for the folders"""
    start_processing = time.time()  
    
    # remove the NaNs from the atzimuth measurements
    MM['azimuth'], MM['azimuth_curation'] = utils.curate_azimuth(MM['azimuth'], folder_name = folder_name)
    MM['M11'] = utils.correct_M11(MM['M11'], instrument)
    try:
        MM['M11_normalized'] = utils.normalize_M11(MM['M11'], instrument)
    except:
        traceback.print_exc()
        pass
    time_azimuth_curation = time.time() - start_processing
    
    start_processing = time.time()
    azimuth_std = azimuth_local_var.get_azimuth_local_var(azimuth = MM['azimuth'], patch_size = 5)
    MM['azimuth_local_var'] = azimuth_std.cpu().numpy()
    time_azimuth_std_processing = time.time() - start_processing

    start_save_MM = time.time()
    parameter_names = utils.load_parameter_names(workflow_mode)
    if path_save is None:
        pass
    else:
        if os.path.isdir(path_save):
            utils.save_file_as_npy(MM['nM'], os.path.join(path_save, "nM.npy"))
    time_save_nM = time.time() - start_save_MM
    
    # Remove keys from MM that are not in parameter_names
    MM = {key: value for key, value in MM.items() if key in parameter_names}
    
    start_rotation = time.time()
    if angle_correction != 0:
        MM = rotate_MM.apply_angle_correction(MM, angle_correction)
    time_rotation = time.time() - start_rotation
    
    times = {'azimuth_curation': time_azimuth_curation, 'azimuth_std_processing': time_azimuth_std_processing,
             'time_save_nM': time_save_nM, 'time_rotation': time_rotation}
    
    return MM, times




    



