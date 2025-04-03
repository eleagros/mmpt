import os
from tqdm import tqdm
from processingmm import libmpMuelMat
from processingmm.utils import get_intensity
from processingmm import libmpMPIdenoisePDDN
import numpy as np


def denoise_intensities(parameters: dict, PDDN_models: dict, to_process: list) -> None:
    """
    denoises the intensities for the given wavelengths

    Parameters
    ----------
    parameters: dict
        the processing parameters in a dictionary format
    to_process: list
        the list of folders to process
    
    Returns
    -------
    None
    
    Raises
    ------
    None
    """        
    print('Denoising inference...')
    for folder in tqdm(to_process):
        for wavelength, model in PDDN_models.items():
            if folder['wavelength'].replace('nm', '') != str(wavelength):
                continue
            
            pathDenoised = folder['path_intensite'].replace('Intensite', 'Intensite_PDDN') if parameters['instrument'] == 'IMP' else folder['path_intensite'].replace('Image', 'Image_PDDN')

            if os.path.exists(pathDenoised):
                folder['path_intensite'] = pathDenoised
            else:
                I = get_intensity(parameters, folder['path_intensite'])
                if parameters['denoise_patch']:
                    I = denoise_patch(I, model)
                else:
                    I, _ = model.Denoise(I)
                    
                if pathDenoised.endswith('.npy'):
                    np.save(pathDenoised, I)
                else:
                    libmpMuelMat.write_cod_data_X3D(I, pathDenoised, VerboseFlag=1)
                folder['path_intensite'] = pathDenoised
                
            folder['polarimetry_fname'] = 'polarimetry_PDDN'
    print('Denoising inference done.')
    print()
    
def load_PDDN_models(parameters: dict) -> dict:
    """
    loads the PDDN models for the given wavelengths
    
    Parameters
    ----------
    parameters: dict
        the processing parameters in a dictionary format
    
    Returns
    -------
    PDDN_models : dict 
        the PDDN models for the given wavelengths
    
    Raises
    ------
    AssertionError
        if the PDDN models do not exist
    """    
    PDDN_models = {}

    print()
    print('Loading PDDN models...')
    for wavelength in parameters['wavelengths']:
        path_model = os.path.join(parameters['PDDN_models_path'], 'PDDN_model_' + str(wavelength) + '_Fresh_HB.pt')
        if os.path.isfile(path_model):
            PDDN_models[wavelength] = libmpMPIdenoisePDDN.MPI_PDDN(path_model)
                
    assert len(PDDN_models) > 0, ("Problem when loading the PDDN models. Do they exist? They should be located in ./src/processingmm/PDDN_model/")
    print('Loading PDDN models done.\n')
    print()

    return PDDN_models


def denoise_patch(I, model):
    height, width = I.shape[0], I.shape[1]  # assuming I is in the format (batch, channels, height, width)
    half_height = height // 2
    half_width = width // 2
    patches = [
        I[:half_height, :half_width],      # top-left
        I[half_height:, :half_width],      # bottom-left
        I[:half_height, half_width:],      # top-right
        I[half_height:, half_width:]       # bottom-right
    ]
                    
    denoised_patches = []
    for i, patch in enumerate(patches):
        patch_denoised, _ = model.Denoise(patch)
        denoised_patches.append(patch_denoised)
                    
    top_row = np.concatenate([denoised_patches[0], denoised_patches[1]])  # Concatenate along width (axis 2)
    bottom_row = np.concatenate([denoised_patches[2], denoised_patches[3]])  # Concatenate along width (axis 2)
    I_denoised = np.concatenate([top_row, bottom_row], axis=1)  # Concatenate along height (axis 1)
    return I_denoised