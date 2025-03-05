import os
from tqdm import tqdm
from processingmm import libmpMuelMat
from processingmm.utils import get_intensity

try:
    from processingmm import libmpMPIdenoisePDDN
except:
    print('PDDN not available. Please install the PDDN package to use it.')
    
    
def denoise_intensities(parameters: dict, to_process: list) -> None:
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
    PDDN_models = load_PDDN_models(parameters)
    
    print('Denoising inference...')
    for folder in tqdm(to_process):
        for wavelength, model in PDDN_models.items():
            pathDenoisedCod = os.path.join(f"{folder['folder_name']}", "raw_data", f"{wavelength}nm", f"{wavelength}_Intensite_PDDN.cod")
            print(pathDenoisedCod)
            if os.path.exists(pathDenoisedCod):
                pass
            else:
                if os.path.exists(os.path.join(f"{folder['folder_name']}", "raw_data", f"{wavelength}nm")):
                    if len(os.listdir(os.path.join(f"{folder['folder_name']}", "raw_data", f"{wavelength}nm"))) > 0:
                        I, _ = get_intensity(f"{folder['folder_name']}", f"{str(wavelength)}nm", False, False)
                        I, _ = model.Denoise(I)
                        print('here')
                        libmpMuelMat.write_cod_data_X3D(I, os.path.join(os.path.join(f"{folder['folder_name']}", "raw_data"), 
                                                            f"{wavelength}nm", f"{wavelength}_Intensite_PDDN.cod"), VerboseFlag=1)
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

