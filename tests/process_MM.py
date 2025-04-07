def main():
    
    import os
    import sys
    from mmpt import processingmm
    
    
    # set the parameters to run the script
    data_source = 'offline'
    input_dirs = [os.path.join(os.path.dirname(__file__), "data_IMP")]
    calib_dir = os.path.join(os.path.dirname(__file__), "calib")
        
    visualization_preset = 'default'
    force_reprocess = True
    
    lu_chipman_backend = 'prediction'
    workflow_mode = 'default'
    save_pdf_figs = False
    
    PDDN_mode = sys.argv[1]
   

    # define which intrument is being used (currently supported 'IMP', 'IMP_v2')
    instrument = sys.argv[2]
    
    input_dirs = [os.path.join(os.path.dirname(__file__), "data_IMP") if instrument == 'IMP' else os.path.join(os.path.dirname(__file__), "data_IMPv2")]
    if instrument == 'IMP':
        wavelengths = [550, 600]
    else:
        wavelengths = [630]
        
    mm_computation_backend = 'c'
    denoise_patch = True
    align_wls = False
    remove_reflection = False

    mmProcessor = processingmm.MuellerMatrixProcessor(data_source = data_source, wavelengths = wavelengths, input_dirs = input_dirs,
                                    calib_dir = calib_dir, visualization_preset = visualization_preset, 
                                    PDDN_mode = PDDN_mode, instrument = instrument, remove_reflection = remove_reflection,
                                    workflow_mode = workflow_mode, force_reprocess = force_reprocess, save_pdf_figs = save_pdf_figs, 
                                    align_wls = align_wls, denoise_patch = denoise_patch, mm_computation_backend = mm_computation_backend,
                                    lu_chipman_backend = lu_chipman_backend)
    
    mmProcessor.batch_process_master()


if __name__ == "__main__":
    main()