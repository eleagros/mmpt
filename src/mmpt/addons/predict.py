import os
import numpy as np
import torch
from omegaconf import OmegaConf
import time
from tqdm import tqdm

from mmpt import utils
from mmpt import libmpMuelMat
from mmpt.addons.polarpred.mm.models import init_mm_model
from mmpt.addons.polarpred.multi_loss import reduce_htgm
from mmpt.addons.polarpred import save_results


def load_models(cfg):
    # model selection
    mm_model = init_mm_model(cfg, train_opt=False)
    model, model_path = load_model(mm_model, cfg)
    return model, mm_model, model_path
    
    
def prediction(parameters, MM = False, mode = 'normal'):
    
    if mode != 'test':
        times, samples = batch_prediction(parameters, MM = MM, mode = mode)
    else:
        print(' [inf] Running test mode with the different models.')
        for mm in [True, False]:
            for id_model in range(0,3):
                times, samples = batch_prediction(parameters, MM = mm, mode = 'fast', id_model = id_model)
        
    return times, samples
        
def batch_prediction(parameters, MM = False, mode = 'normal', id_model = 0):
    
    times = {}
    times['preparation'] = {'load_config': [], 'load_models': []}
    
    start = time.time()  
    parameters['force_reprocess'] = True
    to_process, _ = utils.get_measurements_to_process(parameters)

    wl = '550nm' if parameters['instrument'] == 'IMP' else '630nm'
    
    samples = []
    for entry in to_process:
        if entry['wavelength'] == wl:
            samples.append((entry['folder_name'], entry['path_intensite']))
        
    cfg = OmegaConf.load(os.path.join(utils.getPolarPredPath(), 'configs/train_local.yml'))
    cfg = OmegaConf.merge(cfg, OmegaConf.load(os.path.join(utils.getPolarPredPath(), 'configs/test.yml')))
    cfg.MM = MM
    cfg.id_model = id_model
    times['preparation']['load_config'] = time.time() - start
    
    start_models = time.time()    
    # model selection
    model, mm_model, model_path = load_models(cfg)
    times['preparation']['load_models'] = time.time() - start_models
    
    times['pre_process'] = {'load_cod': [], 'load_calib': [], 'switch_cuda': [], 'get_tensor': [], 'total': []}
    times['predict'] = []
    times['save'] = []
    
    start_processing = time.time()
    
    for sample, path_intensite in tqdm(samples):
        
        path_output = os.path.join(sample, 'predictions')
        os.makedirs(path_output, exist_ok=True)
    
        input, times['pre_process'] = utils.preprocess_intensities(parameters, mm_model, times['pre_process'], sample = sample, 
                                                                   predict = True, path_intensite = path_intensite)
            
        start_predict = time.time()
        preds = predict(model, input)
        times['predict'].append(time.time() - start_predict)
            
        start_save = time.time()
        save_results.save_predictions(preds, input, sample, mode = mode, model_path = model_path, path_intensite = path_intensite)
        times['save'].append(time.time() - start_save)
                
    times['total'] = time.time() - start_processing
    times['per_sample'] = times['total'] / len(samples)
    
    times["pre_process"]["load_cod"] = np.mean(times["pre_process"]["load_cod"])
    times["pre_process"]["load_calib"] = np.mean(times["pre_process"]["load_calib"])
    times["pre_process"]["switch_cuda"] = np.mean(times["pre_process"]["switch_cuda"])
    times["pre_process"]["get_tensor"] = np.mean(times["pre_process"]["get_tensor"])
    times["pre_process"]["total"] = np.mean(times["pre_process"]["total"])
    times["predict"] = np.mean(times["predict"])
    times["save"] = np.mean(times["save"])
    return times, samples


def predict(model, input):
    preds = model(input)
    preds = reduce_htgm(preds)
    return preds

    
def load_model(mm_model, cfg):
    
    n_channels = mm_model.ochs if cfg.data_subfolder.__contains__('raw') else len(cfg.feature_keys)
    if cfg.model == 'unet':
        from mmpt.addons.polarpred.segment_models.unet import UNet
        model = UNet(n_channels=n_channels, n_classes=cfg.class_num+cfg.bg_opt, shallow=cfg.shallow)
    else:
        raise Exception('Model %s not recognized' % cfg.model)

    model = model.to(memory_format=torch.channels_last)
    model.to(device=cfg.device)
    model.eval()
    
    models_path = os.path.join(utils.getPolarPredPath(), 'ckpts', 'MM') if cfg.MM else os.path.join(utils.getPolarPredPath(), 
                                                                                        'ckpts', 'LuChipman')
    model_path = os.path.join(models_path, os.listdir(models_path)[cfg.id_model])
        
    state_dict = torch.load(os.path.join(utils.getPolarPredPath(), model_path), map_location=cfg.device)
    model.load_state_dict(state_dict) if cfg.model != 'resnet' else model.model.load_state_dict(state_dict)    
    return model, model_path
